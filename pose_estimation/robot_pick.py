"""
robot_pick.py — one-shot pick move for the UR5.

Flow (single pass, then stop):
  1. Load hand-eye calibration (handeye_result.npz): T_base_cam, K, D.
  2. Capture one OAK-D RGB + depth frame at 1920x1080 (matches the calibrated K),
     then CLOSE the camera before touching the robot.
  3. Detect the parcel (YOLO-OBB) -> SAM mask -> grasp pose in the CAMERA frame
     (reuses the estimator in pose.py); take only its position.
  4. Transform the pick POINT into the robot base frame via T_base_cam.
  5. Move to it with a collision-safe staged path at FIXED orientation:
        down (in place) -> over (above the pick) -> down (to a vertical standoff).
     The tool orientation is held at whatever the robot currently has — no
     reorientation (a vacuum cup seals on a straight-down approach). Each segment
     is verified (moveL result + protective-stop state); the robot stops at the
     standoff and reports whether it actually reached the target.

This moves a real robot. No place motion and no gripper/suction actuation are
performed. The TCP is assumed to be configured on the pendant at the suction tip.
"""

import numpy as np
import depthai as dai
from ultralytics import YOLO, SAM

from pose import (
    GraspPoseEstimator,
    mask_from_sam,
    mask_from_obb,
    CLASS_NAMES,
    DEPTH_SCALE,
    SUCTION_PATCH_RADIUS,
)


# ============================================================
# CONFIG
# ============================================================
ROBOT_IP = "192.168.1.10"
HANDEYE_NPZ = "handeye_result.npz"
MODEL_PATH = "best.pt"
SAM_MODEL_PATH = "sam2.1_t.pt"

# Must match the resolution the calibrated K corresponds to (1920x1080).
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

STANDOFF_M = 0.10          # stop this far ABOVE the pick (base +Z), vertical approach
MOVE_SPEED = 0.10          # m/s   (conservative)
MOVE_ACCEL = 0.30          # m/s^2 (conservative)

# Safe horizontal-traverse height (base-frame Z). MUST clear the camera/stand
# (camera sits ~0.695 m above the base) AND stay above the standoff. Tune this and
# confirm it in a dry run before trusting it.
Z_TRAVERSE_M = 0.50

# Soft reachability envelope (UR5 max reach ~0.85 m). Purely a sanity guard so we
# never command a wildly wrong target; not a substitute for the robot's own limits.
MAX_REACH_M = 0.85
Z_MIN_M, Z_MAX_M = -0.30, 1.00


# ============================================================
# Calibration
# ============================================================
def load_handeye(path):
    data = np.load(path)
    T_base_cam = data["T_base_cam"].astype(np.float64)
    K = data["K"].astype(np.float64)
    D = data["D"].astype(np.float64)
    return T_base_cam, K, D


# ============================================================
# Camera pipeline (1920x1080, depth aligned to CAM_A) — mirrors pose.py/core
# ============================================================
def create_pipeline(device: dai.Device):
    pipeline = dai.Pipeline(device)

    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.build(boardSocket=dai.CameraBoardSocket.CAM_A)

    cam_left = pipeline.create(dai.node.Camera)
    cam_left.build(boardSocket=dai.CameraBoardSocket.CAM_B)

    cam_right = pipeline.create(dai.node.Camera)
    cam_right.build(boardSocket=dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(FRAME_WIDTH, FRAME_HEIGHT)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    cam_left.requestOutput((640, 400)).link(stereo.left)
    cam_right.requestOutput((640, 400)).link(stereo.right)

    rgb_out = cam_rgb.requestOutput((FRAME_WIDTH, FRAME_HEIGHT))
    depth_out = stereo.depth
    return pipeline, rgb_out, depth_out


def capture_and_detect(rgb_queue, depth_queue, model, warmup=15):
    """Grab a few frames so auto-exposure settles, then detect the best parcel."""
    rgb_frame = depth_frame = None
    for _ in range(warmup):
        rgb_frame = rgb_queue.get().getCvFrame()
        depth_frame = depth_queue.get().getFrame()  # uint16 mm

    results = model(rgb_frame, verbose=False)
    obb = results[0].obb
    if obb is None or len(obb.cls) == 0:
        return rgb_frame, depth_frame, None, None, 0.0

    best = int(np.argmax(obb.conf.cpu().numpy()))
    corners = obb.xyxyxyxy[best].cpu().numpy().reshape(4, 2)
    cls_name = CLASS_NAMES.get(int(obb.cls[best]), str(int(obb.cls[best])))
    conf = float(obb.conf[best])
    return rgb_frame, depth_frame, corners, cls_name, conf


def run_vision(model, sam_model, estimator):
    """Capture one frame, close the camera, and return the pick point (camera frame)."""
    device = dai.Device()
    try:
        pipeline, rgb_out, depth_out = create_pipeline(device)
        rgb_queue = rgb_out.createOutputQueue()
        depth_queue = depth_out.createOutputQueue()
        pipeline.start()
        print("\nCapturing frame and detecting parcel...")
        rgb_frame, depth_frame, corners, cls_name, conf = capture_and_detect(
            rgb_queue, depth_queue, model
        )
    finally:
        device.close()  # release the camera before we touch the robot
    print("Camera closed.")

    if corners is None:
        print("No parcel detected — aborting. Robot NOT moved.")
        return None
    print(f"Detected: {cls_name} (conf={conf:.3f})")

    mask = mask_from_sam(sam_model, rgb_frame, corners)
    if mask is None or mask.sum() < 50:
        print("  [warn] SAM returned an empty mask — falling back to OBB polygon")
        mask = mask_from_obb(corners, rgb_frame.shape)

    pose, _ = estimator.estimate(
        depth_frame, mask, corners, patch_radius=SUCTION_PATCH_RADIUS
    )
    if pose is None:
        print("No valid grasp pose found — aborting. Robot NOT moved.")
        return None

    print("\nGrasp pose (camera frame):")
    print(f"  position (m): {np.round(pose['position'], 4)}")
    print(f"  normal:       {np.round(pose['normal'], 4)}  (orientation NOT used)")
    print(f"  flatness:     {pose['flatness_score']:.5f}  inliers: {pose['inlier_count']}")
    return pose


# ============================================================
# Geometry
# ============================================================
def pick_point_base(T_base_cam, position_cam):
    """Transform a camera-frame point (meters) into the robot base frame."""
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(position_cam, dtype=np.float64)
    return (T_base_cam @ p)[:3]


def within_envelope(xyz):
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    radius = float(np.hypot(x, y))
    ok = (radius <= MAX_REACH_M) and (Z_MIN_M <= z <= Z_MAX_M)
    return ok, radius


# ============================================================
# Robot motion
# ============================================================
def _is_protective_stopped(rtde_r):
    fn = getattr(rtde_r, "isProtectiveStopped", None)
    try:
        return bool(fn()) if fn is not None else False
    except Exception:
        return False


def move_segment(rtde_c, rtde_r, label, wp):
    """Blocking moveL for one waypoint; verify it completed and wasn't stopped."""
    print(f"  [{label}] moveL -> xyz=[{wp[0]:.4f}, {wp[1]:.4f}, {wp[2]:.4f}]")
    try:
        ok = rtde_c.moveL(list(wp), MOVE_SPEED, MOVE_ACCEL)
    except Exception as exc:  # controller-side stop / disconnect
        print(f"  [{label}] moveL raised: {exc}")
        return False, None

    stopped = _is_protective_stopped(rtde_r)
    achieved = rtde_r.getActualTCPPose()
    if (ok is False) or stopped:
        print(f"  [{label}] FAILED (moveL returned {ok}, protective_stop={stopped})")
        print(f"           achieved TCP: {np.round(achieved, 4)}")
        return False, achieved
    print(f"  [{label}] done. TCP: {np.round(achieved, 4)}")
    return True, achieved


def build_waypoints(cur_pose, standoff_base):
    """
    Staged path at FIXED orientation (held from cur_pose). Every segment is strictly
    axis-aligned (pure vertical or pure horizontal) so the arm never sweeps a diagonal:
      W1 move VERTICALLY at the current XY to the traverse height (up or down),
      W2 traverse horizontally to above the pick at the traverse height,
      W3 descend vertically to the standoff.
    """
    ori = [float(cur_pose[3]), float(cur_pose[4]), float(cur_pose[5])]
    cx, cy = float(cur_pose[0]), float(cur_pose[1])
    sx, sy, sz = float(standoff_base[0]), float(standoff_base[1]), float(standoff_base[2])

    waypoints = [
        ("W1 vertical-to-traverse", [cx, cy, Z_TRAVERSE_M, *ori]),
        ("W2 traverse-above-pick", [sx, sy, Z_TRAVERSE_M, *ori]),
        ("W3 descend-to-standoff", [sx, sy, sz, *ori]),
    ]
    return waypoints, ori


# ============================================================
# Main
# ============================================================
def main():
    T_base_cam, K, D = load_handeye(HANDEYE_NPZ)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print("Loaded hand-eye calibration:")
    print(f"  intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  (D loaded, not applied)")
    print(f"  T_base_cam:\n{np.array2string(T_base_cam, precision=4)}")

    print(f"Loading YOLO: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Loading SAM: {SAM_MODEL_PATH}")
    sam_model = SAM(SAM_MODEL_PATH)
    estimator = GraspPoseEstimator(fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=DEPTH_SCALE)

    pose = run_vision(model, sam_model, estimator)
    if pose is None:
        return

    pick_base = pick_point_base(T_base_cam, pose["position"])
    standoff_base = np.array([pick_base[0], pick_base[1], pick_base[2] + STANDOFF_M])
    print(f"\nPick point (base frame):    {np.round(pick_base, 4)}")
    print(f"Standoff ({STANDOFF_M * 100:.0f} cm above): {np.round(standoff_base, 4)}")

    if standoff_base[2] > Z_TRAVERSE_M:
        print(f"Standoff z={standoff_base[2]:.3f} m is ABOVE the traverse height "
              f"Z_TRAVERSE_M={Z_TRAVERSE_M:.3f} m — misconfigured. Robot NOT moved.")
        return

    # The camera's base-frame height is the translation in T_base_cam. The traverse
    # height must stay clearly below it so a vertical/horizontal move can't reach it.
    cam_height = float(T_base_cam[2, 3])
    if Z_TRAVERSE_M > cam_height - 0.05:
        print(f"Z_TRAVERSE_M={Z_TRAVERSE_M:.3f} m is too close to the camera height "
              f"{cam_height:.3f} m — lower it. Robot NOT moved.")
        return

    # Connect to read robot state and build the path. No motion happens until moveL.
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    print(f"\nConnecting to UR5 at {ROBOT_IP} (read-only until you confirm)...")
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    try:
        cur_pose = rtde_r.getActualTCPPose()
        print(f"Current TCP pose: {np.round(cur_pose, 4)}")
        if _is_protective_stopped(rtde_r):
            print("Robot is already in a PROTECTIVE STOP — clear it on the pendant first. "
                  "Robot NOT moved.")
            return

        waypoints, ori = build_waypoints(cur_pose, standoff_base)

        print("\nPlanned staged path (orientation held constant — no reorientation):")
        print(f"  held rotvec (rad): {np.round(ori, 4)}")
        print(f"  speed={MOVE_SPEED} m/s, accel={MOVE_ACCEL} m/s^2, traverse z={Z_TRAVERSE_M} m")
        plan_ok = True
        for label, wp in waypoints:
            ok_wp, radius = within_envelope(wp)
            plan_ok = plan_ok and ok_wp
            flag = "OK" if ok_wp else "OUT OF ENVELOPE"
            print(f"  {label}: xyz=[{wp[0]:.4f}, {wp[1]:.4f}, {wp[2]:.4f}]  r={radius:.3f} m  [{flag}]")
        if not plan_ok:
            print("At least one waypoint is outside the safe envelope — aborting. Robot NOT moved.")
            return

        print("\nConfirm the path above clears the camera and stand before moving.")
        print("  Ensure the workspace is clear of people/obstacles and the e-stop is within reach.")
        print("  The robot must be powered on and in 'Remote Control' mode.")
        ans = input("Type 'yes' to move, anything else to abort: ").strip().lower()
        if ans != "yes":
            print("Aborted by user — robot NOT moved.")
            return

        print("\nExecuting staged move...")
        reached_all = True
        achieved = None
        for label, wp in waypoints:
            ok, achieved = move_segment(rtde_c, rtde_r, label, wp)
            if not ok:
                reached_all = False
                print("Stopping — remaining waypoints skipped.")
                break

        if reached_all and achieved is not None:
            target_xyz = np.array(waypoints[-1][1][:3])
            delta = float(np.linalg.norm(np.array(achieved[:3]) - target_xyz))
            if delta < 0.01:
                print(f"\nREACHED standoff (delta {delta * 1000:.1f} mm).")
            else:
                print(f"\nDID NOT REACH cleanly — off by {delta * 1000:.1f} mm.")
            print(f"Final TCP pose: {np.round(achieved, 4)}")
        else:
            print("\nDID NOT REACH — motion stopped early (see failed segment above).")
            if achieved is not None:
                print(f"Final TCP pose: {np.round(achieved, 4)}")

    finally:
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Done. Robot stopped; exiting.")


if __name__ == "__main__":
    main()
