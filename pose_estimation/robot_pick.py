"""
robot_pick.py — one-shot pick move for the UR5.

Flow (single pass, then stop):
  1. Load hand-eye calibration (handeye_result.npz): T_base_cam, K, D.
  2. Capture one OAK-D RGB + depth frame at 1920x1080 (matches the calibrated K).
  3. Detect the parcel (YOLO-OBB) -> SAM mask -> grasp pose in the CAMERA frame
     (reuses the estimator in pose.py).
  4. Transform the grasp to the ROBOT BASE frame via T_base_cam, offset back along
     the surface normal to a safe STANDOFF pose.
  5. Print the target + a reachability check, ask for explicit confirmation, then
     issue a single blocking moveL. After the move, stop and exit.

This moves a real robot. No place motion and no gripper/suction actuation are
performed. The TCP is assumed to be configured on the pendant at the suction tip.
"""

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation as R
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

STANDOFF_M = 0.10          # retract distance from the contact point, along +normal
MOVE_SPEED = 0.10          # m/s   (conservative)
MOVE_ACCEL = 0.30          # m/s^2 (conservative)

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


# ============================================================
# Camera-frame grasp -> base-frame standoff target
# ============================================================
def build_tool_target_cam(pose, standoff_m):
    """
    Build the desired TCP pose in the CAMERA frame:
      - tool approach axis (+Z) points INTO the surface (= -normal),
      - tool X follows the OBB long edge (from the grasp frame's x-axis),
      - origin retracted by `standoff_m` along +normal (away from the surface).
    Returns a 4x4 homogeneous transform T_cam_tool.
    """
    normal = np.asarray(pose["normal"], dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    z_tool = -normal  # approach direction: into the surface

    # Reference long-axis from the grasp frame's x column; re-orthogonalize vs z_tool.
    x_ref = np.asarray(pose["rotation_matrix"][:, 0], dtype=np.float64)
    x_tool = x_ref - np.dot(x_ref, z_tool) * z_tool
    if np.linalg.norm(x_tool) < 1e-6:
        # Degenerate (x_ref parallel to z_tool): fall back to an arbitrary perpendicular.
        for axis in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
            x_tool = np.cross(axis, z_tool)
            if np.linalg.norm(x_tool) > 1e-6:
                break
    x_tool = x_tool / (np.linalg.norm(x_tool) + 1e-12)

    y_tool = np.cross(z_tool, x_tool)
    y_tool = y_tool / (np.linalg.norm(y_tool) + 1e-12)
    x_tool = np.cross(y_tool, z_tool)  # re-orthogonalize for a clean rotation matrix

    R_cam_tool = np.column_stack([x_tool, y_tool, z_tool])

    # Standoff: move away from the surface (toward the camera) along +normal.
    standoff_cam = np.asarray(pose["position"], dtype=np.float64) + normal * standoff_m

    T = np.eye(4)
    T[:3, :3] = R_cam_tool
    T[:3, 3] = standoff_cam
    return T


def pose_to_movel(T_base_target):
    """4x4 base-frame pose -> UR moveL vector [x, y, z, rx, ry, rz] (m, axis-angle rad)."""
    t = T_base_target[:3, 3]
    rotvec = R.from_matrix(T_base_target[:3, :3]).as_rotvec()
    return [float(t[0]), float(t[1]), float(t[2]),
            float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]


def within_envelope(T_base_target):
    x, y, z = T_base_target[:3, 3]
    radius = float(np.hypot(x, y))
    ok = (radius <= MAX_REACH_M) and (Z_MIN_M <= z <= Z_MAX_M)
    return ok, radius


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

    device = dai.Device()
    pipeline, rgb_out, depth_out = create_pipeline(device)
    estimator = GraspPoseEstimator(fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=DEPTH_SCALE)
    rgb_queue = rgb_out.createOutputQueue()
    depth_queue = depth_out.createOutputQueue()
    pipeline.start()

    try:
        print("\nCapturing frame and detecting parcel...")
        rgb_frame, depth_frame, corners, cls_name, conf = capture_and_detect(
            rgb_queue, depth_queue, model
        )
        if corners is None:
            print("No parcel detected — aborting. Robot NOT moved.")
            return
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
            return

        print("\nGrasp pose (camera frame):")
        print(f"  position (m): {np.round(pose['position'], 4)}")
        print(f"  normal:       {np.round(pose['normal'], 4)}")
        print(f"  flatness:     {pose['flatness_score']:.5f}  inliers: {pose['inlier_count']}")

        T_cam_target = build_tool_target_cam(pose, STANDOFF_M)
        T_base_target = T_base_cam @ T_cam_target
        target = pose_to_movel(T_base_target)

        print(f"\nStandoff target (base frame, {STANDOFF_M * 100:.0f} cm along +normal):")
        print(f"  xyz (m):      [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
        print(f"  rotvec (rad): [{target[3]:.4f}, {target[4]:.4f}, {target[5]:.4f}]")

        ok, radius = within_envelope(T_base_target)
        print(f"  reach radius: {radius:.3f} m (limit {MAX_REACH_M} m), z={target[2]:.3f} m")
        if not ok:
            print("Target is OUTSIDE the safe envelope — aborting. Robot NOT moved.")
            return

        print(f"\nAbout to move the UR5 at {ROBOT_IP} to the standoff pose above the pick.")
        print(f"  speed={MOVE_SPEED} m/s, accel={MOVE_ACCEL} m/s^2, blocking moveL.")
        print("  Ensure the workspace is clear of people/obstacles and the e-stop is within reach.")
        print("  The robot must be powered on and in 'Remote Control' mode.")
        ans = input("Type 'yes' to move, anything else to abort: ").strip().lower()
        if ans != "yes":
            print("Aborted by user — robot NOT moved.")
            return

        # Import here so the dry-run path works even if ur_rtde isn't installed yet.
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        print(f"Connecting to UR5 at {ROBOT_IP}...")
        rtde_c = RTDEControlInterface(ROBOT_IP)
        rtde_r = RTDEReceiveInterface(ROBOT_IP)
        try:
            print(f"Current TCP pose: {np.round(rtde_r.getActualTCPPose(), 4)}")
            print("Moving...")
            rtde_c.moveL(target, MOVE_SPEED, MOVE_ACCEL)
            print(f"Move complete. Achieved TCP pose: {np.round(rtde_r.getActualTCPPose(), 4)}")
        finally:
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()

    finally:
        device.close()
        print("Done. Robot stopped; exiting.")


if __name__ == "__main__":
    main()
