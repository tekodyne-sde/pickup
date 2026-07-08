"""
robot_pick.py — one-shot parcel pick-point estimation (ML side only; NO robot).

Flow (single pass, then stop):
  1. Load hand-eye calibration (handeye_result.npz): T_base_cam, K, D.
  2. Capture one OAK-D RGB + depth frame at 1920x1080 (matches the calibrated K).
  3. Detect the parcel (YOLO-OBB) -> SAM mask -> grasp pose in the CAMERA frame
     (reuses the estimator in pose.py).
  4. Transform the pick POINT into the robot base frame via T_base_cam, print it,
     and save the debug artifacts (.png red-dot image, .ply cloud, .npz coords).

Run as a script for a local end-to-end test of the pipeline. The same pipeline
is exposed to the robot side as an HTTP API by pose_service.py (see
POSE_SERVICE_API.md) — the robot side consumes the pick point and does all
motion itself.
"""

import os
import time

# If the OAK-D firmware crashes (X_LINK_ERROR), depthai's crash-dump collector
# segfaults the whole process inside device.close(). Disable it so a device drop
# raises a normal, retryable exception instead.
os.environ.setdefault("DEPTHAI_DISABLE_CRASHDUMP_COLLECTION", "1")

import numpy as np
import depthai as dai
import open3d as o3d
from ultralytics import YOLO, SAM

from pose import (
    GraspPoseEstimator,
    mask_from_sam,
    mask_from_obb,
    save_annotated_snapshot,
    CLASS_NAMES,
    DEPTH_SCALE,
    SUCTION_PATCH_RADIUS,
)


# ============================================================
# CONFIG
# ============================================================
HANDEYE_NPZ = "handeye_result.npz"
MODEL_PATH = "best.pt"
SAM_MODEL_PATH = "sam2.1_t.pt"

# Must match the resolution the calibrated K corresponds to (1920x1080).
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Stream this long before capturing so autofocus/auto-exposure converge
# (15 frames wasn't enough — captures came out blurry). Raise if still soft.
CAMERA_SETTLE_S = 3.0

# All robot-side concerns (standoff, speed, clearances, path planning) belong
# to the robot team — we only produce the pick point.


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


def capture_and_detect(rgb_queue, depth_queue, model, settle_s=CAMERA_SETTLE_S):
    """Stream frames for settle_s so autofocus/AE converge, then detect on the last."""
    rgb_frame = depth_frame = None
    t_end = time.time() + settle_s
    while rgb_frame is None or time.time() < t_end:
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


def run_vision(model, sam_model, estimator, debug_prefix):
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

    pose, pcd = estimator.estimate(
        depth_frame, mask, corners, patch_radius=SUCTION_PATCH_RADIUS, cls_name=cls_name
    )
    if pose is None:
        print("No valid grasp pose found — aborting. Robot NOT moved.")
        return None
    pose["class_name"] = cls_name
    pose["confidence"] = conf

    print("\nGrasp pose (camera frame):")
    print(f"  position (m): {np.round(pose['position'], 4)}")
    print(f"  normal:       {np.round(pose['normal'], 4)}  (orientation NOT used)")
    print(f"  flatness:     {pose['flatness_score']:.5f}  inliers: {pose['inlier_count']}")

    # Debug artifacts: annotated PNG (red dot = pick pixel) + camera-frame point
    # cloud of the parcel, so the commanded coordinates can be verified offline.
    save_annotated_snapshot(rgb_frame, corners, cls_name, pose,
                            estimator.fx, estimator.fy, estimator.cx, estimator.cy,
                            f"{debug_prefix}.png", mask=mask)
    o3d.io.write_point_cloud(f"{debug_prefix}.ply", pcd)
    print(f"  Saved point cloud        -> {debug_prefix}.ply")
    return pose


# ============================================================
# Geometry
# ============================================================
def pick_point_base(T_base_cam, position_cam):
    """Transform a camera-frame point (meters) into the robot base frame."""
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(position_cam, dtype=np.float64)
    return (T_base_cam @ p)[:3]


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

    debug_prefix = time.strftime("debug_pick_%Y%m%d_%H%M%S")

    # ponytail: one retry — the OAK-D watchdog reboots the device after a
    # firmware crash, so a second open usually succeeds.
    try:
        pose = run_vision(model, sam_model, estimator, debug_prefix)
    except RuntimeError as exc:
        print(f"Camera error: {exc}\nRetrying once in 5 s...")
        time.sleep(5)
        pose = run_vision(model, sam_model, estimator, debug_prefix)
    if pose is None:
        return

    pick_base = pick_point_base(T_base_cam, pose["position"])
    # Everything needed to re-check the math offline: camera-frame grasp point,
    # the transform, and the base-frame point the robot will be commanded to.
    np.savez(f"{debug_prefix}.npz",
             position_cam=pose["position"], normal_cam=pose["normal"],
             T_base_cam=T_base_cam, pick_base=pick_base)
    print(f"Saved coordinate dump -> {debug_prefix}.npz")
    normal_base = T_base_cam[:3, :3] @ pose["normal"]
    print(f"\nPick point (base frame):    {np.round(pick_base, 4)}")
    print(f"Normal (base frame):        {np.round(normal_base, 4)}")
    print("Done — verify with verify_pick.py; the robot side gets this via pose_service.py.")


if __name__ == "__main__":
    main()
