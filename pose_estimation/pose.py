"""
Real-time parcel pick pipeline test — OAK-D camera, no robot.

Live loop:
  - Streams RGB + depth from OAK-D
  - Runs YOLO-OBB (best.pt) on each frame, draws detection overlay
  - Press 's' to snapshot current frame -> run full pose pipeline ->
        print position/normal/quaternion/flatness, save annotated image,
        and open an Open3D 3D view of the point cloud + chosen grasp
        patch + gripper axes for visual verification.
  - Press 'q' to quit.

Fill in DEPTH_SCALE and SUCTION_PATCH_RADIUS below. Camera intrinsics are
pulled automatically from the OAK-D at runtime (matches your existing
camera_pipeline.py: depth aligned to CAM_A, resized to FRAME_WIDTH/HEIGHT).
"""

import time
import threading
import numpy as np
import cv2
import open3d as o3d
import depthai as dai
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO, SAM


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "best.pt"
# ultralytics auto-downloads this on first use (~40MB)
SAM_MODEL_PATH = "sam2.1_t.pt"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# uint16 millimeters -> 1000.0 | already float meters -> 1.0
DEPTH_SCALE = 1000.0

# your suction cup radius in meters (e.g. 0.02 = 2cm)
SUCTION_PATCH_RADIUS = 0.02

CLASS_NAMES = {0: "box", 1: "brown_bag", 2: "white_bag"}

# Patches whose mean plane residual is within this count as "flat"; among flat
# patches the most CENTRAL one wins (stops boxes being picked at an edge).
FLAT_TOL_M = 0.005
# box: flat everywhere -> pure center pick
CLASS_FLAT_TOL = {"box": np.inf, "brown_bag": FLAT_TOL_M, "white_bag": FLAT_TOL_M}
INFER_EVERY_N_FRAMES = 3  # run YOLO every Nth frame for smoother live preview


# ============================================================
# Camera pipeline (mirrors your existing core/camera_pipeline.py)
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


def get_rgb_intrinsics(device: dai.Device):
    calib = device.readCalibration()
    intr = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A,
        resizeWidth=FRAME_WIDTH,
        resizeHeight=FRAME_HEIGHT
    )
    fx = intr[0][0]
    fy = intr[1][1]
    cx = intr[0][2]
    cy = intr[1][2]
    return fx, fy, cx, cy


# ============================================================
# Pose estimator (same geometric core as before)
# ============================================================
class GraspPoseEstimator:
    def __init__(self, fx, fy, cx, cy, depth_scale=1000.0):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.depth_scale = depth_scale

    def deproject_mask_to_pointcloud(self, depth_map, mask):
        ys, xs = np.where(mask)
        depths = depth_map[ys, xs].astype(np.float32) / self.depth_scale
        valid = depths > 0
        ys, xs, depths = ys[valid], xs[valid], depths[valid]

        X = (xs - self.cx) * depths / self.fx
        Y = (ys - self.cy) * depths / self.fy
        Z = depths
        points = np.stack([X, Y, Z], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def denoise(self, pcd, voxel_size=0.003, nb_neighbors=20, std_ratio=2.0):
        pcd = pcd.voxel_down_sample(voxel_size)
        if len(pcd.points) < nb_neighbors:
            return pcd
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd

    def find_flattest_patch(self, pcd, patch_radius=0.02, stride=0.01,
                             min_points=15, plane_thresh=0.003, flat_tol=FLAT_TOL_M):
        points = np.asarray(pcd.points)
        if len(points) < min_points:
            return None

        centroid_xy = points[:, :2].mean(axis=0)
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        candidates = []
        x_range = np.arange(x_min + patch_radius, x_max - patch_radius, stride)
        y_range = np.arange(y_min + patch_radius, y_max - patch_radius, stride)
        if len(x_range) == 0 or len(y_range) == 0:
            x_range = [points[:, 0].mean()]
            y_range = [points[:, 1].mean()]

        for cx in x_range:
            for cy in y_range:
                dist_xy = np.linalg.norm(points[:, :2] - np.array([cx, cy]), axis=1)
                window_mask = dist_xy < patch_radius
                if window_mask.sum() < min_points:
                    continue

                window_points = points[window_mask]
                window_pcd = o3d.geometry.PointCloud()
                window_pcd.points = o3d.utility.Vector3dVector(window_points)

                try:
                    plane_model, inliers = window_pcd.segment_plane(
                        distance_threshold=plane_thresh, ransac_n=3, num_iterations=200
                    )
                except RuntimeError:
                    continue

                a, b, c, d = plane_model
                normal_norm = np.linalg.norm([a, b, c])
                inlier_pts = window_points[inliers]
                residuals = np.abs((inlier_pts @ np.array([a, b, c])) + d) / normal_norm
                flatness_score = residuals.mean()
                # tolerance test uses ALL patch points (inlier-only residuals are
                # capped by plane_thresh, so they'd call any bump "flat")
                resid_all = np.abs((window_points @ np.array([a, b, c])) + d) / normal_norm
                resid_p95 = float(np.percentile(resid_all, 95))
                center_dist = np.linalg.norm(np.array([cx, cy]) - centroid_xy)
                combined_score = flatness_score + 0.1 * center_dist

                candidates.append({
                    'plane_model': (a, b, c, d),
                    'flatness': flatness_score,
                    'resid_p95': resid_p95,
                    'inlier_count': len(inliers),
                    'window_points': window_points[inliers],
                    'combined_score': combined_score,
                    'center_dist': center_dist
                })

        if not candidates:
            return None

        # Among patches flat enough (95% of points within flat_tol of the plane)
        # the most central wins; if none qualify fall back to the old score.
        flat = [c for c in candidates if c['resid_p95'] <= flat_tol]
        if flat:
            best = min(flat, key=lambda c: c['center_dist'])
        else:
            best = min(candidates, key=lambda c: c['combined_score'])
        a, b, c, d = best['plane_model']
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        if normal[2] > 0:
            normal = -normal

        grasp_center_3d = best['window_points'].mean(axis=0)

        return {
            'position': grasp_center_3d,
            'normal': normal,
            'flatness_score': best['flatness'],
            'resid_p95': best['resid_p95'],
            'inlier_count': best['inlier_count'],
            'patch_points': best['window_points']
        }

    def compute_yaw_axis(self, obb_corners_2d, depth_map, normal):
        corners = np.array(obb_corners_2d)
        edge_lengths = [
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[1])
        ]
        if edge_lengths[0] >= edge_lengths[1]:
            p1, p2 = corners[0], corners[1]
        else:
            p1, p2 = corners[1], corners[2]

        def deproj(pt):
            u, v = int(pt[0]), int(pt[1])
            u = np.clip(u, 0, depth_map.shape[1] - 1)
            v = np.clip(v, 0, depth_map.shape[0] - 1)
            z = depth_map[v, u] / self.depth_scale
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy
            return np.array([x, y, z])

        p1_3d, p2_3d = deproj(p1), deproj(p2)
        # OBB corners can land outside the object / in depth holes (z=0) —
        # the axis is then meaningless and would build a null rotation matrix.
        if p1_3d[2] <= 0 or p2_3d[2] <= 0:
            return None
        long_axis = p2_3d - p1_3d
        norm = np.linalg.norm(long_axis)
        if norm < 1e-6:
            return None
        long_axis = long_axis / norm
        long_axis_proj = long_axis - np.dot(long_axis, normal) * normal
        norm = np.linalg.norm(long_axis_proj)
        if norm < 1e-6:
            return None
        return long_axis_proj / norm

    def build_pose(self, position, normal, x_axis):
        z_axis = normal
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        x_axis = np.cross(y_axis, z_axis)

        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        quat = R.from_matrix(rot_matrix).as_quat()

        return {'position': position, 'quaternion': quat, 'rotation_matrix': rot_matrix}

    def estimate(self, depth_map, mask, obb_corners_2d, patch_radius=0.02, full_object_pcd=None,
                 cls_name=None):
        pcd = self.deproject_mask_to_pointcloud(depth_map, mask)
        if len(pcd.points) < 20:
            print("  [FAIL] too few valid depth points in mask")
            return None, pcd

        pcd = self.denoise(pcd)
        flat_tol = CLASS_FLAT_TOL.get(cls_name, FLAT_TOL_M)
        patch_result = self.find_flattest_patch(pcd, patch_radius=patch_radius, flat_tol=flat_tol)
        if patch_result is None:
            print("  [FAIL] no flat patch found meeting minimum point criteria")
            return None, pcd

        x_axis = self.compute_yaw_axis(obb_corners_2d, depth_map, patch_result['normal'])
        if x_axis is None:
            # OBB corners had no usable depth — yaw is arbitrary, so use any
            # axis orthogonal to the normal (the robot uses position only).
            print("  [warn] no depth at OBB corners — using fallback yaw axis")
            n = patch_result['normal']
            ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(n, ref)
            x_axis = x_axis / np.linalg.norm(x_axis)
        pose = self.build_pose(patch_result['position'], patch_result['normal'], x_axis)
        pose['normal'] = patch_result['normal']
        pose['flatness_score'] = patch_result['flatness_score']
        pose['inlier_count'] = patch_result['inlier_count']
        pose['patch_points'] = patch_result['patch_points']

        return pose, pcd


# ============================================================
# Helpers
# ============================================================
def mask_from_obb(corners_2d, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d.astype(np.int32)], 1)
    return mask.astype(bool)


def obb_to_axis_aligned_bbox(corners_2d, image_shape):
    """Tight axis-aligned [x1, y1, x2, y2] around the OBB corners, clamped to the frame."""
    h, w = image_shape[:2]
    xs, ys = corners_2d[:, 0], corners_2d[:, 1]
    x1 = float(np.clip(xs.min(), 0, w - 1))
    y1 = float(np.clip(ys.min(), 0, h - 1))
    x2 = float(np.clip(xs.max(), 0, w - 1))
    y2 = float(np.clip(ys.max(), 0, h - 1))
    return [x1, y1, x2, y2]


def mask_from_sam(sam_model, rgb_frame, corners_2d):
    """
    Segment the detected object with SAM, prompted by the OBB's axis-aligned
    bounding box. Returns a boolean mask hugging the true object boundary, or
    None if SAM produced no mask (caller should fall back to the OBB polygon).
    """
    h, w = rgb_frame.shape[:2]
    bbox = obb_to_axis_aligned_bbox(corners_2d, rgb_frame.shape)

    results = sam_model(rgb_frame, bboxes=[bbox], verbose=False)
    r = results[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None

    m = r.masks.data[0].cpu().numpy().astype(np.uint8)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    # Constrain SAM to the detected instance: intersect with the OBB polygon so
    # a mask that bleeds onto a neighboring object can't leak into the point cloud.
    obb = mask_from_obb(corners_2d, rgb_frame.shape)
    m = np.logical_and(m.astype(bool), obb)
    return m


def project_point(pt_3d, fx, fy, cx, cy):
    x, y, z = pt_3d
    if z <= 0:
        return None
    u = int((x * fx / z) + cx)
    v = int((y * fy / z) + cy)
    return (u, v)


def draw_live_overlay(frame, corners_2d, class_name, conf, fps):
    vis = frame.copy()
    if corners_2d is not None:
        cv2.polylines(vis, [corners_2d.astype(np.int32)], True, (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        origin = tuple(corners_2d[0].astype(int))
        cv2.putText(vis, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"FPS: {fps:.1f}  |  type 'capture' in terminal to trigger pose",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def save_annotated_snapshot(rgb_frame, corners_2d, class_name, pose, fx, fy, cx, cy, out_path, mask=None):
    vis = rgb_frame.copy()

    # cyan outline of the actual mask used for the point cloud (SAM or OBB fallback)
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)

    cv2.polylines(vis, [corners_2d.astype(np.int32)], True, (0, 255, 0), 2)

    if pose is None:
        cv2.putText(vis, f"{class_name}: NO VALID GRASP", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        grasp_uv = project_point(pose['position'], fx, fy, cx, cy)
        if grasp_uv is not None:
            cv2.circle(vis, grasp_uv, 8, (0, 0, 255), -1)
            normal_tip = pose['position'] - pose['normal'] * 0.08
            tip_uv = project_point(normal_tip, fx, fy, cx, cy)
            if tip_uv is not None:
                cv2.arrowedLine(vis, grasp_uv, tip_uv, (255, 0, 0), 2, tipLength=0.3)
        label = f"{class_name} | flatness={pose['flatness_score']:.4f} | pts={pose['inlier_count']}"
        cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite(out_path, vis)
    print(f"  Saved annotated snapshot -> {out_path}")


def show_open3d_view(full_pcd, pose):
    """
    Blocking 3D view: full object point cloud (gray), the chosen flat
    patch (red), and a coordinate frame at the grasp pose showing the
    gripper's approach axes. Close the window to resume the live feed.
    """
    full_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geometries = [full_pcd]

    if pose is not None:
        patch_pcd = o3d.geometry.PointCloud()
        patch_pcd.points = o3d.utility.Vector3dVector(pose['patch_points'])
        patch_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(patch_pcd)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        transform = np.eye(4)
        transform[:3, :3] = pose['rotation_matrix']
        transform[:3, 3] = pose['position']
        frame.transform(transform)
        geometries.append(frame)

        # normal arrow: line from grasp point back along -normal (approach direction)
        approach_point = pose['position'] - pose['normal'] * 0.08
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([pose['position'], approach_point])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]])
        geometries.append(line)

    print("  Opening Open3D view — close the window to resume live feed...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Grasp pose verification (gray=object, red=chosen flat patch, axes=gripper frame, blue=approach)"
    )


# ============================================================
# Terminal trigger (type commands + Enter instead of keyboard focus on video window)
# ============================================================
capture_event = threading.Event()
quit_event = threading.Event()


def terminal_listener():
    print("Type 'capture' (or 'c') + Enter to trigger a snapshot.")
    print("Type 'quit' (or 'q') + Enter to exit.\n")
    while not quit_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd in ("capture", "c"):
            capture_event.set()
        elif cmd in ("quit", "q", "exit"):
            quit_event.set()
        elif cmd:
            print(f"Unknown command: '{cmd}'. Use 'capture' or 'quit'.")


# ============================================================
# Main loop
# ============================================================
def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Loading SAM model: {SAM_MODEL_PATH}")
    sam_model = SAM(SAM_MODEL_PATH)

    device = dai.Device()
    fx, fy, cx, cy = get_rgb_intrinsics(device)
    print(f"Camera intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    pipeline, rgb_out, depth_out = create_pipeline(device)
    estimator = GraspPoseEstimator(fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=DEPTH_SCALE)

    rgb_queue = rgb_out.createOutputQueue()
    depth_queue = depth_out.createOutputQueue()
    pipeline.start()

    frame_count = 0
    last_corners = None
    last_class_name = None
    last_conf = 0.0
    prev_time = time.time()
    snapshot_idx = 0

    listener_thread = threading.Thread(target=terminal_listener, daemon=True)
    listener_thread.start()

    print("Live feed running (video window can stay unfocused).\n")

    try:
        while pipeline.isRunning():
            rgb_msg = rgb_queue.get()
            depth_msg = depth_queue.get()

            rgb_frame = rgb_msg.getCvFrame()
            depth_frame = depth_msg.getFrame()  # uint16, mm (per your StereoDepth config)

            frame_count += 1
            if frame_count % INFER_EVERY_N_FRAMES == 0:
                results = model(rgb_frame, verbose=False)
                obb = results[0].obb
                if obb is not None and len(obb.cls) > 0:
                    best_idx = int(np.argmax(obb.conf.cpu().numpy()))
                    last_class_name = CLASS_NAMES.get(int(obb.cls[best_idx]), str(int(obb.cls[best_idx])))
                    last_conf = float(obb.conf[best_idx])
                    last_corners = obb.xyxyxyxy[best_idx].cpu().numpy().reshape(4, 2)
                else:
                    last_corners = None

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            display = draw_live_overlay(rgb_frame, last_corners, last_class_name, last_conf, fps)
            cv2.imshow("Live feed", display)
            cv2.waitKey(1)  # keep the OpenCV window responsive; not used for triggering

            if quit_event.is_set():
                break

            if capture_event.is_set():
                capture_event.clear()

                if last_corners is None:
                    print("No detection currently available — can't capture.")
                    continue

                print(f"\n--- Capture {snapshot_idx} ---")
                print(f"Detected: {last_class_name} (conf={last_conf:.3f})")

                mask = mask_from_sam(sam_model, rgb_frame, last_corners)
                if mask is None or mask.sum() < 50:
                    print("  [warn] SAM returned an empty mask — falling back to OBB polygon")
                    mask = mask_from_obb(last_corners, rgb_frame.shape)
                else:
                    print(f"  SAM mask: {int(mask.sum())} px "
                          f"(OBB polygon would be {int(mask_from_obb(last_corners, rgb_frame.shape).sum())} px)")

                pose, full_pcd = estimator.estimate(
                    depth_frame, mask, last_corners, patch_radius=SUCTION_PATCH_RADIUS,
                    cls_name=last_class_name
                )

                if pose is None:
                    print("=> NO VALID GRASP FOUND")
                else:
                    print("=> GRASP POSE FOUND")
                    print(f"  Position (m): {pose['position']}")
                    print(f"  Normal: {pose['normal']}")
                    print(f"  Quaternion (x,y,z,w): {pose['quaternion']}")
                    print(f"  Flatness score: {pose['flatness_score']:.5f}")
                    print(f"  Inlier points: {pose['inlier_count']}")

                out_path = f"snapshot_{snapshot_idx}_result.jpg"
                save_annotated_snapshot(rgb_frame, last_corners, last_class_name,
                                         pose, fx, fy, cx, cy, out_path, mask=mask)

                show_open3d_view(full_pcd, pose)
                print("\nReady — type 'capture' again or 'quit' to exit.\n")
                snapshot_idx += 1

    finally:
        quit_event.set()
        cv2.destroyAllWindows()
        device.close()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()