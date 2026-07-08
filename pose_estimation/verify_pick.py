"""verify_pick.py — offline check of a robot_pick.py debug dump (.ply + .npz).

Answers, in order:
  1. VISION:      does the grasp point actually lie on the parcel point cloud?
  2. MATH:        does T_base_cam @ position_cam reproduce the commanded pick_base?
  3. CALIBRATION: (optional) how far is pick_base from where the robot REALLY
                  touches? Jog the suction tip onto the pick spot, read the TCP
                  xyz off the pendant, and pass it with --tcp.

Usage:
  python verify_pick.py                          # latest debug_pick_* in cwd
  python verify_pick.py debug_pick_20260708_140102
  python verify_pick.py --tcp -0.2634 -0.4337 0.2425
"""

import argparse
import glob
import sys

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="?",
                    help="dump prefix, e.g. debug_pick_20260708_140102 (default: latest)")
    ap.add_argument("--tcp", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="pendant TCP xyz (m) with the tip touching the pick spot")
    ap.add_argument("--no-view", action="store_true", help="skip the Open3D window")
    args = ap.parse_args()

    prefix = args.prefix or max(glob.glob("debug_pick_*.npz"), default=None)
    if prefix is None:
        sys.exit("No debug_pick_*.npz found — run robot_pick.py first.")
    prefix = prefix[:-4] if prefix.endswith(".npz") else prefix
    print(f"Checking {prefix}.npz / .ply\n")

    d = np.load(f"{prefix}.npz")
    p_cam, n_cam = d["position_cam"], d["normal_cam"]
    T_base_cam, pick_base = d["T_base_cam"], d["pick_base"]

    pcd = o3d.io.read_point_cloud(f"{prefix}.ply")
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        sys.exit("Point cloud is empty — capture was bad; rerun robot_pick.py.")

    # 1. VISION: grasp point should sit on the parcel surface.
    surf_mm = float(np.linalg.norm(pts - p_cam, axis=1).min()) * 1000
    ok = surf_mm < 10
    print(f"[1 vision]  grasp point to nearest cloud point: {surf_mm:.1f} mm  "
          f"{'[OK]' if ok else '[SUSPECT: generated coordinates are off the parcel]'}")

    # 2. MATH: recompute the base-frame point robot_pick.py commanded.
    recomputed = (T_base_cam @ np.append(p_cam, 1.0))[:3]
    math_mm = float(np.linalg.norm(recomputed - pick_base)) * 1000
    ok = math_mm < 0.1
    print(f"[2 math]    T_base_cam @ position_cam vs saved pick_base: {math_mm:.3f} mm  "
          f"{'[OK]' if ok else '[SUSPECT: transform bug in robot_pick.py]'}")
    print(f"            commanded pick point (base): {np.round(pick_base, 4)}")

    # 3. CALIBRATION: commanded point vs where the robot physically touches.
    if args.tcp is not None:
        delta = np.array(args.tcp) - pick_base
        err_mm = float(np.linalg.norm(delta)) * 1000
        ok = err_mm < 10
        print(f"[3 calib]   measured TCP - pick_base: "
              f"dx={delta[0]*1000:+.1f} dy={delta[1]*1000:+.1f} dz={delta[2]*1000:+.1f} mm  "
              f"|err|={err_mm:.1f} mm  "
              f"{'[OK]' if ok else '[SUSPECT: hand-eye calibration is stale/off]'}")
    else:
        print("[3 calib]   skipped — jog the tip onto the pick spot and rerun with --tcp X Y Z")

    if args.no_view:
        return

    # 3D view: gray parcel cloud, red sphere = grasp point, blue line = approach.
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere.translate(p_cam)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([p_cam, p_cam - n_cam * 0.08])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]])
    print("\nOpening Open3D view (gray=parcel, red=grasp point, blue=approach)...")
    o3d.visualization.draw_geometries([pcd, sphere, line], window_name=prefix)


if __name__ == "__main__":
    main()
