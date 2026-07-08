"""Self-check for grasp-patch selection: `python test_grasp_selection.py`.

box rule (flat_tol=inf)  -> picks the CENTER of a flat box top
bag rule (flat_tol=5mm)  -> picks the most central patch that is actually flat,
                            skipping a wrinkled region
"""

import numpy as np
import open3d as o3d

from pose import GraspPoseEstimator, CLASS_FLAT_TOL, FLAT_TOL_M


def cloud(z):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([X, Y, z], axis=1))
    return pcd


rng = np.random.default_rng(0)
xs, ys = np.meshgrid(np.linspace(-0.10, 0.10, 80), np.linspace(-0.075, 0.075, 60))
X, Y = xs.ravel(), ys.ravel()
est = GraspPoseEstimator(fx=1489.5, fy=1487.7, cx=963.4, cy=542.0)

# flat box top (1 mm sensor noise): box rule must grasp the center
flat = cloud(0.45 + rng.normal(0, 0.001, X.size))
r = est.find_flattest_patch(flat, flat_tol=CLASS_FLAT_TOL["box"])
d_mm = np.linalg.norm(r['position'][:2]) * 1000
assert d_mm < 10, f"box rule picked {d_mm:.1f} mm off center"
print(f"box rule:    {d_mm:.1f} mm from center  [OK]")

# crumpled bag: 8 mm wrinkles across the center, flat outside — bag rule must
# land off the wrinkles on a genuinely flat patch
wrinkle = np.where(np.abs(X) < 0.04,
                   0.008 * np.sin(X * 2 * np.pi / 0.01) * np.sin(Y * 2 * np.pi / 0.01), 0.0)
bag = cloud(0.45 + rng.normal(0, 0.0005, X.size) + wrinkle)
r = est.find_flattest_patch(bag, flat_tol=FLAT_TOL_M)
assert abs(r['position'][0]) > 0.04, f"bag rule grasped the wrinkles at {r['position']}"
assert r['resid_p95'] <= FLAT_TOL_M
print(f"bag rule:    grasp x={r['position'][0]:.3f} (wrinkles span |x|<0.04), "
      f"p95 resid {r['resid_p95']*1000:.2f} mm  [OK]")
print("all checks passed")
