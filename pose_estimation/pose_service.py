"""
pose_service.py — parcel pose estimation as a FastAPI service (ML side ONLY).

This service does NOT talk to the robot. The robot side calls POST /pose;
we capture one OAK-D frame, detect the parcel, estimate the grasp point, and
return it in both the camera frame and the robot BASE frame (transformed with
our hand-eye calibration). What the robot does with it is the robot side's job.

Run:
    uvicorn pose_service:app --host 0.0.0.0 --port 8000

API contract for consumers: see POSE_SERVICE_API.md (also /docs in a browser).
"""

import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException

from ultralytics import YOLO, SAM
from pose import GraspPoseEstimator, DEPTH_SCALE
from robot_pick import (
    load_handeye,
    run_vision,
    pick_point_base,
    HANDEYE_NPZ,
    MODEL_PATH,
    SAM_MODEL_PATH,
)

app = FastAPI(title="Parcel pose estimation service")
_busy = threading.Lock()  # one camera -> one capture at a time

# Loaded once at startup (model load is seconds; per-request would be minutes).
T_BASE_CAM, K, _D = load_handeye(HANDEYE_NPZ)
MODEL = YOLO(MODEL_PATH)
SAM_MODEL = SAM(SAM_MODEL_PATH)
ESTIMATOR = GraspPoseEstimator(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                               depth_scale=DEPTH_SCALE)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/pose")
def estimate_pose():
    """Capture one frame and return the grasp point, or detected=false."""
    if not _busy.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="capture already in progress")
    try:
        debug_prefix = time.strftime("debug_pick_%Y%m%d_%H%M%S")
        try:
            pose = run_vision(MODEL, SAM_MODEL, ESTIMATOR, debug_prefix)
        except RuntimeError:
            # OAK-D firmware crash: the watchdog reboots the device; retry once.
            time.sleep(5)
            try:
                pose = run_vision(MODEL, SAM_MODEL, ESTIMATOR, debug_prefix)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=f"camera error: {exc}")

        if pose is None:
            return {"detected": False,
                    "message": "no parcel detected or no valid grasp pose"}

        pick_base = pick_point_base(T_BASE_CAM, pose["position"])
        normal_base = T_BASE_CAM[:3, :3] @ pose["normal"]
        np.savez(f"{debug_prefix}.npz",
                 position_cam=pose["position"], normal_cam=pose["normal"],
                 T_base_cam=T_BASE_CAM, pick_base=pick_base)

        # API positions/lengths are in MILLIMETERS (robot side asked for mm).
        # Internal math and the debug .npz stay in meters.
        mm = lambda vec: [round(float(v) * 1000, 1) for v in vec]
        return {
            "detected": True,
            "class_name": pose["class_name"],
            "confidence": round(float(pose["confidence"]), 3),
            "pick_base": mm(pick_base),
            "normal_base": [round(float(v), 4) for v in normal_base],
            "position_cam": mm(pose["position"]),
            "normal_cam": [round(float(v), 4) for v in pose["normal"]],
            "flatness_mm": round(float(pose["flatness_score"]) * 1000, 2),
            "inliers": int(pose["inlier_count"]),
            "debug_prefix": debug_prefix,
        }
    finally:
        _busy.release()
