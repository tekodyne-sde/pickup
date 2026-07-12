"""
pose_service.py — parcel pose estimation as a FastAPI service (ML side ONLY).

The camera is ALWAYS LIVE: a background thread streams RGB+depth and runs YOLO
detection continuously, auto-reconnecting if the OAK-D drops. POST /pose takes
the latest live detection, runs SAM + grasp estimation, and returns the pick
point in MILLIMETERS (robot base frame, via our hand-eye calibration). If no
parcel is in view, the request keeps retrying on fresh frames for up to
NO_DETECT_RETRY_S before answering detected=false.

This service does NOT talk to the robot.

Run:
    python pose_service.py
    (equivalent to: uvicorn pose_service:app --host 0.0.0.0 --port 8001)

To keep ML inference off the EtherCAT/driver cores (2-3), pin it to cores
4-11 at launch:
    cmd /c start /affinity FF0 python pose_service.py

API contract for consumers: see POSE_SERVICE_API.md (also /docs in a browser).
"""

import os
import threading
import time
import traceback

os.environ.setdefault("DEPTHAI_DISABLE_CRASHDUMP_COLLECTION", "1")

import cv2
import numpy as np
import depthai as dai
from fastapi import FastAPI, HTTPException
from ultralytics import YOLO, SAM

from pose import GraspPoseEstimator, CLASS_NAMES, DEPTH_SCALE
from robot_pick import (
    load_handeye,
    create_pipeline,
    estimate_from_frame,
    pick_point_base,
    HANDEYE_NPZ,
    MODEL_PATH,
    SAM_MODEL_PATH,
)

INFER_EVERY_N_FRAMES = 3   # YOLO on every Nth live frame
NO_DETECT_RETRY_S = 5.0    # a request retries on live frames this long before "no parcel"
LIVE_STALE_S = 5.0         # newest frame older than this => camera considered down
MIN_CONFIDENCE = 0.75      # detections below this are treated as "no object"

app = FastAPI(title="Parcel pose estimation service")
_cancel = threading.Event()  # set by /reset to abort a waiting /pose request

# Busy slot for "one estimation at a time" — an owner token instead of a plain
# Lock so /reset can FORCE-clear it even if an estimation is hung/slow. A zombie
# estimation finishing later releases only its own token (no-op after a reset).
_state_lock = threading.Lock()
_busy_token = 0    # 0 = idle; unique per acquisition (timestamps can collide)
_busy_since = 0.0
_token_seq = 0


def _acquire_busy():
    global _busy_token, _busy_since, _token_seq
    with _state_lock:
        if _busy_token:
            return None
        _token_seq += 1
        _busy_token = _token_seq
        _busy_since = time.time()
        return _busy_token


def _release_busy(token):
    global _busy_token, _busy_since
    with _state_lock:
        if _busy_token == token:
            _busy_token = 0
            _busy_since = 0.0

# Loaded once at startup (model load is seconds; per-request would be minutes).
T_BASE_CAM, K, _D = load_handeye(HANDEYE_NPZ)
MODEL = YOLO(MODEL_PATH)
SAM_MODEL = SAM(SAM_MODEL_PATH)
ESTIMATOR = GraspPoseEstimator(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                               depth_scale=DEPTH_SCALE)

# Latest live frame + detection, written only by the camera thread.
_latest = {"rgb": None, "depth": None, "corners": None,
           "cls_name": None, "conf": 0.0, "stamp": 0.0}
_latest_lock = threading.Lock()
_frame_stamp = [0.0]  # updated EVERY frame, even while YOLO is paused
_estimating = threading.Event()  # pauses live YOLO only during SAM/estimation


def _camera_loop():
    """Stream + detect forever; reopen the device whenever it drops."""
    while True:
        try:
            device = dai.Device()
            try:
                pipeline, rgb_out, depth_out = create_pipeline(device)
                rgb_queue = rgb_out.createOutputQueue()
                depth_queue = depth_out.createOutputQueue()
                pipeline.start()
                print("[camera] live")
                n = 0
                while True:
                    rgb = rgb_queue.get().getCvFrame()
                    depth = depth_queue.get().getFrame()  # uint16 mm
                    _frame_stamp[0] = time.time()
                    n += 1
                    # Skip live YOLO only while SAM/estimation runs (they fight
                    # for the CPU and estimations crawl). NOT while a request
                    # merely waits for a detection — it needs YOLO running.
                    if n % INFER_EVERY_N_FRAMES or _estimating.is_set():
                        continue
                    results = MODEL(rgb, verbose=False)
                    obb = results[0].obb
                    corners, cls_name, conf = None, None, 0.0
                    if obb is not None and len(obb.cls) > 0:
                        best = int(np.argmax(obb.conf.cpu().numpy()))
                        if float(obb.conf[best]) >= MIN_CONFIDENCE:
                            corners = obb.xyxyxyxy[best].cpu().numpy().reshape(4, 2)
                            cls_name = CLASS_NAMES.get(int(obb.cls[best]), str(int(obb.cls[best])))
                            conf = float(obb.conf[best])
                    with _latest_lock:
                        _latest.update(rgb=rgb, depth=depth, corners=corners,
                                       cls_name=cls_name, conf=conf, stamp=time.time())
            finally:
                device.close()
        except Exception as exc:
            print(f"[camera] error: {exc} — reconnecting in 5 s")
            time.sleep(5)


threading.Thread(target=_camera_loop, daemon=True).start()


def _camera_is_live():
    """Frames are arriving (independent of whether YOLO is currently paused)."""
    return _frame_stamp[0] > 0 and (time.time() - _frame_stamp[0]) < LIVE_STALE_S


def _detection_fresh(snap):
    return snap["stamp"] > 0 and (time.time() - snap["stamp"]) < LIVE_STALE_S


def _save_detection_png(snap, debug_prefix):
    """What the detector saw, saved BEFORE any calculation — for auditing
    false positives (e.g. 'detected a parcel' on an empty bin)."""
    vis = snap["rgb"].copy()
    corners = snap["corners"].astype(np.int32)
    cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
    cv2.putText(vis, f"{snap['cls_name']} {snap['conf']:.2f}",
                tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    path = f"{debug_prefix}_detect.png"
    cv2.imwrite(path, vis)
    return path


@app.get("/health")
def health():
    with _latest_lock:
        snap = dict(_latest)
    busy_for = round(time.time() - _busy_since, 1) if _busy_since else 0
    return {"ok": True,
            "camera_live": _camera_is_live(),
            "parcel_in_view": snap["corners"] is not None and _detection_fresh(snap),
            "estimating_for_s": busy_for}


@app.post("/reset")
def reset():
    """Force-clear the busy slot (even if an estimation is hung), abort any
    /pose waiting for a detection, and clear the current detection state.
    The camera stream keeps running."""
    global _busy_token, _busy_since
    with _state_lock:
        was_busy = _busy_token != 0
        busy_for = round(time.time() - _busy_since, 1) if was_busy else 0
        _busy_token = 0
        _busy_since = 0.0
    _cancel.set()
    with _latest_lock:
        _latest.update(corners=None, cls_name=None, conf=0.0)
    print(f"[pose] /reset called (was_busy={was_busy}, busy_for={busy_for} s) — "
          "busy slot force-cleared, waiting requests cancelled, detection state cleared")
    return {"ok": True, "was_busy": was_busy, "busy_for_s": busy_for,
            "message": "busy slot force-cleared; waiting requests cancelled; "
                       "detection state cleared"}


@app.post("/pose")
def estimate_pose():
    """Grasp point from the latest live detection; retries while nothing is in view."""
    token = _acquire_busy()
    if token is None:
        busy_for = time.time() - _busy_since
        raise HTTPException(status_code=409,
                            detail=f"estimation already in progress for {busy_for:.0f} s "
                                   f"(POST /reset to force-clear)")
    try:
        _cancel.clear()
        # Wait for a fresh frame WITH a detection, up to the retry window.
        deadline = time.time() + NO_DETECT_RETRY_S
        while True:
            if _cancel.is_set():
                return {"detected": False, "message": "cancelled by /reset"}
            with _latest_lock:
                snap = dict(_latest)
            if _detection_fresh(snap) and snap["corners"] is not None:
                break
            if time.time() >= deadline:
                if not _camera_is_live():
                    raise HTTPException(status_code=503,
                                        detail="camera not live (starting or reconnecting) — try again")
                return {"detected": False,
                        "message": f"no parcel detected within {NO_DETECT_RETRY_S:.0f} s"}
            time.sleep(0.3)

        # Log + save what the detector saw BEFORE any calculation, so false
        # positives on an empty bin can be audited.
        debug_prefix = time.strftime("debug_pick_%Y%m%d_%H%M%S")
        detect_png = _save_detection_png(snap, debug_prefix)
        print(f"[pose] request: detected {snap['cls_name']} "
              f"(conf={snap['conf']:.3f}) -> {detect_png}")

        t0 = time.time()
        _estimating.set()  # pause live YOLO so it doesn't compete with SAM
        try:
            pose = estimate_from_frame(SAM_MODEL, ESTIMATOR, snap["rgb"], snap["depth"],
                                       snap["corners"], snap["cls_name"], snap["conf"],
                                       debug_prefix)
            print(f"[pose] estimation took {time.time() - t0:.1f} s")
        except Exception as exc:
            # Never let a bad frame 500 the API — report it as a failed estimate.
            traceback.print_exc()
            return {"detected": False,
                    "message": f"estimation failed on this frame: {exc}",
                    "class_name": snap["cls_name"],
                    "confidence": round(float(snap["conf"]), 3),
                    "debug_prefix": debug_prefix}
        finally:
            _estimating.clear()
        if pose is None:
            return {"detected": False,
                    "message": "parcel detected but no valid grasp pose",
                    "class_name": snap["cls_name"],
                    "confidence": round(float(snap["conf"]), 3),
                    "debug_prefix": debug_prefix}

        pick_base = pick_point_base(T_BASE_CAM, pose["position"])
        normal_base = T_BASE_CAM[:3, :3] @ pose["normal"]
        np.savez(f"{debug_prefix}.npz",
                 position_cam=pose["position"], normal_cam=pose["normal"],
                 T_base_cam=T_BASE_CAM, pick_base=pick_base)
        print(f"[pose] pick pose: base (mm)={np.round(pick_base * 1000, 1)} "
              f"normal={np.round(normal_base, 4)} "
              f"class={pose['class_name']} conf={pose['confidence']:.3f}")

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
        _release_busy(token)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
