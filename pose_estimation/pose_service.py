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

import frame_capture
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

INFER_EVERY_N_FRAMES = 3   # background YOLO on every Nth live frame (feeds /health)
NO_DETECT_RETRY_S = 5.0    # a request retries on fresh frames this long before "no parcel"
LIVE_STALE_S = 5.0         # newest frame older than this => camera considered down
MIN_CONFIDENCE = 0.75      # detections below this are treated as "no object"
SETTLE_S = 0.5             # belt-deceleration settle before capturing (package comes to rest)
FRESH_SKIP_FRAMES = 2      # seq must advance this much so the frame is exposed AFTER the request
FRESH_CAPTURE_TIMEOUT_S = 3.0  # max wait for a genuinely-new frame before 503

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

# Latest background detection, written by the camera thread — feeds /health only.
_latest = {"rgb": None, "depth": None, "corners": None,
           "cls_name": None, "conf": 0.0, "stamp": 0.0}
_latest_lock = threading.Lock()
_frame_stamp = [0.0]  # updated EVERY frame, even while YOLO is paused
_estimating = threading.Event()  # pauses live YOLO only during SAM/estimation

# Newest RAW frame + a monotonic seq, published EVERY frame by the camera thread.
# A /pose request waits on this (never touches the DepthAI queues) to obtain a
# frame captured AFTER the request — i.e. after the belt stopped.
_capture = frame_capture.FrameBuffer()


def _detect_obb(rgb):
    """YOLO-OBB on one BGR frame -> (corners[4x2], cls_name, conf) for the best box
    that is >= MIN_CONFIDENCE AND a known class; otherwise (None, None, 0.0).
    An unknown class index clamps to no-detection (never a raw number string)."""
    results = MODEL(rgb, verbose=False)
    obb = results[0].obb
    if obb is None or len(obb.cls) == 0:
        return None, None, 0.0
    best = int(np.argmax(obb.conf.cpu().numpy()))
    conf = float(obb.conf[best])
    cls_name = frame_capture.resolve_class_name(int(obb.cls[best]), conf,
                                                MIN_CONFIDENCE, CLASS_NAMES)
    if cls_name is None:
        return None, None, 0.0
    corners = obb.xyxyxyxy[best].cpu().numpy().reshape(4, 2)
    return corners, cls_name, conf


def _camera_loop():
    """Stream + detect forever; reopen the device whenever it drops."""
    while True:
        try:
            device = dai.Device()
            try:
                pipeline, rgb_out, depth_out = create_pipeline(device)
                # maxSize=1, non-blocking: no FIFO backlog — .get() always yields
                # the newest frame instead of the oldest buffered one.
                rgb_queue = rgb_out.createOutputQueue(maxSize=1, blocking=False)
                depth_queue = depth_out.createOutputQueue(maxSize=1, blocking=False)
                pipeline.start()
                print("[camera] live")
                n = 0
                while True:
                    rgb_msg = rgb_queue.get()
                    depth_msg = depth_queue.get()
                    rgb = rgb_msg.getCvFrame()
                    depth = depth_msg.getFrame()  # uint16 mm
                    _frame_stamp[0] = time.time()
                    # Publish the newest raw frame EVERY iteration (before the YOLO
                    # skip) so a /pose request can wait for a genuinely post-request
                    # frame even while background YOLO is paused during estimation.
                    _capture.publish(rgb, depth, rgb_msg.getTimestamp().total_seconds())
                    n += 1
                    # Background YOLO (feeds /health) skips every non-Nth frame and
                    # pauses entirely while SAM/estimation runs (they fight for CPU).
                    if n % INFER_EVERY_N_FRAMES or _estimating.is_set():
                        continue
                    corners, cls_name, conf = _detect_obb(rgb)
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


def _run_estimation(snap):
    """SAM -> grasp pose -> debug artifacts -> response body, from a detected frame.
    Caller MUST already hold the busy slot and have _estimating set. Never raises on
    a bad frame — a failure becomes a detected:false body (class_name null)."""
    debug_prefix = time.strftime("debug_pick_%Y%m%d_%H%M%S")
    detect_png = _save_detection_png(snap, debug_prefix)
    print(f"[pose] request: detected {snap['cls_name']} "
          f"(conf={snap['conf']:.3f}) -> {detect_png}")

    t0 = time.time()
    try:
        pose = estimate_from_frame(SAM_MODEL, ESTIMATOR, snap["rgb"], snap["depth"],
                                   snap["corners"], snap["cls_name"], snap["conf"],
                                   debug_prefix)
        print(f"[pose] estimation took {time.time() - t0:.1f} s")
    except Exception as exc:
        # Never let a bad frame 500 the API — report it as a failed estimate.
        traceback.print_exc()
        return frame_capture.no_detection_response(
            f"estimation failed on this frame: {exc}", debug_prefix=debug_prefix)
    if pose is None:
        return frame_capture.no_detection_response(
            "parcel detected but no valid grasp pose", debug_prefix=debug_prefix)

    pick_base = pick_point_base(T_BASE_CAM, pose["position"])
    normal_base = T_BASE_CAM[:3, :3] @ pose["normal"]
    np.savez(f"{debug_prefix}.npz",
             position_cam=pose["position"], normal_cam=pose["normal"],
             T_base_cam=T_BASE_CAM, pick_base=pick_base)
    print(f"[pose] pick pose: base (mm)={np.round(pick_base * 1000, 1)} "
          f"normal={np.round(normal_base, 4)} "
          f"class={pose['class_name']} conf={pose['confidence']:.3f}")

    # API positions/lengths are in MILLIMETERS; internal math/npz stay in meters.
    mm = lambda vec: [round(float(v) * 1000, 1) for v in vec]
    return frame_capture.detection_response(
        class_name=pose["class_name"],
        confidence=pose["confidence"],
        pick_base=mm(pick_base),
        normal_base=[round(float(v), 4) for v in normal_base],
        position_cam=mm(pose["position"]),
        normal_cam=[round(float(v), 4) for v in pose["normal"]],
        flatness_mm=round(float(pose["flatness_score"]) * 1000, 2),
        inliers=int(pose["inlier_count"]),
        debug_prefix=debug_prefix,
    )


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
    """Grasp point from a FRESH frame captured AFTER this request arrives.

    Flow: settle (let the just-stopped belt/package come to rest) -> wait for a
    frame exposed after the request (never the stale FIFO backlog) -> detect on
    THAT frame -> SAM + grasp estimate. Retries on subsequent fresh frames while
    nothing is in view, up to NO_DETECT_RETRY_S. class_name is present in every
    response and is null whenever detected is false.
    """
    token = _acquire_busy()
    if token is None:
        busy_for = time.time() - _busy_since
        raise HTTPException(status_code=409,
                            detail=f"estimation already in progress for {busy_for:.0f} s "
                                   f"(POST /reset to force-clear)")
    try:
        _cancel.clear()
        if not _camera_is_live():
            raise HTTPException(status_code=503,
                                detail="camera not live (starting or reconnecting) — try again")

        # Settle: the belt stop is a servo velocity->0; give the package a moment
        # to come to rest before we freeze a frame.
        if not frame_capture.sleep_with_cancel(SETTLE_S, _cancel):
            return frame_capture.no_detection_response("cancelled by /reset")

        _estimating.set()  # pause background YOLO so it doesn't compete with our YOLO+SAM
        try:
            deadline = time.time() + NO_DETECT_RETRY_S
            while True:
                res = frame_capture.wait_for_fresh_frame(
                    _capture, FRESH_SKIP_FRAMES, FRESH_CAPTURE_TIMEOUT_S,
                    _cancel, _camera_is_live)
                if res["status"] == "cancelled":
                    return frame_capture.no_detection_response("cancelled by /reset")
                if res["status"] == "timeout":
                    raise HTTPException(status_code=503,
                                        detail="camera stalled — no fresh frame; try again")

                fresh = res["frame"]
                corners, cls_name, conf = _detect_obb(fresh["rgb"])
                if corners is not None:
                    snap = {"rgb": fresh["rgb"], "depth": fresh["depth"],
                            "corners": corners, "cls_name": cls_name, "conf": conf}
                    return _run_estimation(snap)
                if time.time() >= deadline:
                    return frame_capture.no_detection_response(
                        f"no parcel detected within {NO_DETECT_RETRY_S:.0f} s")
                # else: loop and grab the next fresh frame to retry detection
        finally:
            _estimating.clear()
    finally:
        _release_busy(token)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
