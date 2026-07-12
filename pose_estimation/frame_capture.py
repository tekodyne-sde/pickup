"""frame_capture.py — hardware-free core of the /pose fresh-frame fix.

Keeps the camera thread as the SOLE owner of the DepthAI queues: it publishes the
newest raw RGB+depth frame plus a monotonic sequence counter into a FrameBuffer,
and a /pose request waits (without ever touching the queues) for the counter to
advance past the value seen at request time. That guarantees the frame it uses was
exposed AFTER the request arrived (i.e. after the belt stopped), instead of a stale
frame from the DepthAI FIFO backlog.

Everything here is pure threading/time logic — no depthai, no models — so it is
unit-tested in tests/test_frame_capture.py with fakes.
"""

import threading
import time


class FrameBuffer:
    """Thread-safe holder of the single newest frame + a monotonic seq counter.

    The camera thread calls publish() every iteration; request threads call
    snapshot()/current_seq(). seq increases by one per publish and never resets,
    so it survives camera reconnects and uniquely orders frames.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._rgb = None
        self._depth = None
        self._seq = 0
        self._cap_ts = 0.0

    def publish(self, rgb, depth, cap_ts):
        with self._lock:
            self._seq += 1
            self._rgb = rgb
            self._depth = depth
            self._cap_ts = cap_ts

    def snapshot(self):
        with self._lock:
            return {"rgb": self._rgb, "depth": self._depth,
                    "seq": self._seq, "cap_ts": self._cap_ts}

    def current_seq(self):
        with self._lock:
            return self._seq


def wait_for_fresh_frame(buffer, skip_frames, timeout_s, cancel, camera_live,
                         now=time.monotonic, sleep=time.sleep, poll_interval=0.02):
    """Block until `buffer` holds a frame exposed strictly after this call.

    Returns one of:
      {"status": "ok", "frame": <snapshot>}   — a fresh frame is available
      {"status": "cancelled"}                  — `cancel` was set (e.g. /reset)
      {"status": "timeout", "camera_live": b}  — no fresh frame within timeout_s

    skip_frames should be >= 2: the frame in flight when the request arrived may
    already be mid-exposure (started before the request), so we require the seq to
    advance by two to be certain the captured frame's exposure began after now.

    `now`/`sleep` are injectable for deterministic tests.
    """
    base = buffer.current_seq()
    target = base + skip_frames
    deadline = now() + timeout_s
    while True:
        if cancel.is_set():
            return {"status": "cancelled"}
        snap = buffer.snapshot()
        if snap["seq"] >= target and snap["rgb"] is not None:
            return {"status": "ok", "frame": snap}
        if now() >= deadline:
            return {"status": "timeout", "camera_live": bool(camera_live())}
        sleep(poll_interval)


def sleep_with_cancel(duration_s, cancel, now=time.monotonic, sleep=time.sleep,
                      poll_interval=0.02):
    """Sleep for duration_s, but bail out early if `cancel` is set.

    Returns True if the full duration elapsed, False if cancelled. Used for the
    belt-deceleration settle before capturing.
    """
    deadline = now() + duration_s
    while now() < deadline:
        if cancel.is_set():
            return False
        sleep(poll_interval)
    return True


def resolve_class_name(cls_index, conf, min_confidence, class_names):
    """Map a detector class index to its package name, or None.

    Returns None when confidence is below min_confidence OR the index is not a
    known class — so a caller never emits a raw/unknown class string. This is the
    clamp that keeps class_name within the closed {box, brown_bag, white_bag} set
    (or null).
    """
    if conf < min_confidence:
        return None
    return class_names.get(int(cls_index))


def no_detection_response(message, **extra):
    """Body for any detected:false outcome. class_name is always present and null."""
    return {"detected": False, "class_name": None, "message": message, **extra}


def detection_response(*, class_name, confidence, pick_base, normal_base,
                       position_cam, normal_cam, flatness_mm, inliers, debug_prefix):
    """Body for a successful pick. class_name is always present and non-null.

    Positions/lengths are in MILLIMETERS (API boundary); normals are unit vectors.
    """
    return {
        "detected": True,
        "class_name": class_name,
        "confidence": round(float(confidence), 3),
        "pick_base": pick_base,
        "normal_base": normal_base,
        "position_cam": position_cam,
        "normal_cam": normal_cam,
        "flatness_mm": flatness_mm,
        "inliers": inliers,
        "debug_prefix": debug_prefix,
    }
