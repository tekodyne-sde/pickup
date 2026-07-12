"""Unit tests for frame_capture — the hardware-free core of the /pose fresh-frame
fix. No camera, no models: FrameBuffer (publish/snapshot/seq), wait_for_fresh_frame
(post-request frame gating + cancel + timeout), sleep_with_cancel (settle), the
class-name resolver (closed enum + unknown-index clamp), and the response builders
(class_name always present, null iff not detected).

Run: python -m pytest tests/ -q
"""

import threading

import frame_capture as fc


# ----------------------------------------------------------------------------
# FrameBuffer
# ----------------------------------------------------------------------------
def test_new_buffer_is_empty_seq_zero():
    buf = fc.FrameBuffer()
    assert buf.current_seq() == 0
    snap = buf.snapshot()
    assert snap["rgb"] is None and snap["depth"] is None
    assert snap["seq"] == 0 and snap["cap_ts"] == 0.0


def test_publish_increments_seq_monotonically():
    buf = fc.FrameBuffer()
    buf.publish("rgb0", "d0", 1.0)
    assert buf.current_seq() == 1
    buf.publish("rgb1", "d1", 2.0)
    assert buf.current_seq() == 2


def test_snapshot_returns_latest_published():
    buf = fc.FrameBuffer()
    buf.publish("first", "df", 1.0)
    buf.publish("second", "ds", 2.5)
    snap = buf.snapshot()
    assert snap["rgb"] == "second"
    assert snap["depth"] == "ds"
    assert snap["cap_ts"] == 2.5
    assert snap["seq"] == 2


def test_publish_is_thread_safe():
    buf = fc.FrameBuffer()
    n_threads, per = 8, 500

    def worker():
        for _ in range(per):
            buf.publish("x", "y", 0.0)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert buf.current_seq() == n_threads * per


# ----------------------------------------------------------------------------
# wait_for_fresh_frame
# ----------------------------------------------------------------------------
def test_wait_returns_ok_once_seq_advances_by_skip_frames():
    buf = fc.FrameBuffer()

    # Each poll-sleep simulates the camera publishing one new frame.
    def fake_sleep(_):
        buf.publish("rgb", "depth", 0.0)

    res = fc.wait_for_fresh_frame(
        buf, skip_frames=2, timeout_s=10.0,
        cancel=_FakeCancel(False), camera_live=lambda: True,
        now=lambda: 0.0, sleep=fake_sleep,
    )
    assert res["status"] == "ok"
    # base seq was 0, skip_frames=2 -> the returned frame must be seq >= 2.
    assert res["frame"]["seq"] >= 2


def test_wait_returns_cancelled_when_cancel_already_set():
    buf = fc.FrameBuffer()
    slept = []
    res = fc.wait_for_fresh_frame(
        buf, skip_frames=2, timeout_s=10.0,
        cancel=_FakeCancel(True), camera_live=lambda: True,
        now=lambda: 0.0, sleep=lambda _: slept.append(1),
    )
    assert res["status"] == "cancelled"
    assert slept == []  # returned before sleeping


def test_wait_returns_cancelled_midway():
    buf = fc.FrameBuffer()
    cancel = _FakeCancel(False)

    def fake_sleep(_):
        cancel.value = True  # cancel arrives after the first poll

    res = fc.wait_for_fresh_frame(
        buf, skip_frames=5, timeout_s=10.0,
        cancel=cancel, camera_live=lambda: True,
        now=lambda: 0.0, sleep=fake_sleep,
    )
    assert res["status"] == "cancelled"


def test_wait_times_out_and_reports_camera_live():
    buf = fc.FrameBuffer()
    clock = _FakeClock(start=0.0, step=1.0)  # advances 1 s per call
    res = fc.wait_for_fresh_frame(
        buf, skip_frames=2, timeout_s=0.0,  # deadline == first now() reading
        cancel=_FakeCancel(False), camera_live=lambda: False,
        now=clock, sleep=lambda _: None,
    )
    assert res["status"] == "timeout"
    assert res["camera_live"] is False


# ----------------------------------------------------------------------------
# sleep_with_cancel (settle)
# ----------------------------------------------------------------------------
def test_settle_completes_when_not_cancelled():
    clock = _FakeClock(start=0.0, step=0.1)
    done = fc.sleep_with_cancel(
        0.5, cancel=_FakeCancel(False),
        now=clock, sleep=lambda _: None, poll_interval=0.1,
    )
    assert done is True


def test_settle_aborts_when_cancelled():
    done = fc.sleep_with_cancel(
        0.5, cancel=_FakeCancel(True),
        now=_FakeClock(0.0, 0.1), sleep=lambda _: None, poll_interval=0.1,
    )
    assert done is False


# ----------------------------------------------------------------------------
# resolve_class_name — closed enum + unknown-index clamp
# ----------------------------------------------------------------------------
CLASS_NAMES = {0: "box", 1: "brown_bag", 2: "white_bag"}


def test_resolve_known_class_above_confidence():
    assert fc.resolve_class_name(0, 0.90, 0.75, CLASS_NAMES) == "box"
    assert fc.resolve_class_name(1, 0.80, 0.75, CLASS_NAMES) == "brown_bag"
    assert fc.resolve_class_name(2, 0.99, 0.75, CLASS_NAMES) == "white_bag"


def test_resolve_below_confidence_is_none():
    assert fc.resolve_class_name(0, 0.50, 0.75, CLASS_NAMES) is None


def test_resolve_unknown_index_is_none_not_raw_number():
    # An out-of-range class index must clamp to None, never a stringified int.
    assert fc.resolve_class_name(9, 0.99, 0.75, CLASS_NAMES) is None


# ----------------------------------------------------------------------------
# Response builders — class_name presence guarantee
# ----------------------------------------------------------------------------
def test_no_detection_response_class_name_is_null_and_present():
    r = fc.no_detection_response("no parcel detected within 5 s")
    assert r["detected"] is False
    assert "class_name" in r and r["class_name"] is None
    assert r["message"] == "no parcel detected within 5 s"


def test_no_detection_response_passes_through_extra_fields():
    r = fc.no_detection_response("estimation failed", debug_prefix="debug_pick_x")
    assert r["class_name"] is None
    assert r["debug_prefix"] == "debug_pick_x"


def test_detection_response_class_name_present_and_valid():
    r = fc.detection_response(
        class_name="box", confidence=0.94,
        pick_base=[-441.7, -448.4, 230.6], normal_base=[0.0, 0.0, 1.0],
        position_cam=[1.4, 60.7, 441.5], normal_cam=[0.0, 0.0, -1.0],
        flatness_mm=1.22, inliers=244, debug_prefix="debug_pick_x",
    )
    assert r["detected"] is True
    assert r["class_name"] == "box"
    assert r["pick_base"] == [-441.7, -448.4, 230.6]
    assert r["confidence"] == 0.94


def test_both_builders_always_include_class_name_key():
    # The client relies on class_name being present in EVERY response body.
    assert "class_name" in fc.no_detection_response("x")
    assert "class_name" in fc.detection_response(
        class_name="white_bag", confidence=0.8, pick_base=[0, 0, 0],
        normal_base=[0, 0, 1], position_cam=[0, 0, 0], normal_cam=[0, 0, -1],
        flatness_mm=0.0, inliers=1, debug_prefix="p",
    )


# ----------------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------------
class _FakeCancel:
    def __init__(self, value):
        self.value = value

    def is_set(self):
        return self.value


class _FakeClock:
    """Returns start, start+step, start+2*step, ... one increment per call."""
    def __init__(self, start=0.0, step=1.0):
        self._t = start
        self._step = step

    def __call__(self):
        t = self._t
        self._t += self._step
        return t
