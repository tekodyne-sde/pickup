# master_context.md — pose_estimation

Long-lived context for the parcel pose-estimation service. Preserves the "why" behind
non-obvious decisions and a running session log. Read this first when picking the work
back up; append a session entry when you finish a chunk.

---

## System at a glance

- **`pose_service.py`** — FastAPI ML service on **:8001**. Owns an OAK-D, streams RGB+depth
  continuously in a background thread, runs YOLO-OBB detection + SAM + grasp-pose estimation.
  Endpoints: `GET /health`, `POST /pose`, `POST /reset`. Returns the pick point in the
  **robot base frame, millimetres** (`pick_base`) via the hand-eye calibration.
- **`robot_pick.py`** — one-shot CLI pipeline + shared helpers (`create_pipeline`,
  `estimate_from_frame`, `pick_point_base`, hand-eye loader). `pose_service.py` imports the
  pipeline builder and estimation helpers from here.
- **`frame_capture.py`** — hardware-free core of the fresh-frame logic (FrameBuffer, fresh-wait,
  settle, class-name resolver, response builders). Unit-tested in `tests/`.
- **`pose.py`** — `GraspPoseEstimator`, mask helpers, `CLASS_NAMES = {0: box, 1: brown_bag,
  2: white_bag}`, debug snapshot.
- **`tune_camera.py`** — live focus/exposure tuning tool (hardware).
- Debug artifacts per request: `<prefix>_detect.png` (raw detection box, saved before any
  calc), `<prefix>.png` (red-dot grasp), `<prefix>.ply` (cloud), `<prefix>.npz` (coords,
  metres). `.npz`/`.ply` are metres; the API is millimetres.

## Consumer / integration (external repo)

- The live client is the **tekostudio** repo's UR driver
  `apps/anantos/src/anantos/drivers/robot.py` (:8002). Its `_query_pose_service()` does
  `POST http://127.0.0.1:8001/pose` (no body, 90 s timeout, `409`=busy, no retries) and reads
  `detected` + `pick_base` (+ now `class_name`). Both the Blockly `pose_query` block and
  `robot_pick` (auto-pose) funnel through it.
- The belt-stop→pose sequence is authored in Blockly: `move_until_sensor`
  (`forward → poll sensor every 200 ms → stop()`) then a separate `pose_query`/`robot_pick`
  block. **No automatic settle** exists between the belt `stop()` and the pose call.
- The frozen `docs/api/perception.md` `/perceive` contract (camera-frame pose, base64
  image+depth in the request) is a **different, unimplemented** interface — not what the live
  client uses. Not touched by this work.

## Key decisions (why)

- **D1 — `/pose` uses a fresh, post-request frame, not the latest streamed one.** The DepthAI
  output queues defaulted to `maxSize=16` FIFO; `.get()` returns the *oldest* buffered frame,
  and the consumer lags the producer, so the service processed seconds-old, motion-blurred,
  in-transit frames. Freshness was judged by `time.time()` at *processing* time, so stale
  frames looked "fresh." Fix: `maxSize=1` queues + the camera thread publishes every frame with
  a monotonic `seq`; `/pose` settles then waits for `seq` to advance (`FRESH_SKIP_FRAMES=2`) so
  the used frame was exposed *after* the request. "Fresh-wait" is NOT a camera reboot/refocus —
  the camera stays live; it just waits ~1–2 frames off the running stream.
- **D2 — `+2` frame skip.** The frame in flight at request time may be mid-exposure (started
  before the request); requiring `seq ≥ base+2` guarantees a frame whose exposure began after
  the request.
- **D3 — ~500 ms server-side settle.** The belt "stop" is a servo velocity→0 and nothing in the
  program adds a settle, so `/pose` settles itself (`SETTLE_S=0.5`) to let the package come to
  rest. Cancellable via `/reset`.
- **D4 — Lock focus + cap exposure at startup.** CAM_A ran continuous autofocus and could hunt
  (soft frames) when a package appeared, and uncapped exposure blurred a moving belt. Since the
  camera-to-belt distance is fixed, `create_pipeline` sets `initialControl.setManualFocus` /
  `setManualExposure`. Values (`RGB_LENS_POSITION`, `RGB_EXPOSURE_US`, `RGB_ISO` in
  `robot_pick.py`) are **hardware-specific** — re-tune with `tune_camera.py` on every camera
  move / recalibration, then paste the printed block.
- **D5 — `class_name` is the classification field; no `type` added.** The client reads the
  existing `class_name`. It is now present in EVERY response and strictly tracks `detected`:
  one of `box|brown_bag|white_bag` when `detected` is true, `null` otherwise. Unknown class
  indices clamp to no-detection so `class_name` is never a raw number. Contract handed to the
  client in `POSE_CLASS_NAME_SPEC.md`.
- **D6 — Fixed `/pose` in place (no new endpoint).** The only live client calls `/pose`; fixing
  it in place keeps the fix zero-client-change and immediately live. Response schema, `409`/`503`,
  and `/reset` cancellation are preserved.

## Tunables (pose_service.py / robot_pick.py)

- `SETTLE_S = 0.5`, `FRESH_SKIP_FRAMES = 2`, `FRESH_CAPTURE_TIMEOUT_S = 3.0` (pose_service.py)
- `NO_DETECT_RETRY_S = 5.0`, `MIN_CONFIDENCE = 0.75`, `INFER_EVERY_N_FRAMES = 3` (unchanged)
- `RGB_LENS_POSITION`, `RGB_EXPOSURE_US`, `RGB_ISO` (robot_pick.py — **placeholders**, tune on HW)

## Testing

- `python -m pytest tests/ -q` — unit tests for `frame_capture` (no hardware/models).
- `python test_grasp_selection.py` — existing grasp-patch self-check.
- Hardware-in-the-loop (human): tune focus with `tune_camera.py`; run `pose_service.py`; drive
  the belt-stop→`/pose` flow and confirm `<prefix>_detect.png` is sharp and in the stopped
  position; check `class_name` present; `409` on overlap; `/reset` cancels; `503` when camera down.

## Open follow-ups (not done)

### DEFERRED — config extraction + project cleanup (its own dedicated effort)
Planned for a later session; requested 2026-07-12. Do this as a standalone, carefully-scoped
refactor — **must not change `pose_service.py`'s behavior, the `/pose` HTTP contract, or the
`class_name`/`pick_base` shapes** the tekostudio client depends on. Suggested scope:
- **Config file:** move the camera-tuning constants (`RGB_LENS_POSITION`, `RGB_EXPOSURE_US`,
  `RGB_ISO`, and likely `FRAME_WIDTH`/`FRAME_HEIGHT`, `CAMERA_SETTLE_S`) out of `robot_pick.py`
  into a single `config.py`, imported by both `create_pipeline` and `tune_camera.py`, so there's
  one obvious place to edit when the camera is repositioned/recalibrated. (Entry point is
  `pose_service.py`; today these live in `robot_pick.py` only because `create_pipeline` does — it
  works via import, but the location is indirect.)
- **Naming conventions:** consistent, descriptive module/function names across the project.
- **Remove stale scripts / dead code:** duplicate `capture_gui.py` / `capture_gui_fixed.py` /
  `capture_gui_fixed_v2.py`; dead `core/camera_pipeline.py` (unused by the service); any other
  one-off scripts no longer needed.
- **Reorganize** into a clean structure (e.g. group service vs. CLI vs. tools) — verify
  `pose_service.py` still imports and runs, tests still pass, before/after.
- Gate: run `python -m pytest tests/ -q` and a smoke start of `pose_service.py` after each move.

### Other
- RGB/depth same-instant synchronization (`dai.node.Sync` / timestamp matching) — currently
  independent `.get()` per stream.
- Repo hygiene: `.cache/depthai/telemetry/**`, `__pycache__/`, `venv/` are tracked and should be
  git-ignored.
- Optional: expose `settle_ms` as an optional `/pose` request field (backward-compatible).

---

## Session log (append-only)

### 2026-07-12 — fresh-frame /pose fix, focus/exposure lock, class_name guarantee
- Root-caused the "pose estimated on a moving/blurred pre-stop frame" report: DepthAI FIFO
  backlog + processing-time freshness stamp (see D1). Confirmed with the debug images
  (`debug_pick_20260712_144959_detect.png` etc.).
- Added `frame_capture.py` + `tests/test_frame_capture.py` (TDD: 17 tests, red→green).
- Reworked `pose_service.py`: `maxSize=1` queues, per-frame `FrameBuffer.publish`, extracted
  `_detect_obb` / `_run_estimation`, rewrote `/pose` to settle → fresh-wait → detect-on-fresh →
  estimate. Background YOLO→`_latest` kept for `/health`.
- Locked focus + capped exposure in `create_pipeline` (`robot_pick.py`) with tunable constants;
  added `tune_camera.py` to tune/verify them.
- Guaranteed `class_name` present in every `/pose` body, null iff `detected` false; clamped
  unknown class indices. Documented in `POSE_SERVICE_API.md` and `POSE_CLASS_NAME_SPEC.md`.
- HW-in-the-loop steps (focus tuning + real belt-stop capture) remain for a human on the cell.
