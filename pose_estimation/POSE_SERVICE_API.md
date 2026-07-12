# Parcel Pose Estimation Service — API Guide (for the robot side)

We are the **ML service**: camera + detection + grasp-point estimation only.
We do **not** connect to or control the robot. Your side calls this API to get
the pick point, then plans and executes the motion yourself.

Interactive docs (try it in a browser): `http://<ml-host>:8001/docs`

## Service startup (ML side)

```
uvicorn pose_service:app --host 0.0.0.0 --port 8001
```

Base URL: `http://<ml-host>:8001`. No auth — keep it on the internal network.

---

## GET /health

Liveness check:
```json
{ "ok": true, "camera_live": true, "parcel_in_view": true, "estimating_for_s": 0 }
```
`estimating_for_s` > 0 means a `/pose` request is currently being processed
(and how long it has been running).
The camera streams and detects **continuously** in the background.
`camera_live: false` means the camera is starting up or reconnecting after a
drop — a `POST /pose` sent now will likely return `503`. Call this before a
pick cycle to fail fast.

---

## POST /pose

Returns a grasp estimate from the **latest live detection**. No request body.

Timing: normally **1–5 s** (segmentation + grasp-patch search; the camera is
already streaming). If no parcel is in view, the service keeps retrying on
fresh frames for up to **5 s** before answering `detected: false`. Use a
client timeout of **30 s**. One request at a time — concurrent calls get `409`.

**Response `200` — parcel found** — all positions/lengths in **MILLIMETERS**
```json
{
  "detected": true,
  "class_name": "white_bag",
  "confidence": 0.507,
  "pick_base":    [-311.5, -454.0, 252.3],
  "normal_base":  [0.0731, 0.1046, 0.9918],
  "position_cam": [1.4, 60.7, 441.5],
  "normal_cam":   [-0.0731, -0.1046, -0.9918],
  "flatness_mm": 1.22,
  "inliers": 244,
  "debug_prefix": "debug_pick_20260708_161834"
}
```

| field          | type      | meaning                                                              |
|----------------|-----------|----------------------------------------------------------------------|
| `detected`     | bool      | `true` = valid grasp point below                                     |
| `class_name`   | string    | `box`, `brown_bag`, or `white_bag`                                   |
| `confidence`   | number    | detector confidence 0–1 (always ≥ 0.75 — weaker detections are treated as no object) |
| `pick_base`    | [x,y,z]   | **grasp point in the ROBOT BASE frame, millimeters** — the value you move to. Already transformed with our hand-eye calibration. |
| `normal_base`  | [x,y,z]   | surface normal at the grasp point, base frame, **unit vector (no unit)**, points up out of the parcel |
| `position_cam` / `normal_cam` | [x,y,z] | same data in the camera frame (mm / unit vector), for cross-checking |
| `flatness_mm`  | number    | mean plane residual of the chosen patch, mm (smaller = flatter)      |
| `inliers`      | int       | points supporting the patch plane fit                                |
| `debug_prefix` | string    | debug artifacts saved ML-side: `<prefix>_detect.png` (raw detection box, saved BEFORE any calculation), `<prefix>.png` (red-dot grasp image), `.ply` point cloud, `.npz` coordinates — npz/ply are in METERS, internal use |

**Response `200` — nothing usable in view**
```json
{ "detected": false, "message": "no parcel detected within 5 s" }
```
Other `detected: false` messages: `"parcel detected but no valid grasp pose"`,
`"estimation failed on this frame: <reason>"` (bad frame — just request again),
and `"cancelled by /reset"`. When a detection existed, these also include
`class_name`, `confidence`, and `debug_prefix` so you can see what it saw.
Not an error — check `detected` before using any field.

**Errors** (body: `{ "detail": "<reason>" }`)

| status | when                                             | your handling            |
|--------|--------------------------------------------------|--------------------------|
| 409    | an estimation is already in progress             | wait and retry, or `POST /reset` |
| 503    | camera not live (starting up or reconnecting after a USB/firmware drop) | retry after ~10 s; alert if persistent |

---

## POST /reset

Clears a stuck cycle: **force-clears the busy slot** (even if an estimation is
hung mid-computation), aborts any `/pose` request still waiting for a detection
(it returns `detected: false, "cancelled by /reset"`), and clears the current
detection state. The camera stream keeps running — no restart needed. A new
`/pose` can be sent immediately after.

```json
{ "ok": true, "was_busy": true, "busy_for_s": 42.5,
  "message": "busy slot force-cleared; waiting requests cancelled; detection state cleared" }
```

---

## What the robot side should build

A client that per pick cycle:

1. `POST /pose` (timeout 30 s); optionally `GET /health` first.
2. If `detected` is `false` → no parcel; skip the cycle.
3. Sanity-check `pick_base` against your own workspace limits — we guarantee
   the point is on the parcel surface, not that it is reachable for your arm.
4. Add your own standoff above `pick_base` (base +Z) and plan a path that
   clears the camera: it hangs at base-frame **z = 695 mm**, roughly above the
   pick area. Anything of yours above the TCP must stay clear of it.
5. Move, verify, actuate suction — all yours.

**Python example**
```python
import requests

r = requests.post("http://<ml-host>:8001/pose", timeout=30)
r.raise_for_status()
data = r.json()
if not data["detected"]:
    print("no parcel:", data["message"])
else:
    x, y, z = data["pick_base"]      # robot base frame, MILLIMETERS
    # -> your motion planning from here (divide by 1000 if your stack wants meters)
```

**curl**
```bash
curl -X POST http://<ml-host>:8001/pose
```

## Coordinate frame notes

- `pick_base` is in the **robot base frame, millimeters**, produced with our
  camera-to-base hand-eye calibration (`T_base_cam`, included in every `.npz`
  debug dump — the npz itself is in meters). Validated by pendant tests to
  reach the parcel correctly.
- If the camera or the robot is ever moved/remounted, the calibration is stale
  — tell us so we re-calibrate before you trust `pick_base` again.
- Orientation: we provide the surface `normal_base` only; a vertical
  (straight-down) approach is assumed for suction. We do not provide a full
  6-DOF grasp orientation.
