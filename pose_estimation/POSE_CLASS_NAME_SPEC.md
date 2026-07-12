# Client Spec — `class_name` field on the pose service `POST /pose` (port 8001)

## Context
The vision/ML pose service (`POST http://<ml-host>:8001/pose`, no request body) exposes the
classified package type via the existing **`class_name`** field. **Do not read a `type` field —
it does not exist.** This spec defines exactly how `class_name` behaves and how the client must
handle it.

## Field contract

`class_name` — the classified package type.

- **Always present** in every `/pose` JSON response body (never missing/omitted/undefined).
- **Type:** `string | null`.
- **Allowed non-null values (closed set, lowercase snake_case):** exactly one of
  `"box"`, `"brown_bag"`, `"white_bag"`. No other string is ever returned.
- **Strictly tracks `detected`:**
  - `detected: true`  → `class_name` is exactly one of the three values above. Never `null`.
  - `detected: false` → `class_name` is `null`. Always.

If a value outside `{"box","brown_bag","white_bag", null}` is ever received, treat it as a
protocol error (fail closed, no pick).

## Response variants (authoritative)

Detection succeeded — a pick point is available:
```json
{ "detected": true, "class_name": "box", "confidence": 0.94,
  "pick_base": [-441.7, -448.4, 230.6], "normal_base": [0.07, 0.10, 0.99] }
```

No usable detection (no parcel / cancelled / grasp failed) — no pick point:
```json
{ "detected": false, "class_name": null, "message": "no parcel detected within 5 s" }
```

`pick_base` is `[x, y, z]` in **millimetres, robot base frame**, present only when
`detected` is `true`. All other existing fields are unchanged.

## Client handling rules (MUST)

1. **Gate every pick decision on `detected`, not on `class_name`.** Only pick when
   `detected === true`.
2. When `detected === true`, `class_name` is guaranteed one of
   `"box" | "brown_bag" | "white_bag"` — safe to switch/route on directly.
3. When `detected === false`, `class_name` is `null` — do not pick, do not infer a type.
4. Do **not** read a `type` field. Migrate any code reading `type` to read `class_name`.
5. Treat `class_name` as a closed enum (three values + `null`); if an unexpected value appears,
   fail closed (no pick) and log it.

## Types

TypeScript:
```ts
type PackageClass = "box" | "brown_bag" | "white_bag";

interface PoseResponse {
  detected: boolean;
  class_name: PackageClass | null;         // non-null iff detected === true
  pick_base?: [number, number, number];    // mm, robot base frame; present iff detected === true
  confidence?: number;
  message?: string;
  // ...other existing fields unchanged
}
```

Python:
```python
resp = r.json()
if resp["detected"]:
    cls = resp["class_name"]          # one of "box" | "brown_bag" | "white_bag"
    x, y, z = resp["pick_base"]       # mm, robot base frame
    # ... route/pick by cls
else:
    assert resp["class_name"] is None
    # no pick this cycle
```

## Error / non-body responses (unchanged)
- `409` — an estimation is already in progress (busy). Body: `{ "detail": "..." }` (no `class_name`).
- `503` — camera not live / no fresh frame. Body: `{ "detail": "..." }` (no `class_name`).

These are HTTP error statuses, not `detected` bodies — handle by status code, then retry / no-pick.
