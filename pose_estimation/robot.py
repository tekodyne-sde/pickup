"""
robot.py — UR5 pick API. Calls pose_service for the pick point, then moves.

ALL UNITS ARE MILLIMETERS (positions, config, results) — matching the
pose_service API, so its pick_base flows straight through with no conversion.
The UR controller itself speaks meters; that translation happens in exactly
one place, at the ur_rtde boundary (_move_segment / _tcp_mm). Rotation values
(rx, ry, rz) remain radians (rotation vector).

Run (own process, alongside pose_service):
    uvicorn robot:app --host 127.0.0.1 --port 8002

API:
    POST /pick  body (optional): {"standoff_mm": 30, "dry_run": false}
        full cycle: POST pose_service /pose -> staged move to the standoff
        above the pick point. dry_run=true plans + checks only (no motion).
    GET  /tcp   current TCP pose (mm + radians).

Staged, axis-aligned path at fixed orientation (vertical suction approach):
    W1 vertical to the traverse height (derived so the TOOL TOP clears the
       camera, floored at the standoff height)
    W2 horizontal to above the pick
    W3 vertical descend to the standoff
Each segment is verified (moveL result + protective stop); motion stops at the
standoff. No gripper/suction actuation.

Also usable as a library: robot.pick([-311.5, -454.0, 252.3], standoff_mm=30)
(pick point in MM, base frame). Raises ValueError (unsafe plan) or
RuntimeError (busy / protective stop) — robot NOT moved in either case.

Manual test:  python robot.py -311.5 -454.0 252.3   (plans, asks, then moves)
"""

import threading

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# ============================================================
# CONFIG — all lengths in MILLIMETERS
# ============================================================
ROBOT_IP = "192.168.1.10"
HANDEYE_NPZ = "handeye_result.npz"  # camera height for the clearance check
POSE_SERVICE_URL = "http://127.0.0.1:8001"  # ML service (pose_service.py)

MOVE_SPEED_MM_S = 100      # (conservative)
MOVE_ACCEL_MM_S2 = 300     # (conservative)

# Ceiling for the horizontal-traverse height; the actual height is derived so
# the TOP of the tool clears the camera (floored at the standoff height).
Z_TRAVERSE_MM = 500.0
# Highest point of tool/wrist ABOVE the TCP tip when pointing down. MEASURED.
TOOL_TOP_ABOVE_TCP_MM = 360.0
CAM_CLEARANCE_MM = 80.0    # min gap between tool top and camera

# Soft reachability envelope (UR5 max reach ~850 mm) — sanity guard only.
MAX_REACH_MM = 850.0
Z_MIN_MM, Z_MAX_MM = -300.0, 1000.0

REACHED_TOL_MM = 10.0      # ended within this of the target = reached

# calibration npz is in meters (internal ML convention) -> mm once, here
CAM_HEIGHT_MM = float(np.load(HANDEYE_NPZ)["T_base_cam"][2, 3]) * 1000.0

_robot_lock = threading.Lock()  # ponytail: one robot -> one lock


# ============================================================
# ur_rtde boundary — the ONLY place meters exist
# ============================================================
def _tcp_mm(tcp_pose_m):
    """UR TCP pose (meters) -> [x,y,z in mm, rx,ry,rz in rad]."""
    return [float(tcp_pose_m[0]) * 1000.0, float(tcp_pose_m[1]) * 1000.0,
            float(tcp_pose_m[2]) * 1000.0,
            float(tcp_pose_m[3]), float(tcp_pose_m[4]), float(tcp_pose_m[5])]


def _move_segment(rtde_c, rtde_r, label, wp_mm):
    """Blocking moveL for one waypoint (mm); verify it completed."""
    print(f"  [{label}] moveL -> xyz=[{wp_mm[0]:.1f}, {wp_mm[1]:.1f}, {wp_mm[2]:.1f}] mm")
    target_m = [wp_mm[0] / 1000.0, wp_mm[1] / 1000.0, wp_mm[2] / 1000.0,
                wp_mm[3], wp_mm[4], wp_mm[5]]
    try:
        ok = rtde_c.moveL(target_m, MOVE_SPEED_MM_S / 1000.0, MOVE_ACCEL_MM_S2 / 1000.0)
    except Exception as exc:  # controller-side stop / disconnect
        print(f"  [{label}] moveL raised: {exc}")
        return False, None

    stopped = _is_protective_stopped(rtde_r)
    achieved = _tcp_mm(rtde_r.getActualTCPPose())
    if (ok is False) or stopped:
        print(f"  [{label}] FAILED (moveL returned {ok}, protective_stop={stopped})")
        return False, achieved
    print(f"  [{label}] done. TCP (mm): {np.round(achieved, 1)}")
    return True, achieved


# ============================================================
# Helpers (mm)
# ============================================================
def within_envelope(xyz_mm):
    x, y, z = float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2])
    radius = float(np.hypot(x, y))
    ok = (radius <= MAX_REACH_MM) and (Z_MIN_MM <= z <= Z_MAX_MM)
    return ok, radius


def _is_protective_stopped(rtde_r):
    fn = getattr(rtde_r, "isProtectiveStopped", None)
    try:
        return bool(fn()) if fn is not None else False
    except Exception:
        return False


def plan_path(pick_base_mm, standoff_mm, cur_pose_mm):
    """Staged, axis-aligned path (mm). Raises ValueError if unsafe."""
    standoff = [float(pick_base_mm[0]), float(pick_base_mm[1]),
                float(pick_base_mm[2]) + float(standoff_mm)]

    # Tool must reach the standoff anyway, so traversing AT standoff height adds
    # no extra camera exposure — floor the camera-safe cap there.
    cam_safe = CAM_HEIGHT_MM - CAM_CLEARANCE_MM - TOOL_TOP_ABOVE_TCP_MM
    z_traverse = min(Z_TRAVERSE_MM, max(cam_safe, standoff[2]))
    gap = CAM_HEIGHT_MM - (z_traverse + TOOL_TOP_ABOVE_TCP_MM)

    warnings = []
    if gap < CAM_CLEARANCE_MM:
        warnings.append(f"tool-top gap to camera only {gap:.0f} mm "
                        f"(< {CAM_CLEARANCE_MM:.0f} mm clearance) — forced by the "
                        f"standoff height; watch the move closely")
    if standoff[2] > z_traverse:
        raise ValueError(f"standoff z={standoff[2]:.1f} mm above traverse ceiling "
                         f"Z_TRAVERSE_MM={Z_TRAVERSE_MM:.1f} mm — misconfigured")

    ori = [float(cur_pose_mm[3]), float(cur_pose_mm[4]), float(cur_pose_mm[5])]
    cx, cy = float(cur_pose_mm[0]), float(cur_pose_mm[1])
    waypoints = [
        ("W1 vertical-to-traverse", [cx, cy, z_traverse, *ori]),
        ("W2 traverse-above-pick", [standoff[0], standoff[1], z_traverse, *ori]),
        ("W3 descend-to-standoff", [*standoff, *ori]),
    ]
    for label, wp in waypoints:
        ok, radius = within_envelope(wp)
        if not ok:
            raise ValueError(f"{label} xyz={[round(v, 1) for v in wp[:3]]} r={radius:.0f} mm "
                             f"is outside the safe envelope")
    return waypoints, z_traverse, gap, warnings


# ============================================================
# Public API (library) — all mm
# ============================================================
def get_tcp():
    """Current TCP pose [x,y,z in mm, rx,ry,rz in rad], base frame."""
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    try:
        return _tcp_mm(rtde_r.getActualTCPPose())
    finally:
        rtde_r.disconnect()


def pick(pick_base_mm, standoff_mm=30.0, dry_run=False):
    """Plan (dry_run=True) or execute the staged move. pick_base in MM.

    Returns a dict: current_tcp, z_traverse_mm, cam_gap_mm, warnings, waypoints,
    executed; after execution also segments, reached, delta_mm, final_tcp.
    """
    if not _robot_lock.acquire(blocking=False):
        raise RuntimeError("robot busy with another move")
    try:
        rtde_r = RTDEReceiveInterface(ROBOT_IP)
        try:
            cur_pose = _tcp_mm(rtde_r.getActualTCPPose())
            if _is_protective_stopped(rtde_r):
                raise RuntimeError("robot is in a PROTECTIVE STOP — clear it on the pendant")

            waypoints, z_traverse, gap, warnings = plan_path(pick_base_mm, standoff_mm, cur_pose)
            result = {
                "current_tcp": cur_pose,
                "z_traverse_mm": round(z_traverse, 1),
                "cam_gap_mm": round(gap),
                "speed_mm_s": MOVE_SPEED_MM_S, "accel_mm_s2": MOVE_ACCEL_MM_S2,
                "warnings": warnings,
                "waypoints": [{"label": label,
                               "xyz": [round(v, 1) for v in wp[:3]],
                               "r_mm": round(float(np.hypot(wp[0], wp[1])))}
                              for label, wp in waypoints],
                "executed": False,
            }
            if dry_run:
                return result

            print(f"\nExecuting staged move to pick_base={list(pick_base_mm)} mm ...")
            rtde_c = RTDEControlInterface(ROBOT_IP)
            try:
                segments, achieved, reached_all = [], None, True
                for label, wp in waypoints:
                    ok, achieved = _move_segment(rtde_c, rtde_r, label, wp)
                    segments.append({"label": label, "ok": ok,
                                     "achieved_tcp": achieved})
                    if not ok:
                        reached_all = False
                        break

                delta = None
                if reached_all and achieved is not None:
                    target = np.array(waypoints[-1][1][:3])
                    delta = float(np.linalg.norm(np.array(achieved[:3]) - target))
                result.update({
                    "executed": True,
                    "segments": segments,
                    "reached": bool(reached_all and delta is not None and delta < REACHED_TOL_MM),
                    "delta_mm": round(delta, 1) if delta is not None else None,
                    "final_tcp": achieved,
                })
                return result
            finally:
                try:
                    rtde_c.stopScript()
                except Exception:
                    pass
                rtde_c.disconnect()
        finally:
            rtde_r.disconnect()
    finally:
        _robot_lock.release()


# ============================================================
# API — full pick cycle: pose_service /pose -> staged move
# ============================================================
app = FastAPI(title="UR5 pick API")


class PickCycleRequest(BaseModel):
    standoff_mm: float = Field(default=30.0, ge=0.0, le=500.0)
    dry_run: bool = False


@app.get("/tcp")
def api_tcp():
    return {"tcp": get_tcp()}


@app.post("/pick")
def api_pick(req: PickCycleRequest = PickCycleRequest()):
    """Ask pose_service for a pick point (mm), then execute the staged move."""
    try:
        r = requests.post(f"{POSE_SERVICE_URL}/pose", timeout=30)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=503,
                            detail=f"pose service not reachable at {POSE_SERVICE_URL}: {exc}")
    if r.status_code != 200:
        detail = r.json().get("detail", r.text) if r.text else str(r.status_code)
        raise HTTPException(status_code=502, detail=f"pose service error: {detail}")
    pose = r.json()
    if not pose.get("detected"):
        return {"detected": False, "moved": False, "pose": pose}

    # pose_service pick_base is already mm — straight through, no conversion.
    try:
        motion = pick(pose["pick_base"], standoff_mm=req.standoff_mm, dry_run=req.dry_run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"detected": True, "moved": motion["executed"],
            "pose": pose, "motion": motion}


# ============================================================
# Manual test: python robot.py X Y Z  (mm, base frame)
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        raise SystemExit("usage: python robot.py X Y Z   (pick point, MM, base frame)")
    target = [float(v) for v in sys.argv[1:4]]

    plan = pick(target, dry_run=True)
    print(f"Current TCP (mm): {np.round(plan['current_tcp'], 1)}")
    print(f"Traverse z={plan['z_traverse_mm']} mm, camera gap {plan['cam_gap_mm']} mm")
    for wp in plan["waypoints"]:
        print(f"  {wp['label']}: xyz={wp['xyz']} mm  r={wp['r_mm']} mm")
    for w in plan["warnings"]:
        print(f"  [warn] {w}")

    print("\nEnsure the workspace is clear and the e-stop is within reach.")
    if input("Type 'yes' to move, anything else to abort: ").strip().lower() == "yes":
        result = pick(target)
        print(f"\nreached={result['reached']} delta={result['delta_mm']} mm")
        print(f"Final TCP (mm): {np.round(result['final_tcp'], 1)}")
    else:
        print("Aborted — robot NOT moved.")
