# OAK-D Capture Studio

A high-performance GUI application for capturing synchronized RGB and depth datasets using the Luxonis OAK-D stereo camera. 

This app merges the DepthAI v3 pipeline architecture (for reliable, high-speed stereo capture) with a modern `CustomTkinter` interface. It generates lossless 1080p `.png` RGB images and raw 16-bit `.npy` depth matrices.

## Setup on a New Machine (read this first)

Everything a colleague needs to go from a fresh clone to a running tool.

### 1. Prerequisites
- **Python 3.12** (developed on 3.12.6). 3.10+ should work; stick to 3.12 to match.
- **Git**, to clone the repo.
- **A GPU is optional.** `ultralytics` pulls in PyTorch automatically and runs on CPU if no CUDA GPU is present — inference is just slower.

### 2. Files that must be present (not always in git)
These are required at runtime and are large binaries — confirm they came across with the clone/copy, or ask for them separately:
| File | Needed by | Notes |
|---|---|---|
| `best.pt` | `pose.py`, `robot_pick.py`, `pose_service.py` | YOLO-OBB parcel detector. **Must be supplied** — not downloadable. |
| `sam2.1_t.pt` | same | SAM segmentation model. Auto-downloads (~40 MB) on first run if missing. |
| `handeye_result.npz` | `robot_pick.py`, `pose_service.py` | Hand-eye calibration (`T_base_cam`, `K`, `D`). **Cell-specific — must be supplied.** Not needed for `capture_gui.py` or `pose.py`. |

### 3. Create the environment and install
**Windows (automatic):** double-click `setup.bat`, or from a terminal in this folder:
```cmd
setup.bat
```
This creates `venv\` and installs everything in `requirements.txt`.

**Manual (any OS):**
```cmd
python -m venv venv
venv\Scripts\activate        # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```
First install is large (PyTorch + Open3D + DepthAI) — allow a few minutes.

### 4. Hardware
- **OAK-D** stereo camera connected via USB (required by every tool except a pure model test).
- **UR5 robot** reachable on the network in **Remote Control** mode — only for `robot_pick.py`.

### 5. Pick an entry point
| Command | What it does | Needs camera | Needs robot |
|---|---|---|---|
| `python capture_gui.py` | GUI to capture RGB+depth datasets | ✅ | — |
| `python pose.py` | Live 6-DoF grasp pose, no robot motion | ✅ | — |
| `python robot_pick.py` | One-shot: detect → move UR5 to standoff above pick | ✅ | ✅ |
| `python pose_service.py` | FastAPI pose service on port 8001 (see `POSE_SERVICE_API.md`) | ✅ | — |

Details for each are in the sections below. If you just want to confirm the environment works without hardware, run `python -c "import torch, ultralytics, depthai, open3d; print('ok')"`.

## Features
- **Live Side-by-Side View:** Native 1080p RGB feed paired with a real-time depth heatmap.
- **Batch Tracking:** Configurable Variant Name and Batch Number parameters to seamlessly organize your datasets on disk.
- **Auto & Manual Capture:** Choose between triggering single shots manually or using an interval-based auto capture.
- **No Input Lag:** The camera and file-saving operations run on an isolated background thread to keep the UI perfectly responsive.

## Installation & Setup

We recommend using a Python virtual environment to install dependencies.

### Automatic Setup (Windows)
1. Double-click the `setup.bat` file in this directory. This script will automatically create a virtual environment (`venv`) and install all required packages.
2. To run the app after setup, open Command Prompt or PowerShell in this directory and type:
   ```cmd
   venv\Scripts\activate
   python capture_gui.py
   ```

### Manual Setup
1. Open a terminal in the `cap` folder.
2. Create a virtual environment:
   ```cmd
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Command Prompt:** `venv\Scripts\activate.bat`
   - **PowerShell:** `.\venv\Scripts\Activate.ps1`
4. Install the dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

## Dataset Structure
Datasets are formatted to be easily loaded into Machine Learning pipelines:

```text
dataset/
└── <variant_name>/
    ├── capture_log.csv
    ├── rgb/
    │   ├── batch_<batch_no>_0001.png
    │   └── ...
    └── depth/
        ├── batch_<batch_no>_0001.npy
        └── ...
```

- **RGB (`.png`)**: Lossless color images.
- **Depth (`.npy`)**: Raw 16-bit numpy matrices containing true millimeter depth measurements.

## Running the App
1. Ensure your OAK-D camera is securely connected via USB.
2. Activate your virtual environment and run `python capture_gui.py`.
3. The UI will appear. Configure your Variant Name, Batch Number, Target Count, and capture interval in the left sidebar.
4. Click **MANUAL CAPTURE** for single shots, or **START AUTO** to capture at the chosen interval.

---

## Grasp Pose Estimation (`pose.py`)

A standalone live tool that streams RGB + depth, detects a parcel with YOLO-OBB
(`best.pt`), segments it with SAM (`sam2.1_t.pt`), deprojects the masked depth into
a point cloud, finds the flattest suction patch via RANSAC plane fitting, and
produces a 6-DoF grasp pose (position + surface normal + quaternion) **in the camera
optical frame**. It does not move any robot.

```cmd
python pose.py
```

Type `capture` (or `c`) + Enter in the terminal to snapshot the current frame, print
the pose, save an annotated image, and open an Open3D 3D view. Type `quit` to exit.

---

## Robot Pick Move — UR5 (`robot_pick.py`)

A **one-shot** script: it detects a parcel, computes the pick point, transforms it
into the UR5 base frame using the hand-eye calibration, and moves the robot to a
**standoff pose 10 cm above the pick** via a collision-safe staged path — then stops.
There is **no place motion and no gripper/suction actuation**.

The tool orientation is **held constant** (whatever the robot currently has — no
reorientation), and the approach is a strictly axis-aligned **down → over → down**
path. A vacuum cup seals on a straight-down approach, so matching the surface tilt is
unnecessary and only risks sweeping the arm into fixtures.

### Prerequisites
- OAK-D connected via USB, and a parcel in view.
- `handeye_result.npz` present in this folder. It must contain:
  - `T_base_cam` — 4×4 base←camera transform (eye-to-hand; camera fixed above the base),
  - `K` — 3×3 intrinsics **at 1920×1080**,
  - `D` — distortion coefficients (loaded but not currently applied).
- `best.pt` present (`sam2.1_t.pt` auto-downloads on first run, ~40 MB).
- UR5 reachable on the network (default `192.168.1.10`) and switched to
  **Remote Control** mode.
- The robot's **TCP configured on the pendant at the suction tool tip** — targets are
  commanded directly to the TCP.
- Install the robot driver (already in `requirements.txt`):
  ```cmd
  pip install ur_rtde
  ```

### Running
```cmd
python robot_pick.py
```
The script closes the camera, connects to the robot (read-only at first), prints the
pick point, the computed **staged waypoints**, and a per-waypoint reachability check,
then waits for confirmation:

```
Type 'yes' to move, anything else to abort:
```

- **Dry run (recommended first):** answer anything **other than** `yes`. The robot
  does **not** move. Confirm each waypoint sits where you expect and that the traverse
  height (`W2`) clearly clears the camera and stand. *(Connecting is read-only, so a
  dry run does need the robot reachable and `ur_rtde` installed.)*
- **Live move:** ensure the workspace is clear and the **e-stop is within reach**,
  then type `yes`. The UR5 runs the three staged moves and reports whether it actually
  reached the standoff, then exits.

### Staged path
All three segments hold the starting orientation and are strictly axis-aligned:

1. **W1 — vertical to traverse:** move straight up/down at the current XY to `Z_TRAVERSE_M`.
2. **W2 — traverse:** move horizontally to directly above the pick, still at `Z_TRAVERSE_M`.
3. **W3 — descend:** move straight down to the 10 cm standoff over the pick.

Each segment is a blocking `moveL` that is verified against its return value and the
robot's protective-stop state; on a stop the script reports the failing segment and the
achieved pose instead of assuming success.

### Configuration (top of `robot_pick.py`)
| Constant | Default | Meaning |
|---|---|---|
| `ROBOT_IP` | `192.168.1.10` | UR5 controller IP. |
| `STANDOFF_M` | `0.10` | Height (m) to stop **above** the pick (vertical approach). |
| `Z_TRAVERSE_M` | `0.50` | Safe horizontal-traverse height (base Z). **Must clear the camera/stand and stay above the standoff** — the script auto-rejects a value within 5 cm of the camera height. Confirm in a dry run. |
| `MOVE_SPEED` / `MOVE_ACCEL` | `0.10` / `0.30` | TCP linear speed (m/s) and acceleration (m/s²). |
| `MAX_REACH_M`, `Z_MIN_M`, `Z_MAX_M` | `0.85`, `-0.30`, `1.00` | Soft envelope; any waypoint outside it aborts the move. |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `1920` / `1080` | Must match the resolution the calibrated `K` was computed at. |

### How it works
1. Load `T_base_cam`, `K`, `D` from `handeye_result.npz`; use `K` for deprojection
   (replacing the OAK factory-default intrinsics).
2. Detect (YOLO-OBB) → mask (SAM ∩ OBB) → grasp **position** in the camera frame
   (the surface normal is computed but **not** used for orientation).
3. Transform the pick point to the base frame: `pick_base = T_base_cam @ [position, 1]`;
   `standoff = pick_base + [0, 0, STANDOFF_M]`.
4. Read the current TCP orientation and hold it; build the down→over→down waypoints.
5. Per-waypoint reachability guard + camera-clearance guard → confirmation prompt →
   verified staged `moveL` sequence → report reached/not → exit.

### Notes & caveats
- **Tune `Z_TRAVERSE_M` for your cell.** The default 0.50 m clears the camera
  (~0.695 m above the base) with ~0.20 m margin, but verify the **horizontal** traverse
  at that height doesn't clip the stand — check the printed `W2` in a dry run.
- **Orientation is held, not corrected.** Ensure the robot's current pose already points
  the cup usefully at the workspace before running; the script won't reorient it.
- **Distortion `D` is not applied** (ideal-pinhole deprojection). Accurate near the
  frame center; expect cm-level error toward the edges.
- The hand-eye calibration must correspond to the same base frame the robot reports.
