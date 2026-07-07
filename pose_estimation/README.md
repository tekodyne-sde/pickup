# OAK-D Capture Studio

A high-performance GUI application for capturing synchronized RGB and depth datasets using the Luxonis OAK-D stereo camera. 

This app merges the DepthAI v3 pipeline architecture (for reliable, high-speed stereo capture) with a modern `CustomTkinter` interface. It generates lossless 1080p `.png` RGB images and raw 16-bit `.npy` depth matrices.

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

A **one-shot** script: it detects a parcel, computes the grasp pose, transforms it
into the UR5 base frame using the hand-eye calibration, and moves the robot to a
**standoff pose** above the pick point — then stops. There is **no place motion and
no gripper/suction actuation**.

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
The script prints the detected class, the camera-frame grasp, the computed
**base-frame target**, and a reachability check, then waits for confirmation:

```
Type 'yes' to move, anything else to abort:
```

- **Dry run (recommended first):** answer anything **other than** `yes`. The robot
  does **not** move; you can confirm the printed target sits over the parcel and is
  within reach.
- **Live move:** ensure the workspace is clear and the **e-stop is within reach**,
  then type `yes`. The UR5 executes a single blocking `moveL` to the standoff pose and
  the script exits.

### Configuration (top of `robot_pick.py`)
| Constant | Default | Meaning |
|---|---|---|
| `ROBOT_IP` | `192.168.1.10` | UR5 controller IP. |
| `STANDOFF_M` | `0.10` | Retract distance (m) back along the surface normal. Reduce only after aim is confirmed. |
| `MOVE_SPEED` / `MOVE_ACCEL` | `0.10` / `0.30` | TCP linear speed (m/s) and acceleration (m/s²). |
| `MAX_REACH_M`, `Z_MIN_M`, `Z_MAX_M` | `0.85`, `-0.30`, `1.00` | Soft envelope; a target outside it aborts the move. |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `1920` / `1080` | Must match the resolution the calibrated `K` was computed at. |

### How it works
1. Load `T_base_cam`, `K`, `D` from `handeye_result.npz`; use `K` for deprojection
   (replacing the OAK factory-default intrinsics).
2. Detect (YOLO-OBB) → mask (SAM ∩ OBB) → grasp pose in the camera frame.
3. Build the tool target: approach axis (tool **+Z**) = **−normal** (into the surface),
   origin retracted `STANDOFF_M` along **+normal**.
4. `T_base_target = T_base_cam @ T_cam_target`, converted to a `moveL` pose
   `[x, y, z, rx, ry, rz]` (meters, axis-angle radians).
5. Reachability guard → confirmation prompt → single `moveL` → exit.

### Notes & caveats
- **Verify tool orientation at the standoff first.** The approach convention assumes
  tool **+Z** is the approach axis; if your UR tool frame differs, the wrist will
  orient incorrectly.
- **Distortion `D` is not applied** (ideal-pinhole deprojection). Accurate near the
  frame center; expect cm-level error toward the edges.
- The hand-eye calibration must correspond to the same base frame the robot reports.
