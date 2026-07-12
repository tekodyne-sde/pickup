"""tune_camera.py — live focus/exposure tuning for the OAK-D RGB (CAM_A).

The pose service LOCKS focus + exposure at startup (see RGB_LENS_POSITION /
RGB_EXPOSURE_US / RGB_ISO in robot_pick.py) so a package arriving on the belt
neither triggers an autofocus hunt nor motion-blurs. Those values are specific to
the camera-to-belt distance, so they must be re-tuned whenever the camera is moved
or recalibrated with the robot.

This tool shows the live RGB stream with the current focus/exposure/ISO and a
focus-sharpness readout (variance of the Laplacian on a center crop — higher is
sharper). Adjust live, find the peak sharpness at the working distance, cap
exposure until a moving package is not blurred, then press 's' to print a
copy-paste block for robot_pick.py.

Run:
    python tune_camera.py

Keys:
    [ / ]   focus  (lens position) -/+        (0-255)
    - / =   exposure -/+  (500 us step)        (100-33000 us)
    ; / '   ISO -/+  (50 step)                 (100-1600)
    a       toggle auto focus + auto exposure (to compare against the lock)
    s       print current values for robot_pick.py
    q / ESC quit (also prints the current values)

Needs the OAK-D connected. This is a hardware tool — not part of the service and
not unit-tested.
"""

import os

os.environ.setdefault("DEPTHAI_DISABLE_CRASHDUMP_COLLECTION", "1")

import cv2
import depthai as dai

from robot_pick import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    RGB_LENS_POSITION,
    RGB_EXPOSURE_US,
    RGB_ISO,
)

# Adjustment steps / limits.
FOCUS_STEP, FOCUS_MIN, FOCUS_MAX = 1, 0, 255
EXP_STEP, EXP_MIN, EXP_MAX = 500, 100, 33000       # microseconds
ISO_STEP, ISO_MIN, ISO_MAX = 50, 100, 1600
DISPLAY_W, DISPLAY_H = 960, 540                     # preview window size
CROP_FRAC = 0.25                                    # center crop for the sharpness metric


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _sharpness(bgr):
    """Variance of the Laplacian on a center crop — higher means sharper focus."""
    h, w = bgr.shape[:2]
    cw, ch = int(w * CROP_FRAC), int(h * CROP_FRAC)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = bgr[y0:y0 + ch, x0:x0 + cw]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _print_values(lens, exp_us, iso):
    print("\n# --- paste into robot_pick.py ---")
    print(f"RGB_LENS_POSITION = {lens}")
    print(f"RGB_EXPOSURE_US = {exp_us}")
    print(f"RGB_ISO = {iso}")
    print("# --------------------------------\n")


def main():
    device = dai.Device()
    try:
        pipeline = dai.Pipeline(device)
        cam = pipeline.create(dai.node.Camera)
        cam.build(boardSocket=dai.CameraBoardSocket.CAM_A)
        # Start from the values the service will lock in.
        cam.initialControl.setManualFocus(RGB_LENS_POSITION)
        cam.initialControl.setManualExposure(RGB_EXPOSURE_US, RGB_ISO)

        rgb_out = cam.requestOutput((FRAME_WIDTH, FRAME_HEIGHT))
        rgb_q = rgb_out.createOutputQueue(maxSize=1, blocking=False)
        ctrl_q = cam.inputControl.createInputQueue()
        pipeline.start()

        lens, exp_us, iso, auto = RGB_LENS_POSITION, RGB_EXPOSURE_US, RGB_ISO, False
        print("[tune] live — adjust with the keys, 's' to print, 'q' to quit")

        def send_manual():
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lens)
            ctrl.setManualExposure(exp_us, iso)
            ctrl_q.send(ctrl)

        def send_auto():
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            ctrl.setAutoExposureEnable()
            ctrl_q.send(ctrl)

        while True:
            frame = rgb_q.get().getCvFrame()
            sharp = _sharpness(frame)

            view = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            mode = "AUTO (comparison)" if auto else "LOCKED"
            lines = [
                f"mode: {mode}",
                f"focus (lens): {lens}",
                f"exposure: {exp_us} us",
                f"iso: {iso}",
                f"sharpness: {sharp:8.1f}  (higher = sharper)",
            ]
            for i, text in enumerate(lines):
                y = 24 + i * 26
                cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 1, cv2.LINE_AA)
            # Center crop rectangle used for the sharpness metric.
            cx0 = int(DISPLAY_W * (0.5 - CROP_FRAC / 2))
            cy0 = int(DISPLAY_H * (0.5 - CROP_FRAC / 2))
            cx1 = int(DISPLAY_W * (0.5 + CROP_FRAC / 2))
            cy1 = int(DISPLAY_H * (0.5 + CROP_FRAC / 2))
            cv2.rectangle(view, (cx0, cy0), (cx1, cy1), (0, 255, 255), 1)
            cv2.imshow("tune_camera — [ ] focus  - = exposure  ; ' iso  a auto  s print  q quit", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("["):
                lens = _clamp(lens - FOCUS_STEP, FOCUS_MIN, FOCUS_MAX); auto = False; send_manual()
            elif key == ord("]"):
                lens = _clamp(lens + FOCUS_STEP, FOCUS_MIN, FOCUS_MAX); auto = False; send_manual()
            elif key == ord("-"):
                exp_us = _clamp(exp_us - EXP_STEP, EXP_MIN, EXP_MAX); auto = False; send_manual()
            elif key == ord("="):
                exp_us = _clamp(exp_us + EXP_STEP, EXP_MIN, EXP_MAX); auto = False; send_manual()
            elif key == ord(";"):
                iso = _clamp(iso - ISO_STEP, ISO_MIN, ISO_MAX); auto = False; send_manual()
            elif key == ord("'"):
                iso = _clamp(iso + ISO_STEP, ISO_MIN, ISO_MAX); auto = False; send_manual()
            elif key == ord("a"):
                auto = not auto
                (send_auto if auto else send_manual)()
            elif key == ord("s"):
                _print_values(lens, exp_us, iso)

        cv2.destroyAllWindows()
        _print_values(lens, exp_us, iso)
    finally:
        device.close()


if __name__ == "__main__":
    main()
