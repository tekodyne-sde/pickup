"""
capture_gui.py — OAK-D Dataset Capture GUI (v3 Pipeline)
========================================================
GUI for capturing datasets using Luxonis OAK-D camera.
Uses DepthAI v3 pipeline and CustomTkinter.
"""

import customtkinter as ctk
import cv2
import numpy as np
import threading
import time
import os
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

try:
    import depthai as dai
    HAS_DEPTHAI = True
except ImportError:
    HAS_DEPTHAI = False

from config import DEFAULT_VARIANT_NAME, DEFAULT_BATCH_NUMBER, DEFAULT_TARGET_COUNT, CAPTURE_INTERVAL_SEC, FRAME_WIDTH, FRAME_HEIGHT
from core.camera_pipeline import create_v3_pipeline
from core.file_io import save_data_pair

# App theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C_BG        = "#0f1117"
C_PANEL     = "#161b22"
C_CARD      = "#1c2130"
C_BORDER    = "#2d3748"
C_ACCENT    = "#3b82f6"
C_ACCENT2   = "#10b981"
C_WARN      = "#f59e0b"
C_DANGER    = "#ef4444"
C_TEXT      = "#e2e8f0"
C_MUTED     = "#64748b"
C_SUCCESS   = "#22c55e"

class CameraThread(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.active = True
        self.latest = {
            "rgb": None, "depth_raw": None, "depth_vis": None, "fps": 0.0
        }
        self._lock = threading.Lock()
        self._fps_times = []

    def run(self):
        if not HAS_DEPTHAI:
            self.app.log("depthai not installed. Running demo mode.")
            self._demo_loop()
            return
        try:
            self._oakd_loop()
        except Exception as e:
            self.app.log(f"OAK-D error: {e}")
            self.app.log("Falling back to demo mode.")
            self._demo_loop()

    def _oakd_loop(self):
        self.app.log("Connecting to OAK-D...")
        with dai.Device(maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
            self.app.log(f"OAK-D connected: {device.getDeviceName()}")
            self.app.set_status("connected")

            pipeline, rgb_out, depth_out = create_v3_pipeline(device)
            
            q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
            q_depth = depth_out.createOutputQueue(maxSize=4, blocking=False)
            
            pipeline.start()

            while self.active:
                in_rgb = q_rgb.tryGet()
                in_depth = q_depth.tryGet()

                if in_rgb is not None and in_depth is not None:
                    rgb_frame = in_rgb.getCvFrame()
                    depth_data = in_depth.getFrame()

                    # High-speed visualization logic from sand_core
                    depth_downscaled = depth_data[::4, ::4]
                    valid_depths = depth_downscaled[depth_downscaled > 0]
                    
                    if len(valid_depths) > 0:
                        min_depth, max_depth = np.percentile(valid_depths, [1, 99])
                        depth_clipped = np.clip(depth_data, min_depth, max_depth)
                        depth_normalized = np.interp(depth_clipped, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                        depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    else:
                        depth_vis = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

                    now = time.time()
                    self._fps_times.append(now)
                    self._fps_times = [t for t in self._fps_times if now - t < 1.0]
                    fps = len(self._fps_times)

                    with self._lock:
                        self.latest.update({
                            "rgb": rgb_frame,
                            "depth_raw": depth_data,
                            "depth_vis": depth_vis,
                            "fps": fps
                        })
                else:
                    time.sleep(0.001)

    def _demo_loop(self):
        self.app.set_status("demo")
        t = 0.0
        while self.active:
            t += 0.05
            h, w = FRAME_HEIGHT, FRAME_WIDTH
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:] = (20, 25, 35)
            cv2.putText(rgb, "DEMO MODE - No OAK-D Camera", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (80,130,255), 3)
            
            noise = np.random.randint(0, 60, (h, w), dtype=np.uint8)
            depth_vis = cv2.applyColorMap(noise, cv2.COLORMAP_JET)
            depth_raw = (noise.astype(np.uint16)) * 20

            with self._lock:
                self.latest.update({
                    "rgb": rgb, "depth_raw": depth_raw, "depth_vis": depth_vis, "fps": 30
                })
            time.sleep(0.033)

    def get_latest(self):
        with self._lock:
            return {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in self.latest.items()
            }

    def stop(self):
        self.active = False


class CaptureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OAK-D Capture Studio")
        self.geometry("1420x860")
        self.minsize(1100, 700)
        self.configure(fg_color=C_BG)

        self.capture_count = 0
        self.auto_active = False
        self.last_auto_time = 0
        self.cam_thread = None
        self.flash_until = 0

        self.opt_variant_name = ctk.StringVar(value=DEFAULT_VARIANT_NAME)
        self.opt_batch_number = ctk.StringVar(value=DEFAULT_BATCH_NUMBER)
        self.opt_output_dir = ctk.StringVar(value=str(Path.cwd() / "dataset"))
        self.opt_target_count = ctk.IntVar(value=DEFAULT_TARGET_COUNT)
        self.opt_interval = ctk.DoubleVar(value=CAPTURE_INTERVAL_SEC)
        self.opt_save_depth_png = ctk.BooleanVar(value=False)
        
        self.thumbnails = []

        self._build_layout()
        self._start_camera()
        self._update_loop()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left sidebar
        self.sidebar = ctk.CTkScrollableFrame(self, width=280, fg_color=C_PANEL, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self._build_sidebar()

        # Center - camera
        self.center = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        self.center.grid(row=0, column=1, sticky="nsew")
        self.center.grid_rowconfigure(0, weight=1)
        self.center.grid_columnconfigure(0, weight=1)
        self._build_center()

        # Right panel
        self.right = ctk.CTkFrame(self, width=260, fg_color=C_PANEL, corner_radius=0)
        self.right.grid(row=0, column=2, sticky="nsew")
        self._build_right()

    def _build_sidebar(self):
        p = self.sidebar
        
        title = ctk.CTkFrame(p, fg_color=C_CARD, corner_radius=8)
        title.pack(fill="x", padx=8, pady=8)
        ctk.CTkLabel(title, text="⬡ CAPTURE STUDIO", font=("Consolas", 13, "bold"), text_color=C_ACCENT).pack(pady=8)
        self.lbl_cam_status = ctk.CTkLabel(title, text="● CAMERA: CONNECTING", font=("Consolas", 10), text_color=C_WARN)
        self.lbl_cam_status.pack(pady=(0,8))

        ctk.CTkLabel(p, text="SESSION", font=("", 11, "bold"), text_color=C_MUTED).pack(anchor="w", padx=12, pady=(14,2))
        
        ctk.CTkLabel(p, text="Variant Name", font=("", 11), text_color=C_MUTED).pack(anchor="w", padx=12)
        ctk.CTkEntry(p, textvariable=self.opt_variant_name, fg_color=C_CARD).pack(fill="x", padx=10, pady=(2,6))

        ctk.CTkLabel(p, text="Batch Number", font=("", 11), text_color=C_MUTED).pack(anchor="w", padx=12)
        ctk.CTkEntry(p, textvariable=self.opt_batch_number, fg_color=C_CARD).pack(fill="x", padx=10, pady=(2,6))

        ctk.CTkLabel(p, text="Output Directory", font=("", 11), text_color=C_MUTED).pack(anchor="w", padx=12)
        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(2,6))
        ctk.CTkEntry(row, textvariable=self.opt_output_dir, fg_color=C_CARD, font=("Consolas", 10)).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="...", width=32, command=self._browse_folder).pack(side="right", padx=(4,0))

        ctk.CTkLabel(p, text="Target Count", font=("", 11), text_color=C_MUTED).pack(anchor="w", padx=12)
        ctk.CTkEntry(p, textvariable=self.opt_target_count, fg_color=C_CARD).pack(fill="x", padx=10, pady=(2,6))

        ctk.CTkLabel(p, text="CAPTURE", font=("", 11, "bold"), text_color=C_MUTED).pack(anchor="w", padx=12, pady=(14,2))
        
        ctk.CTkLabel(p, text="Auto-capture interval (sec)", font=("", 11), text_color=C_MUTED).pack(anchor="w", padx=12)
        slider = ctk.CTkSlider(p, from_=0.5, to=10.0, variable=self.opt_interval)
        slider.pack(fill="x", padx=12, pady=(2,0))
        lbl_int = ctk.CTkLabel(p, text="2.0s", font=("Consolas", 11))
        lbl_int.pack(anchor="e", padx=14)
        self.opt_interval.trace_add("write", lambda *_: lbl_int.configure(text=f"{self.opt_interval.get():.1f}s"))

        ctk.CTkSwitch(p, text="Save Depth PNG (Debug)", variable=self.opt_save_depth_png).pack(anchor="w", padx=12, pady=(10, 0))

    def _build_center(self):
        p = self.center

        self.preview_frame = ctk.CTkFrame(p, fg_color=C_BG, corner_radius=0)
        self.preview_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure((0,1), weight=1)

        self.rgb_label = ctk.CTkLabel(self.preview_frame, text="", fg_color=C_CARD)
        self.rgb_label.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")

        self.depth_label = ctk.CTkLabel(self.preview_frame, text="", fg_color=C_CARD)
        self.depth_label.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")

        ctrl = ctk.CTkFrame(p, fg_color=C_PANEL, height=64, corner_radius=0)
        ctrl.pack(fill="x")
        ctrl.pack_propagate(False)

        self.btn_capture = ctk.CTkButton(ctrl, text="● MANUAL CAPTURE", width=140, height=44, fg_color=C_ACCENT, font=("", 14, "bold"), command=self._manual_capture)
        self.btn_capture.pack(side="left", padx=12, pady=10)

        self.btn_auto = ctk.CTkButton(ctrl, text="⏱ START AUTO", width=110, height=44, fg_color=C_CARD, border_color=C_ACCENT, border_width=1, font=("", 13, "bold"), command=self._toggle_auto)
        self.btn_auto.pack(side="left", padx=4, pady=10)

        ctk.CTkButton(ctrl, text="🗑 RESET", width=100, height=44, fg_color="transparent", hover_color=C_DANGER, border_color=C_DANGER, border_width=1, command=self._reset_session).pack(side="right", padx=12, pady=10)
        ctk.CTkButton(ctrl, text="📂 OPEN FOLDER", width=120, height=44, fg_color="transparent", border_color=C_BORDER, border_width=1, command=self._open_folder).pack(side="right", padx=4, pady=10)

        self.status_bar = ctk.CTkLabel(p, text="Ready", height=22, fg_color=C_CARD, corner_radius=0, text_color=C_MUTED, font=("Consolas", 10), anchor="w")
        self.status_bar.pack(fill="x")

    def _build_right(self):
        p = self.right
        
        card = ctk.CTkFrame(p, fg_color=C_CARD, corner_radius=8)
        card.pack(fill="x", padx=8, pady=8)

        self.lbl_count = ctk.CTkLabel(card, text="0", font=("Consolas", 48, "bold"), text_color=C_ACCENT)
        self.lbl_count.pack(pady=(10,0))
        ctk.CTkLabel(card, text="images captured", font=("", 11), text_color=C_MUTED).pack()
        self.progress_bar = ctk.CTkProgressBar(card, height=6, progress_color=C_ACCENT, fg_color=C_BORDER)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=12, pady=(8,4))
        self.lbl_progress = ctk.CTkLabel(card, text="0 / 100", font=("Consolas", 10), text_color=C_MUTED)
        self.lbl_progress.pack(pady=(0,10))

        ctk.CTkLabel(p, text="RECENT", font=("Consolas", 9), text_color=C_MUTED).pack(anchor="w", padx=10, pady=(4,2))
        self.thumb_frame = ctk.CTkScrollableFrame(p, fg_color=C_CARD, corner_radius=8)
        self.thumb_frame.pack(fill="x", padx=8, pady=(0,8))
        self.thumb_labels = []
        for _ in range(3):
            lbl = ctk.CTkLabel(self.thumb_frame, text="", fg_color=C_BORDER, width=228, height=64, corner_radius=4)
            lbl.pack(pady=2)
            self.thumb_labels.append(lbl)

        ctk.CTkLabel(p, text="LOG", font=("Consolas", 9), text_color=C_MUTED).pack(anchor="w", padx=10, pady=(4,2))
        self.log_box = ctk.CTkTextbox(p, fg_color=C_CARD, border_color=C_BORDER, border_width=1, font=("Consolas", 10), text_color=C_TEXT)
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.log_box.configure(state="disabled")

    def _start_camera(self):
        self.cam_thread = CameraThread(self)
        self.cam_thread.start()

    def _update_loop(self):
        try:
            self._render_frame()
            self._handle_auto_capture()
        except Exception as e:
            print(f"Update loop error: {e}")
        self.after(33, self._update_loop)

    def _render_frame(self):
        if not self.cam_thread:
            return

        data = self.cam_thread.get_latest()
        rgb = data.get("rgb")
        depth = data.get("depth_vis")

        if rgb is None or depth is None:
            return

        fps = data.get("fps", 0)
        cv2.putText(rgb, f"FPS: {fps}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        self._update_preview(self.rgb_label, rgb)
        self._update_preview(self.depth_label, depth)

    def _update_preview(self, widget, frame):
        if len(frame.shape) == 3:
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            display = frame

        h, w = display.shape[:2]

        # Available drawing area
        widget.update_idletasks()
        aw = widget.winfo_width()
        ah = widget.winfo_height()

        if aw < 50 or ah < 50:
            aw = 640
            ah = 480

        scale = min(aw / w, ah / h)

        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        resized = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(resized)

        # Use Tk PhotoImage directly to avoid CTkImage cropping.
        photo = ImageTk.PhotoImage(img)

        widget.configure(image=photo)
        widget.image = photo

    def _handle_auto_capture(self):
        if not self.auto_active: return
        now = time.time()
        if now - self.last_auto_time >= self.opt_interval.get():
            self._do_capture()
            self.last_auto_time = now

    def _do_capture(self):
        if not self.cam_thread: return
        data = self.cam_thread.get_latest()
        rgb = data.get("rgb")
        depth_raw = data.get("depth_raw")

        if rgb is None or depth_raw is None:
            self.log("No valid frame available for capture.")
            return

        self.capture_count += 1
        n = self.capture_count
        var_name = self.opt_variant_name.get()
        batch = self.opt_batch_number.get()
        out_root = self.opt_output_dir.get()

        success, paths = save_data_pair(rgb, depth_raw, out_root, var_name, batch, n)
        
        if success:
            if self.opt_save_depth_png.get():
                depth_vis = data.get("depth_vis")
                if depth_vis is not None:
                    png_path = Path(out_root) / var_name / "depth" / f"batch_{batch}_{n:04d}_debug.png"
                    cv2.imwrite(str(png_path), depth_vis)
                    paths = paths + (str(png_path),)

            self._log_csv(out_root, var_name, batch, n, paths)
            self.flash_until = time.time() + 0.15
            self._update_count_ui()
            self.log(f"Captured #{n:04d} (Batch {batch})")
            self.status_bar.configure(text=f"Captured #{n:04d} to {out_root}/{var_name}")
            
            # thumbnail
            small = cv2.resize(rgb, (228, 64))
            pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            self.thumbnails.insert(0, ctk.CTkImage(pil, size=(228, 64)))
            self.thumbnails = self.thumbnails[:3]
            for i, lbl in enumerate(self.thumb_labels):
                if i < len(self.thumbnails):
                    lbl.configure(image=self.thumbnails[i])

        if self.auto_active and n >= self.opt_target_count.get():
            self._toggle_auto()
            self.log("Target reached. Auto-capture stopped.")

    def _log_csv(self, root, var_name, batch, count, paths):
        log_dir = Path(root) / var_name
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "capture_log.csv"
        write_head = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_head: w.writerow(["batch", "capture_id", "timestamp", "files"])
            w.writerow([batch, count, datetime.now().isoformat(), "; ".join(paths)])

    def _update_count_ui(self):
        n = self.capture_count
        t = self.opt_target_count.get()
        self.lbl_count.configure(text=str(n))
        self.progress_bar.set(min(n / max(t, 1), 1.0))
        self.lbl_progress.configure(text=f"{n} / {t}")

    def _manual_capture(self):
        self._do_capture()

    def _toggle_auto(self):
        self.auto_active = not self.auto_active
        self.last_auto_time = 0
        if self.auto_active:
            self.btn_auto.configure(fg_color=C_ACCENT, text="⏱ STOP AUTO")
            self.log("Auto-capture started")
        else:
            self.btn_auto.configure(fg_color=C_CARD, text="⏱ START AUTO")
            self.log("Auto-capture stopped")

    def _reset_session(self):
        if messagebox.askyesno("Reset", "Reset capture counter to 0?"):
            self.capture_count = 0
            self.auto_active = False
            self.btn_auto.configure(fg_color=C_CARD, text="⏱ START AUTO")
            self._update_count_ui()
            self.log("Session counter reset.")

    def _browse_folder(self):
        d = filedialog.askdirectory(initialdir=self.opt_output_dir.get())
        if d: self.opt_output_dir.set(d)

    def _open_folder(self):
        p = Path(self.opt_output_dir.get()) / self.opt_variant_name.get()
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(str(p))

    def log(self, msg):
        self.after(0, lambda: self._log_sync(msg))

    def _log_sync(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{ts} {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, st):
        if st == "connected":
            self.lbl_cam_status.configure(text="● CAMERA: CONNECTED", text_color=C_SUCCESS)
        else:
            self.lbl_cam_status.configure(text="● CAMERA: DEMO MODE", text_color=C_WARN)

    def _on_close(self):
        if self.cam_thread: self.cam_thread.stop()
        self.destroy()

if __name__ == "__main__":
    app = CaptureApp()
    app.mainloop()
