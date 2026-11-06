"""Tkinter main window: live view, acquisition modes, settings, saving.

Implements the README requirements that concern the acquisition UI:
mode switching, save controls (folder/identifier/format), camera
settings (auto/manual exposure, fps, gain, black level, color space,
white balance), live display with ROI zoom, and the Streaming drop
counter. Perspective correction and intensity calibration hooks into
the display and variant saving are layered on in later steps.

Acquisition runs on a worker thread; frames are handed over via the
``on_frame`` callbacks (latest-wins) and completion/errors via a queue
drained by the Tk event loop.
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageTk

from ..acquisition import (AcquisitionError, AutoExposureController,
                           FrameBundle, SnapshotSession, StreamingSession,
                           StreamingStats, TimeSeriesSession, TimeSeriesStats,
                           acquisition_timestamp, camera_fps_max,
                           make_processor, minimum_interval_s)
from ..acquisition.processing import COLOR_SPACES
from ..camera import CameraService
from ..config import Configuration, ConfigurationError, max_fps
from ..perspective import PerspectiveTransform
from ..profiles import (extract_profiles, gaussian_fit, gaussian_overlay,
                        profile_stats)
from . import display

#: Fixed display canvas size, in pixels.
CANVAS_SIZE = (720, 540)
#: UI refresh period for the event/queue poll, in ms.
_POLL_MS = 33

MODE_SNAPSHOT = "Snapshot"
MODE_STREAMING = "Streaming"
MODE_TIME_SERIES = "Time Series"
MODES = (MODE_SNAPSHOT, MODE_STREAMING, MODE_TIME_SERIES)


class MainWindow:
    """Root application window for one connected camera."""

    def __init__(self, root: tk.Tk, service: CameraService, camera) -> None:
        self._root = root
        self._service = service
        self._camera = camera
        self._config = Configuration()
        self._processor = make_processor(self._config.color_space,
                                         self._config.wb_gains)
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._latest_bundle: Optional[FrameBundle] = None
        self._latest_lock = threading.Lock()
        self._roi = None                # (x0, y0, x1, y1) in image pixels
        self._geom: Optional[display.ViewGeometry] = None
        self._drawing = False
        self._draw_start = (0, 0)
        self._draw_rect = None
        self._live_rect = (0, 0, 0, 0)
        self._frame_counter = 0
        self._photo: Optional[ImageTk.PhotoImage] = None
        # Auto-exposure values frozen from the last Streaming run (README:
        # switching Streaming -> Time Series freezes them as manual).
        self._last_auto: Optional[tuple] = None
        self._live_ae: Optional[tuple] = None
        self._perspective: Optional[PerspectiveTransform] = None
        self._build_ui()
        self._poll_events()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._root.title("Chromoscope DAQ")
        self._build_menu()
        main = ttk.Frame(self._root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        self._build_mode_frame(left)
        self._build_settings_frame(left)
        self._build_acquisition_frame(left)
        self._build_canvas(right)
        self._on_mode_change()  # initial state: Streaming selected

    def _build_menu(self) -> None:
        menu = tk.Menu(self._root)
        correction = tk.Menu(menu, tearoff=False)
        correction.add_command(label="Perspective correction...",
                               command=self._open_perspective)
        menu.add_cascade(label="Correction", menu=correction)
        self._root.config(menu=menu)

    def _open_perspective(self) -> None:
        from .perspective_dialog import PerspectiveDialog
        PerspectiveDialog(
            self._root, self._perspective,
            on_apply=self._set_perspective,
            get_last_frame=lambda: getattr(self, "_last_bundle", None))

    def _set_perspective(self, transform: PerspectiveTransform) -> None:
        self._perspective = transform
        # The corrected display mode is now available (README) — switch
        # to it so the user immediately sees the result.
        self._corrected_radio.config(state="normal")
        self._display_mode_var.set("corrected")
        if getattr(self, "_last_bundle", None) is not None:
            self._render(self._last_bundle)
        self._status_var.set(
            "Perspective correction defined: "
            f"{len(transform.source_points)} points, "
            f"X {transform.x_range} mm, Y {transform.y_range} mm.")

    def _build_mode_frame(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Acquisition mode", padding=6)
        box.pack(fill="x", pady=(0, 6))
        self._mode_var = tk.StringVar(value=MODE_STREAMING)
        for mode in MODES:
            ttk.Radiobutton(box, text=mode, value=mode,
                            variable=self._mode_var,
                            command=self._on_mode_change).pack(anchor="w")

    def _build_settings_frame(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Settings", padding=6)
        box.pack(fill="x", pady=(0, 6))
        row = 0

        self._auto_exposure_var = tk.BooleanVar(
            value=self._config.auto_exposure)
        ttk.Checkbutton(box, text="Auto exposure (Streaming)",
                        variable=self._auto_exposure_var).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self._exposure_var = tk.StringVar(
            value=str(int(self._config.exposure_us)))
        self._fps_var = tk.StringVar(value=str(self._config.fps))
        self._gain_var = tk.StringVar(value=str(self._config.gain_db))
        self._black_var = tk.StringVar(value=str(self._config.black_level))
        for label, var, note in (
                ("Exposure [us]", self._exposure_var, ""),
                ("Frame rate [fps]", self._fps_var, ""),
                ("Gain [dB]", self._gain_var, ""),
                ("Black level", self._black_var, "")):
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(box, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w")
            row += 1

        ttk.Label(box, text="Color space").grid(row=row, column=0,
                                                sticky="w")
        self._color_var = tk.StringVar(value=self._config.color_space)
        ttk.Combobox(box, textvariable=self._color_var, width=10,
                     values=list(COLOR_SPACES),
                     state="readonly").grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(box, text="WB gains R/G/B").grid(row=row, column=0,
                                                   sticky="w")
        wb_frame = ttk.Frame(box)
        wb_frame.grid(row=row, column=1, sticky="w")
        self._wb_vars = tuple(tk.StringVar(value=str(g))
                              for g in self._config.wb_gains)
        for var in self._wb_vars:
            ttk.Entry(wb_frame, textvariable=var, width=5).pack(
                side="left", padx=2)
        row += 1

        ttk.Button(box, text="Apply settings", command=self._apply_settings
                   ).grid(row=row, column=0, columnspan=2, sticky="we",
                          pady=(4, 0))

    def _build_acquisition_frame(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Acquisition", padding=6)
        box.pack(fill="x", pady=(0, 6))

        buttons = ttk.Frame(box)
        buttons.pack(fill="x")
        self._start_button = ttk.Button(buttons, text="Start",
                                        command=self._start)
        self._start_button.pack(side="left")
        self._stop_button = ttk.Button(buttons, text="Stop",
                                       command=self._stop, state="disabled")
        self._stop_button.pack(side="left", padx=4)

        self._save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Save data", variable=self._save_var
                        ).pack(anchor="w")

        path_frame = ttk.Frame(box)
        path_frame.pack(fill="x")
        self._dir_var = tk.StringVar(value="data")
        ttk.Entry(path_frame, textvariable=self._dir_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(path_frame, text="...", width=3,
                   command=self._browse).pack(side="left")

        id_frame = ttk.Frame(box)
        id_frame.pack(fill="x")
        ttk.Label(id_frame, text="Identifier:").pack(side="left")
        self._identifier_var = tk.StringVar(value="run")
        ttk.Entry(id_frame, textvariable=self._identifier_var,
                  width=18).pack(side="left", padx=4)
        ttk.Label(id_frame, text="Format:").pack(side="left")
        self._format_var = tk.StringVar(value="tif")
        self._format_combo = ttk.Combobox(
            id_frame, textvariable=self._format_var, width=8,
            values=("tif", "jpg"), state="readonly")
        self._format_combo.pack(side="left")
        self._last_fmt = "tif"

        ttk.Label(box, text="Streaming duration [s] "
                            "(0 = until Stop):").pack(anchor="w",
                                                      pady=(4, 0))
        self._duration_var = tk.StringVar(value="0")
        ttk.Entry(box, textvariable=self._duration_var, width=10).pack(
            anchor="w")

        ttk.Label(box, text="Time Series dt [s] / images "
                            "(0 images = until Stop):").pack(anchor="w")
        ts_frame = ttk.Frame(box)
        ts_frame.pack(anchor="w")
        self._dt_var = tk.StringVar(value="1.0")
        self._max_frames_var = tk.StringVar(value="10")
        ttk.Entry(ts_frame, textvariable=self._dt_var, width=10).pack(
            side="left", padx=(0, 4))
        ttk.Entry(ts_frame, textvariable=self._max_frames_var, width=6).pack(
            side="left")

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(box, textvariable=self._status_var).pack(
            anchor="w", pady=(6, 0))

    def _on_mode_change(self) -> None:
        """Make the format selector reflect what the mode actually saves."""
        mode = self._mode_var.get()
        if mode == MODE_SNAPSHOT:
            self._format_combo.config(state="readonly")
            self._format_var.set(self._last_fmt)
        else:
            if self._format_combo.cget("state") == "readonly":
                self._last_fmt = self._format_var.get()
            self._format_combo.config(state="disabled")
            self._format_var.set("mp4" if mode == MODE_STREAMING
                                 else "tif + OME")

    def _build_canvas(self, parent) -> None:
        self._canvas = tk.Canvas(parent, width=CANVAS_SIZE[0],
                                 height=CANVAS_SIZE[1], bg="black")
        self._canvas.grid(row=0, column=0, sticky="nsew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        controls = ttk.Frame(parent)
        controls.grid(row=1, column=0, sticky="we", pady=4)
        ttk.Button(controls, text="Reset view",
                   command=self._reset_view).pack(side="left")
        ttk.Label(controls, text="Display:").pack(side="left", padx=(12, 2))
        self._display_mode_var = tk.StringVar(value="raw")
        ttk.Radiobutton(controls, text="Raw", value="raw",
                        variable=self._display_mode_var).pack(side="left")
        # Enabled once a perspective correction is defined (README).
        self._corrected_radio = ttk.Radiobutton(
            controls, text="Corrected (mm)", value="corrected",
            variable=self._display_mode_var, state="disabled")
        self._corrected_radio.pack(side="left", padx=4)

        info = ttk.Frame(parent)
        info.grid(row=2, column=0, sticky="we")
        ttk.Label(info, text="Profiles:").pack(side="left")
        self._profile_info_var = tk.StringVar(value="none")
        for value, label in (("none", "No information"),
                             ("stats", "mean/σ statistics"),
                             ("fit", "Gaussian + baseline fit")):
            ttk.Radiobutton(info, text=label, value=value,
                            variable=self._profile_info_var).pack(
                side="left", padx=4)

        ttk.Label(parent,
                  text="Horizontal profile (Y = 0 mm):").grid(
            row=3, column=0, sticky="w")
        self._h_profile_canvas = tk.Canvas(parent, height=90, bg="black",
                                           highlightthickness=0)
        self._h_profile_canvas.grid(row=4, column=0, sticky="we")
        ttk.Label(parent,
                  text="Vertical profile (X = 0 mm):").grid(
            row=5, column=0, sticky="w")
        self._v_profile_canvas = tk.Canvas(parent, height=90, bg="black",
                                           highlightthickness=0)
        self._v_profile_canvas.grid(row=6, column=0, sticky="we")
        self._clear_profiles()

        self._canvas.bind("<Button-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

    def _clear_profiles(self) -> None:
        for canvas in (self._h_profile_canvas, self._v_profile_canvas):
            canvas.delete("all")
            canvas.create_text(6, 4, text="—", anchor="nw", fill="#666666")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _apply_settings(self) -> None:
        if self._worker is not None:
            messagebox.showinfo("Busy", "Stop the acquisition before "
                                        "changing settings.")
            return
        try:
            config = self._read_config()
        except ConfigurationError as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        try:
            config.apply_to_camera(self._camera)
        except Exception as error:  # noqa: BLE001 - SDK errors
            messagebox.showerror("Camera error", str(error))
            return
        self._config = config
        self._processor = make_processor(config.color_space,
                                         config.wb_gains)
        self._status_var.set("Settings applied.")

    def _read_config(self) -> Configuration:
        """Build a validated Configuration from the current UI values."""
        return Configuration(
            auto_exposure=self._auto_exposure_var.get(),
            exposure_us=float(self._exposure_var.get()),
            fps=float(self._fps_var.get()),
            gain_db=float(self._gain_var.get()),
            black_level=int(float(self._black_var.get())),
            color_space=self._color_var.get(),
            wb_gains=tuple(float(var.get()) for var in self._wb_vars),
        )

    # ------------------------------------------------------------------
    # Acquisition control
    # ------------------------------------------------------------------

    def _start(self) -> None:
        if self._worker is not None:
            return
        try:
            config = self._read_config()
        except (ConfigurationError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        # Apply camera-resident settings for the run (manual exposure,
        # gain, black level; auto mode skips exposure).
        try:
            config.apply_to_camera(self._camera)
        except Exception as error:  # noqa: BLE001 - SDK errors
            messagebox.showerror("Camera error", str(error))
            return
        self._config = config
        self._processor = make_processor(config.color_space,
                                         config.wb_gains)

        mode = self._mode_var.get()
        save_enabled = self._save_var.get()
        save_dir = self._dir_var.get().strip() or "data"
        identifier = (self._identifier_var.get().strip()
                      .replace("/", "_").replace("\\", "_")) or "run"
        os.makedirs(save_dir, exist_ok=True)
        ts = acquisition_timestamp()
        self._live_ae = None

        try:
            target = self._build_target(mode, config, ts, identifier,
                                        save_dir, save_enabled)
        except (ConfigurationError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return

        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run_worker, args=(target,), daemon=True,
            name="acquisition")
        self._start_button.config(state="disabled")
        self._stop_button.config(state="normal")
        self._status_var.set(f"{mode} running...")
        self._worker.start()

    def _build_target(self, mode, config, ts, identifier, save_dir,
                      save_enabled) -> Callable[[], object]:
        camera = self._camera
        on_frame = self._on_frame
        if mode == MODE_SNAPSHOT:
            session = SnapshotSession(
                camera, ts, identifier, save_dir=save_dir,
                save_enabled=save_enabled,
                fmt=self._format_var.get(), processor=self._processor,
                on_frame=on_frame)
            return session.run
        if mode == MODE_STREAMING:
            fps = config.fps
            fps_cap = max_fps(camera.exposure_time_us, camera)
            if fps_cap is not None and fps > fps_cap:
                fps = fps_cap
            auto = None
            if config.auto_exposure:
                gain_lo, gain_hi = camera.gain_range
                auto = AutoExposureController(
                    camera.exposure_time_range_us,
                    (camera.convert_gain_to_decibels(gain_lo),
                     camera.convert_gain_to_decibels(gain_hi)))
            duration = float(self._duration_var.get() or 0.0)
            session = StreamingSession(
                camera, fps=fps, timestamp=ts, identifier=identifier,
                save_dir=save_dir, save_enabled=save_enabled,
                duration_s=duration or None, auto_exposure=auto,
                processor=self._processor, on_frame=self._make_on_frame(auto))
            return lambda: session.run(self._stop_event)
        # Time Series: manual exposure (README), frozen from the last
        # streaming auto-exposure run when available.
        exposure_us = config.exposure_us
        if self._last_auto is not None:
            exposure_us = self._last_auto[0]
        dt_min = minimum_interval_s(exposure_us, camera_fps_max(camera))
        interval = max(float(self._dt_var.get() or 1.0), dt_min)
        max_frames = int(float(self._max_frames_var.get() or 0))
        session = TimeSeriesSession(
            camera, interval_s=interval, timestamp=ts,
            identifier=identifier, save_dir=save_dir,
            save_enabled=save_enabled,
            max_frames=max_frames or None,
            processor=self._processor, on_frame=on_frame)
        return lambda: session.run(self._stop_event)

    def _run_worker(self, target: Callable[[], object]) -> None:
        try:
            self._events.put(("done", target()))
        except Exception as error:  # noqa: BLE001 - reported in the UI
            self._events.put(("error", error))

    def _stop(self) -> None:
        self._stop_event.set()

    def _browse(self) -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            self._dir_var.set(chosen)

    def _on_frame(self, bundle: FrameBundle, index: int = 0) -> None:
        with self._latest_lock:
            self._latest_bundle = bundle
        self._frame_counter = index

    def _make_on_frame(self, auto: Optional[AutoExposureController]
                       ) -> Callable[[FrameBundle, int], None]:
        """on_frame that also captures live auto-exposure state."""
        def on_frame(bundle: FrameBundle, index: int) -> None:
            self._on_frame(bundle, index)
            if auto is not None:
                self._live_ae = (auto.exposure_us, auto.gain_db,
                                 bundle.mean_luminance)
        return on_frame

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                self._finish_run()
                self._handle_worker_error(payload)
                return  # reconnect handling may have replaced everything
            if kind == "done":
                self._report_done(payload)
                self._finish_run()
        self._draw_latest()
        self._update_live_status()
        self._root.after(_POLL_MS, self._poll_events)

    def _update_live_status(self) -> None:
        """While running, show live progress (frames, auto-exposure state)."""
        if self._worker is None:
            return
        parts = [f"{self._mode_var.get()}: {self._frame_counter} frames"]
        if self._live_ae is not None:
            exposure_us, gain_db, luminance = self._live_ae
            parts.append(f"exposure {exposure_us:.1f} us, "
                         f"gain {gain_db:.1f} dB, "
                         f"luminance {luminance:.1f} / 1023")
        self._status_var.set("  |  ".join(parts) + " ...")

    def _finish_run(self) -> None:
        self._worker = None
        self._start_button.config(state="normal")
        self._stop_button.config(state="disabled")

    def _handle_worker_error(self, error: Exception) -> None:
        if isinstance(error, (KeyboardInterrupt, AcquisitionError)):
            text = str(error)
        else:
            text = f"{type(error).__name__}: {error}"
        self._status_var.set(f"Stopped: {text}")
        if self._service.connection_lost:
            try:
                self._camera = self._service.handle_connection_lost()
                self._status_var.set("Camera reconnected.")
            except Exception:  # noqa: BLE001 - UserAborted + SDK errors
                self._root.destroy()
                return

    def _report_done(self, result: object) -> None:
        if isinstance(result, StreamingStats):
            if result.exposure_us > 0:
                self._last_auto = (result.exposure_us, result.gain_db)
            self._status_var.set(
                f"Streaming finished: {result.frames} frames, "
                f"{result.dropped_mp4} dropped, exposure "
                f"{result.exposure_us:.1f} us, gain {result.gain_db:.1f} dB")
        elif isinstance(result, TimeSeriesStats):
            self._status_var.set(
                f"Time Series finished: {result.frames} frames "
                f"at dt = {result.interval_s:g} s")
        elif isinstance(result, FrameBundle):
            self._status_var.set(
                f"Snapshot finished (mean luminance "
                f"{result.mean_luminance:.1f} / 1023)")

    # ------------------------------------------------------------------
    # Display + ROI
    # ------------------------------------------------------------------

    def _draw_latest(self) -> None:
        with self._latest_lock:
            bundle = self._latest_bundle
            self._latest_bundle = None
        if bundle is None:
            return
        self._render(bundle)

    def _render(self, bundle: FrameBundle) -> None:
        self._last_bundle = bundle  # kept for re-renders on ROI changes
        if (self._display_mode_var.get() == "corrected"
                and self._perspective is not None):
            self._render_corrected(bundle)
        else:
            self._render_raw(bundle)

    def _render_raw(self, bundle: FrameBundle) -> None:
        rgb8 = display.bundle_to_rgb8(bundle)
        canvas_size = (max(self._canvas.winfo_width(), 1),
                       max(self._canvas.winfo_height(), 1))
        geom = display.compute_view((rgb8.shape[1], rgb8.shape[0]),
                                    canvas_size, self._roi)
        self._geom = geom
        crop = display.crop_for_view(rgb8, geom)
        target_size = (max(1, int(geom.crop_w * geom.scale)),
                       max(1, int(geom.crop_h * geom.scale)))
        image = Image.fromarray(crop).resize(target_size, Image.NEAREST)
        self._photo = ImageTk.PhotoImage(image)
        self._canvas.delete("all")
        self._canvas.create_image(geom.offset_x, geom.offset_y,
                                  image=self._photo, anchor="nw")
        if self._drawing and self._draw_rect is not None:
            self._canvas.coords(self._draw_rect, *self._live_rect)

    def _render_corrected(self, bundle: FrameBundle) -> None:
        transform = self._perspective
        assert transform is not None
        rgb8 = display.bundle_to_rgb8(bundle)
        corrected = transform.apply(rgb8)
        canvas_size = (max(self._canvas.winfo_width(), 1),
                       max(self._canvas.winfo_height(), 1))
        geom = display.compute_view((corrected.shape[1], corrected.shape[0]),
                                    canvas_size)
        self._geom = geom
        crop = display.crop_for_view(corrected, geom)
        target_size = (max(1, int(geom.crop_w * geom.scale)),
                       max(1, int(geom.crop_h * geom.scale)))
        self._photo = ImageTk.PhotoImage(
            Image.fromarray(crop).resize(target_size, Image.NEAREST))
        self._canvas.delete("all")
        self._canvas.create_image(geom.offset_x, geom.offset_y,
                                  image=self._photo, anchor="nw")
        # Calibrated axes (mm) — corner labels on the fitted image.
        x_min, x_max = transform.x_range
        y_min, y_max = transform.y_range
        self._canvas.create_text(
            6, 6, text=f"Y: {y_min:g} .. {y_max:g} mm",
            anchor="nw", fill="white")
        self._canvas.create_text(
            canvas_size[0] - 6, canvas_size[1] - 6,
            text=f"X: {x_min:g} .. {x_max:g} mm",
            anchor="se", fill="white")
        self._draw_profiles(bundle)

    def _draw_profiles(self, bundle: FrameBundle) -> None:
        transform = self._perspective
        assert transform is not None
        corrected_luma = transform.apply(bundle.luminance16)
        profiles = extract_profiles(corrected_luma, transform.x_range,
                                    transform.y_range)
        mode = self._profile_info_var.get()
        # README: the Gaussian fit is not available in Streaming (compute
        # budget) — fall back to statistics.
        effective = ("stats" if (mode == "fit"
                                 and self._mode_var.get() == MODE_STREAMING)
                     else mode)
        for canvas, label_text, coords, values in (
                (self._h_profile_canvas, "Horizontal (Y = 0 mm)",
                 profiles.x_mm, profiles.horizontal),
                (self._v_profile_canvas, "Vertical (X = 0 mm)",
                 profiles.y_mm, profiles.vertical)):
            overlay = None
            if effective == "stats":
                stats = profile_stats(coords, values)
                overlay = gaussian_overlay(
                    coords, baseline=float(values.min()),
                    amplitude=float(values.max() - values.min()),
                    center=stats["mean"], sigma=max(stats["sigma"], 1e-6))
                text = (f"{label_text}: center {stats['mean']:.2f} mm, "
                        f"width {stats['sigma']:.2f} mm")
            elif effective == "fit":
                fit = gaussian_fit(coords, values)
                overlay = gaussian_overlay(coords, **fit)
                text = (f"{label_text}: x0 {fit['center']:.2f} mm, "
                        f"FWHM {fit['sigma'] * 2.3548:.2f} mm, "
                        f"A {fit['amplitude']:.1f}, "
                        f"base {fit['baseline']:.1f}")
            else:
                text = label_text
            self._draw_profile_curve(canvas, coords, values, overlay, text)

    @staticmethod
    def _draw_profile_curve(canvas, coords, values, overlay, text) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 60)
        height = max(canvas.winfo_height(), 30)
        values = np.asarray(values, dtype=np.float64)
        if values.size < 2:
            return
        vmin, vmax = float(values.min()), float(values.max())
        span = (vmax - vmin) or 1.0
        c0, c1 = float(coords[0]), float(coords[-1])
        cspan = (c1 - c0) or 1.0

        def xy(coord: float, value: float):
            px = 4 + (coord - c0) / cspan * (width - 8)
            py = height - 4 - (value - vmin) / span * (height - 14)
            return px, py

        points = []
        for coord, value in zip(coords, values):
            points.extend(xy(float(coord), float(value)))
        canvas.create_line(*points, fill="#00ffff")
        if overlay is not None:
            points = []
            for coord, value in zip(coords, np.asarray(overlay)):
                points.extend(xy(float(coord), float(value)))
            canvas.create_line(*points, fill="#ffff00", dash=(4, 2))
        canvas.create_text(6, 4, text=text, anchor="nw", fill="white")

    def _reset_view(self) -> None:
        self._roi = None
        if getattr(self, "_last_bundle", None) is not None:
            self._render(self._last_bundle)

    # -- ROI selection (full view only) ---------------------------------

    def _on_press(self, event) -> None:
        if (self._roi is not None or self._geom is None
                or self._display_mode_var.get() != "raw"):
            return  # ROI zoom is a raw-mode feature (README)
        self._drawing = True
        self._draw_start = (event.x, event.y)
        self._draw_rect = self._canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="yellow")

    def _on_drag(self, event) -> None:
        if not self._drawing:
            return
        self._live_rect = (self._draw_start[0], self._draw_start[1],
                           event.x, event.y)
        self._canvas.coords(self._draw_rect, *self._live_rect)

    def _on_release(self, event) -> None:
        if not self._drawing:
            return
        self._drawing = False
        self._canvas.delete(self._draw_rect)
        self._draw_rect = None
        if self._geom is None:
            return
        x0, y0 = self._geom.canvas_to_image(self._draw_start[0],
                                            self._draw_start[1])
        x1, y1 = self._geom.canvas_to_image(event.x, event.y)
        image_size = (self._camera.image_width_pixels,
                      self._camera.image_height_pixels)
        self._roi = display.normalize_roi((x0, y0, x1, y1), image_size)
        if self._roi is not None and \
                getattr(self, "_last_bundle", None) is not None:
            self._render(self._last_bundle)  # zoom the on-screen frame now

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop any running acquisition and dispose camera + SDK."""
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=5.0)
        self._service.shutdown()


def main() -> int:
    """Connect to the camera (dialog flow) and run the main window."""
    import sys

    from ..camera import CameraServiceError, UserAbortedError

    service = CameraService()
    try:
        camera = service.connect()
    except UserAbortedError:
        print("No camera connected — exiting.")
        return 1
    except CameraServiceError as error:
        print(f"Fatal: {error}", file=sys.stderr)
        return 1

    root = tk.Tk()
    window = MainWindow(root, service, camera)

    def on_close() -> None:
        window.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    print("Camera and SDK disposed.")
    return 0
