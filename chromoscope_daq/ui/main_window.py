"""Tkinter main window: live view, acquisition modes, settings, saving.

Layout: left column with acquisition controls, center column with the 2D
view (plus colorbar in flux units), right column with the beam-profile
canvases. Implements the README requirements: mode switching, save
controls (folder/identifier/format), camera settings (auto/manual
exposure, fps, gain, black level, color space, white balance), live
display with ROI zoom, configurable profile cuts with statistics/fit,
perspective-correction grid overlay, flux colorbar, correction
indicators, and the Streaming drop counter.

Acquisition runs on a worker thread; frames are handed over via the
``on_frame`` callbacks (latest-wins) and completion/errors via a queue
drained by the Tk event loop. All settings (including perspective
correction and intensity calibration) are restored between sessions and
can be saved/loaded as configuration files (File menu).
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
                           acquisition_timestamp, build_basename,
                           camera_fps_max, make_processor, minimum_interval_s)
from ..acquisition.processing import COLOR_SPACES
from ..calibration import IntensityCalibration
from ..camera import CameraService
from ..config import Configuration, ConfigurationError, max_fps
from ..perspective import PerspectiveError, PerspectiveTransform
from ..profiles import (extract_profiles, gaussian_fit, gaussian_overlay,
                        nice_ticks, profile_stats)
from .. import session_config
from ..variants import TimeSeriesVariantWriters, save_snapshot_variants
from . import display

#: Fixed display canvas size, in pixels.
CANVAS_SIZE = (720, 540)
#: Profile canvas geometry.
PROFILE_SIZE = (300, 130)
#: Colorbar width, in pixels.
_COLORBAR_WIDTH = 52
#: UI refresh period for the event/queue poll, in ms.
_POLL_MS = 33

MODE_SNAPSHOT = "Snapshot"
MODE_STREAMING = "Streaming"
MODE_TIME_SERIES = "Time Series"
MODES = (MODE_SNAPSHOT, MODE_STREAMING, MODE_TIME_SERIES)

#: Flux units (README: intensity calibration output).
FLUX_UNITS = "p/cm²/s"


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
        self._perspective_source: Optional[str] = None
        self._calibration: Optional[IntensityCalibration] = None
        self._calibration_source: Optional[str] = None
        self._build_ui()
        self._on_mode_change()  # initial state: Streaming selected
        # Restore the last session's settings (camera configuration,
        # corrections, cuts, acquisition parameters) automatically.
        self._load_session(session_config.default_path(), quiet=True)
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
        center = ttk.Frame(main)
        center.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(main)
        right.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        self._build_mode_frame(left)
        self._build_settings_frame(left)
        self._build_acquisition_frame(left)
        self._build_display(center)
        self._build_profile_column(right)

    def _build_menu(self) -> None:
        menu = tk.Menu(self._root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Load configuration...",
                              command=self._load_configuration)
        file_menu.add_command(label="Save configuration as...",
                              command=self._save_configuration_as)
        file_menu.add_command(label="Save as default",
                              command=self._save_default)
        menu.add_cascade(label="File", menu=file_menu)
        correction = tk.Menu(menu, tearoff=False)
        correction.add_command(label="Perspective correction...",
                               command=self._open_perspective)
        correction.add_command(label="Intensity calibration...",
                               command=self._open_calibration)
        menu.add_cascade(label="Correction", menu=correction)
        self._root.config(menu=menu)

    # -- left column ----------------------------------------------------

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
        for label, var in (
                ("Exposure [us]", self._exposure_var),
                ("Frame rate [fps]", self._fps_var),
                ("Gain [dB]", self._gain_var),
                ("Black level", self._black_var)):
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
        # Variant saving (README): corrected/calibrated copies, not
        # available in Streaming.
        self._save_corrected_var = tk.BooleanVar(value=False)
        self._save_corrected_check = ttk.Checkbutton(
            box, text="Also save corrected variant",
            variable=self._save_corrected_var, state="disabled")
        self._save_corrected_check.pack(anchor="w")
        self._save_calibrated_var = tk.BooleanVar(value=False)
        self._save_calibrated_check = ttk.Checkbutton(
            box, text="Also save calibrated variant (float tif)",
            variable=self._save_calibrated_var, state="disabled")
        self._save_calibrated_check.pack(anchor="w")

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
        ttk.Label(box, textvariable=self._status_var,
                  wraplength=220, justify="left").pack(
            anchor="w", pady=(6, 0), fill="x")

    # -- center column: 2D view + colorbar -------------------------------

    def _build_display(self, parent) -> None:
        controls = ttk.Frame(parent)
        controls.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 4))
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
        ttk.Label(controls, text="Units:").pack(side="left", padx=(12, 2))
        self._units_var = tk.StringVar(value="pixel")
        ttk.Radiobutton(controls, text="Pixel value", value="pixel",
                        variable=self._units_var).pack(side="left")
        # Enabled once an intensity calibration is defined (README: the
        # user switches between pixel value and particle flux).
        self._flux_radio = ttk.Radiobutton(
            controls, text="Particle flux", value="flux",
            variable=self._units_var, state="disabled")
        self._flux_radio.pack(side="left", padx=4)
        self._grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Axis grid",
                        variable=self._grid_var).pack(side="left",
                                                      padx=(12, 0))

        view = ttk.Frame(parent)
        view.grid(row=1, column=0, columnspan=2, sticky="nsew")
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(view, width=CANVAS_SIZE[0],
                                 height=CANVAS_SIZE[1], bg="black")
        self._canvas.grid(row=0, column=0, sticky="nsew")
        view.rowconfigure(0, weight=1)
        view.columnconfigure(0, weight=1)

        self._colorbar = tk.Canvas(view, width=_COLORBAR_WIDTH,
                                   height=CANVAS_SIZE[1], bg="black",
                                   highlightthickness=0)
        self._colorbar.grid(row=0, column=1, sticky="ns")
        self._colorbar_unit = ttk.Label(view, text=f"[{FLUX_UNITS}]")
        self._colorbar_unit.grid(row=1, column=1, sticky="w")

        self._canvas.bind("<Button-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

    # -- right column: corrections + profiles ----------------------------

    def _build_profile_column(self, parent) -> None:
        indicators = ttk.LabelFrame(parent, text="Corrections", padding=4)
        indicators.pack(fill="x", pady=(0, 6))
        self._perspective_status_var = tk.StringVar()
        self._calibration_status_var = tk.StringVar()
        ttk.Label(indicators, text="Perspective:").grid(
            row=0, column=0, sticky="w")
        ttk.Label(indicators, textvariable=self._perspective_status_var,
                  wraplength=180, justify="left").grid(
            row=0, column=1, sticky="w")
        ttk.Label(indicators, text="Calibration:").grid(
            row=1, column=0, sticky="w")
        ttk.Label(indicators, textvariable=self._calibration_status_var,
                  wraplength=180, justify="left").grid(
            row=1, column=1, sticky="w")

        box = ttk.LabelFrame(parent, text="Profiles", padding=4)
        box.pack(fill="x", pady=(0, 6))
        self._profile_info_var = tk.StringVar(value="none")
        for value, label in (("none", "No information"),
                             ("stats", "mean/σ statistics"),
                             ("fit", "Gaussian + baseline fit")):
            ttk.Radiobutton(box, text=label, value=value,
                            variable=self._profile_info_var).pack(
                anchor="w")

        cuts = ttk.Frame(box)
        cuts.pack(fill="x", pady=(4, 0))
        ttk.Label(cuts, text="Vertical cut X [mm]:").grid(
            row=0, column=0, sticky="w")
        self._cut_x_var = tk.StringVar(value="0.0")
        ttk.Entry(cuts, textvariable=self._cut_x_var, width=7).grid(
            row=0, column=1, sticky="w", padx=4)
        ttk.Label(cuts, text="Horizontal cut Y [mm]:").grid(
            row=1, column=0, sticky="w")
        self._cut_y_var = tk.StringVar(value="0.0")
        ttk.Entry(cuts, textvariable=self._cut_y_var, width=7).grid(
            row=1, column=1, sticky="w", padx=4)

        self._h_label_var = tk.StringVar()
        ttk.Label(parent, textvariable=self._h_label_var).pack(
            anchor="w", pady=(4, 0))
        self._h_profile_canvas = tk.Canvas(
            parent, width=PROFILE_SIZE[0], height=PROFILE_SIZE[1],
            bg="black", highlightthickness=0)
        self._h_profile_canvas.pack(fill="x")
        self._v_label_var = tk.StringVar()
        ttk.Label(parent, textvariable=self._v_label_var).pack(
            anchor="w", pady=(4, 0))
        self._v_profile_canvas = tk.Canvas(
            parent, width=PROFILE_SIZE[0], height=PROFILE_SIZE[1],
            bg="black", highlightthickness=0)
        self._v_profile_canvas.pack(fill="x")
        self._clear_profiles()

    def _clear_profiles(self) -> None:
        for canvas in (self._h_profile_canvas, self._v_profile_canvas):
            canvas.delete("all")
            canvas.create_text(6, 4, text="—", anchor="nw", fill="#666666")

    # ------------------------------------------------------------------
    # Corrections (perspective / calibration) + indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_correction(obj, source: Optional[str]) -> str:
        if obj is None:
            return "not loaded"
        if source:
            return f"loaded from {os.path.basename(source)}"
        return "defined (unsaved)"

    def _update_indicators(self) -> None:
        self._perspective_status_var.set(
            self._describe_correction(self._perspective,
                                      self._perspective_source))
        self._calibration_status_var.set(
            self._describe_correction(self._calibration,
                                      self._calibration_source))

    def _open_perspective(self) -> None:
        from .perspective_dialog import PerspectiveDialog
        PerspectiveDialog(
            self._root, self._perspective,
            on_apply=self._set_perspective,
            get_last_frame=lambda: getattr(self, "_last_bundle", None))

    def _open_calibration(self) -> None:
        from .calibration_dialog import CalibrationDialog
        CalibrationDialog(self._root, self._calibration,
                          on_apply=self._set_calibration)

    def _set_calibration(self, calibration: IntensityCalibration,
                         source: Optional[str] = None) -> None:
        self._calibration = calibration
        self._calibration_source = source
        # The flux unit toggle becomes available (README: switch on/off).
        self._flux_radio.config(state="normal")
        self._update_variant_states()
        self._update_indicators()
        self._save_session(session_config.default_path())
        self._status_var.set(
            f"Intensity calibration {self._describe_correction(calibration, source)}: "
            f"{calibration.kind} {calibration.coefficients}. "
            "Particle flux display and calibrated variant saving enabled.")

    def _set_perspective(self, transform: PerspectiveTransform,
                         source: Optional[str] = None) -> None:
        self._perspective = transform
        self._perspective_source = source
        # The corrected display mode is now available (README) — switch
        # to it so the user immediately sees the result.
        self._corrected_radio.config(state="normal")
        self._display_mode_var.set("corrected")
        self._update_variant_states()
        self._update_indicators()
        self._save_session(session_config.default_path())
        if getattr(self, "_last_bundle", None) is not None:
            self._render(self._last_bundle)
        self._status_var.set(
            "Perspective correction "
            f"{self._describe_correction(transform, source)}: "
            f"X {transform.x_range} mm, Y {transform.y_range} mm.")

    # ------------------------------------------------------------------
    # Settings persistence (session defaults + configuration files)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _session_state(self) -> dict:
        def entry(var, convert, default):
            try:
                return convert(var.get())
            except (TypeError, ValueError):
                return default

        return {
            "configuration": {
                "auto_exposure": bool(self._auto_exposure_var.get()),
                "exposure_us": entry(self._exposure_var, float,
                                     self._config.exposure_us),
                "fps": entry(self._fps_var, float, self._config.fps),
                "gain_db": entry(self._gain_var, float,
                                 self._config.gain_db),
                "black_level": entry(self._black_var,
                                     lambda v: int(float(v)),
                                     self._config.black_level),
                "color_space": self._color_var.get(),
                "wb_gains": [entry(var, float, g)
                             for var, g in zip(self._wb_vars,
                                               self._config.wb_gains)],
            },
            "perspective": {
                "definition": (self._perspective.to_dict()
                               if self._perspective is not None else None),
                "source": self._perspective_source,
            },
            "calibration": {
                "definition": (self._calibration.to_dict()
                               if self._calibration is not None else None),
                "source": self._calibration_source,
            },
            "display": {
                "cut_x_mm": entry(self._cut_x_var, float, 0.0),
                "cut_y_mm": entry(self._cut_y_var, float, 0.0),
                "units": self._units_var.get(),
                "display_mode": self._display_mode_var.get(),
                "axis_grid": bool(self._grid_var.get()),
            },
            "acquisition": {
                "save_dir": self._dir_var.get(),
                "identifier": self._identifier_var.get(),
                "duration_s": entry(self._duration_var, float, 0.0),
                "dt_s": entry(self._dt_var, float, 1.0),
                "max_frames": entry(self._max_frames_var,
                                    lambda v: int(float(v)), 10),
                "snapshot_format": self._format_var.get(),
            },
        }

    def _apply_session_state(self, state: dict) -> None:
        """Restore UI state from a settings dict (raises on bad data)."""
        config_data = state.get("configuration") or {}
        if config_data:
            wb = config_data.get("wb_gains")
            config = Configuration(
                auto_exposure=bool(config_data.get("auto_exposure", True)),
                exposure_us=self._to_float(config_data.get("exposure_us"),
                                           1000.0),
                fps=self._to_float(config_data.get("fps"), 30.0),
                gain_db=self._to_float(config_data.get("gain_db"), 0.0),
                black_level=int(self._to_float(config_data.get("black_level"),
                                               0)),
                color_space=config_data.get("color_space",
                                            self._config.color_space),
                wb_gains=(tuple(wb) if wb and len(wb) == 3
                          else self._config.wb_gains),
            )
            config.validate()
            self._config = config
            self._processor = make_processor(config.color_space,
                                             config.wb_gains)
            self._auto_exposure_var.set(config.auto_exposure)
            self._exposure_var.set(str(int(config.exposure_us)))
            self._fps_var.set(str(config.fps))
            self._gain_var.set(str(config.gain_db))
            self._black_var.set(str(config.black_level))
            self._color_var.set(config.color_space)
            for var, gain in zip(self._wb_vars, config.wb_gains):
                var.set(str(gain))

        perspective = state.get("perspective") or {}
        definition = perspective.get("definition")
        if definition:
            transform = PerspectiveTransform.from_dict(definition)
            transform.validate()
            self._perspective = transform
            self._perspective_source = perspective.get("source")
            self._corrected_radio.config(state="normal")
        else:
            self._perspective = None
            self._perspective_source = None
            self._corrected_radio.config(state="disabled")

        calibration_data = state.get("calibration") or {}
        definition = calibration_data.get("definition")
        if definition:
            calibration = IntensityCalibration.from_dict(definition)
            calibration.validate()
            self._calibration = calibration
            self._calibration_source = calibration_data.get("source")
            self._flux_radio.config(state="normal")
        else:
            self._calibration = None
            self._calibration_source = None
            self._flux_radio.config(state="disabled")
            self._units_var.set("pixel")

        display_state = state.get("display") or {}
        self._cut_x_var.set(str(self._to_float(display_state.get("cut_x_mm"),
                                               0.0)))
        self._cut_y_var.set(str(self._to_float(display_state.get("cut_y_mm"),
                                               0.0)))
        units = display_state.get("units", "pixel")
        self._units_var.set(units if (units == "pixel"
                                      or self._calibration is not None)
                            else "pixel")
        self._grid_var.set(bool(display_state.get("axis_grid", True)))
        mode = display_state.get("display_mode", "raw")
        self._display_mode_var.set(
            mode if (mode == "raw" or self._perspective is not None)
            else "raw")

        acquisition = state.get("acquisition") or {}
        self._dir_var.set(str(acquisition.get("save_dir", "data")))
        self._identifier_var.set(str(acquisition.get("identifier", "run")))
        self._duration_var.set(str(self._to_float(
            acquisition.get("duration_s"), 0.0)))
        self._dt_var.set(str(self._to_float(acquisition.get("dt_s"), 1.0)))
        self._max_frames_var.set(str(int(self._to_float(
            acquisition.get("max_frames"), 10))))
        fmt = acquisition.get("snapshot_format")
        if fmt in ("tif", "jpg"):
            self._format_var.set(fmt)
            self._last_fmt = fmt

        self._update_variant_states()
        self._update_indicators()

    def _save_session(self, path: str) -> None:
        try:
            session_config.save_settings(path, self._session_state())
        except OSError as error:
            self._status_var.set(f"Could not save settings: {error}")

    def _load_session(self, path: str, quiet: bool = False) -> bool:
        try:
            state = session_config.load_settings(path)
            self._apply_session_state(state)
        except (session_config.SettingsError, PerspectiveError,
                CalibrationError, ConfigurationError, ValueError) as error:
            if quiet:
                self._status_var.set(f"Settings not restored: {error}")
            else:
                messagebox.showerror("Settings", str(error))
            return False
        if not quiet:
            self._status_var.set(f"Configuration loaded from {path}.")
        return True

    def _save_configuration_as(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=(("Chromoscope DAQ configuration", "*.json"),))
        if not path:
            return
        self._save_session(path)
        self._status_var.set(f"Configuration saved to {path}.")

    def _load_configuration(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=(("Chromoscope DAQ configuration", "*.json"),))
        if not path:
            return
        self._load_session(path, quiet=False)

    def _save_default(self) -> None:
        self._save_session(session_config.default_path())
        self._status_var.set("Default settings saved.")

    # ------------------------------------------------------------------
    # Camera settings
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
        self._save_session(session_config.default_path())
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
            save_corrected = self._save_corrected_var.get()
            save_calibrated = self._save_calibrated_var.get()
            fmt = self._format_var.get()
            session = SnapshotSession(
                camera, ts, identifier, save_dir=save_dir,
                save_enabled=save_enabled,
                fmt=fmt, processor=self._processor,
                on_frame=on_frame)

            def run_snapshot():
                bundle = session.run()
                if save_enabled and (save_corrected or save_calibrated):
                    path_no_ext = os.path.join(
                        save_dir, build_basename(ts, identifier))
                    save_snapshot_variants(
                        bundle, self._perspective, self._calibration,
                        path_no_ext, fmt, save_corrected, save_calibrated)
                return bundle

            return run_snapshot
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
        # Variant writers (corrected/calibrated multi-frame tifs) are
        # created up front and fed from the frame callback.
        variant_writers = None
        if save_enabled:
            save_corrected = self._save_corrected_var.get()
            save_calibrated = self._save_calibrated_var.get()
            if (self._perspective is not None
                    and (save_corrected or save_calibrated)):
                variant_writers = TimeSeriesVariantWriters(
                    ts, identifier, save_dir, self._perspective,
                    self._calibration, save_corrected, save_calibrated)
        if variant_writers is not None:
            def ts_on_frame(bundle: FrameBundle, index: int) -> None:
                self._on_frame(bundle, index)
                variant_writers.write(bundle)
            frame_callback: Callable = ts_on_frame
        else:
            frame_callback = self._on_frame
        session = TimeSeriesSession(
            camera, interval_s=interval, timestamp=ts,
            identifier=identifier, save_dir=save_dir,
            save_enabled=save_enabled,
            max_frames=max_frames or None,
            processor=self._processor, on_frame=frame_callback)

        def run_ts():
            try:
                return session.run(self._stop_event)
            finally:
                if variant_writers is not None:
                    variant_writers.close()

        return run_ts

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

    def _show_image(self, shown: np.ndarray,
                    geom: display.ViewGeometry) -> None:
        crop = display.crop_for_view(shown, geom)
        target_size = (max(1, int(geom.crop_w * geom.scale)),
                       max(1, int(geom.crop_h * geom.scale)))
        self._photo = ImageTk.PhotoImage(
            Image.fromarray(crop).resize(target_size, Image.NEAREST))
        self._canvas.delete("all")
        self._canvas.create_image(geom.offset_x, geom.offset_y,
                                  image=self._photo, anchor="nw")

    def _render_raw(self, bundle: FrameBundle) -> None:
        rgb8 = display.bundle_to_rgb8(bundle)
        canvas_size = (max(self._canvas.winfo_width(), 1),
                       max(self._canvas.winfo_height(), 1))
        geom = display.compute_view((rgb8.shape[1], rgb8.shape[0]),
                                    canvas_size, self._roi)
        self._geom = geom
        self._show_image(rgb8, geom)
        self._clear_colorbar()
        if self._drawing and self._draw_rect is not None:
            self._canvas.coords(self._draw_rect, *self._live_rect)

    def _render_corrected(self, bundle: FrameBundle) -> None:
        transform = self._perspective
        assert transform is not None
        flux_mode = (self._units_var.get() == "flux"
                     and self._calibration is not None)
        corrected_luma = transform.apply(bundle.luminance16)
        if flux_mode:
            # Particle flux: grayscale display, normalized to the peak.
            scalar = self._calibration.apply(corrected_luma)
            peak = float(scalar.max())
            shown = (scalar / (peak if peak > 0 else 1.0) * 255.0
                     ).clip(0, 255).astype(np.uint8)
            self._draw_colorbar(peak)
        else:
            shown = transform.apply(display.bundle_to_rgb8(bundle))
            self._clear_colorbar()
        canvas_size = (max(self._canvas.winfo_width(), 1),
                       max(self._canvas.winfo_height(), 1))
        geom = display.compute_view((shown.shape[1], shown.shape[0]),
                                    canvas_size)
        self._geom = geom
        self._show_image(shown, geom)
        # Axis grid + corner range labels.
        if self._grid_var.get():
            self._draw_grid(shown.shape[1], shown.shape[0], geom, transform)
        x_min, x_max = transform.x_range
        y_min, y_max = transform.y_range
        self._canvas.create_text(
            6, 6, text=f"Y: {y_min:g} .. {y_max:g} mm",
            anchor="nw", fill="white")
        self._canvas.create_text(
            canvas_size[0] - 6, canvas_size[1] - 6,
            text=f"X: {x_min:g} .. {x_max:g} mm",
            anchor="se", fill="white")
        if flux_mode:
            self._canvas.create_text(
                canvas_size[0] - 6, 6, text="particle flux",
                anchor="ne", fill="#ffff00")
        self._draw_profiles(corrected_luma if not flux_mode else scalar,
                            "flux" if flux_mode else "luminance")

    def _draw_grid(self, width: int, height: int, geom, transform) -> None:
        """Dashed mm grid over the corrected view, with tick labels."""
        small = ("TkDefaultFont", 7)
        canvas_w = max(self._canvas.winfo_width(), 1)
        canvas_h = max(self._canvas.winfo_height(), 1)
        for x_mm in nice_ticks(*transform.x_range):
            px, _ = transform.mm_to_output((x_mm, 0.0), transform.x_range,
                                           transform.y_range,
                                           (width, height))
            cx1, cy1 = geom.image_to_canvas(px, 0)
            cx2, cy2 = geom.image_to_canvas(px, height)
            self._canvas.create_line(cx1, cy1, cx2, cy2,
                                     fill="#b0b0b0", dash=(2, 5))
            self._canvas.create_text(cx1, canvas_h - 2,
                                     text=f"{x_mm:g}", anchor="s",
                                     fill="#dddddd", font=small)
        for y_mm in nice_ticks(*transform.y_range):
            _, py = transform.mm_to_output((0.0, y_mm), transform.x_range,
                                           transform.y_range,
                                           (width, height))
            cx1, cy1 = geom.image_to_canvas(0, py)
            cx2, cy2 = geom.image_to_canvas(width, py)
            self._canvas.create_line(cx1, cy1, cx2, cy2,
                                     fill="#b0b0b0", dash=(2, 5))
            self._canvas.create_text(2, cy1, text=f"{y_mm:g}", anchor="w",
                                     fill="#dddddd", font=small)

    def _draw_colorbar(self, peak: float) -> None:
        bar = self._colorbar
        bar.delete("all")
        height = max(bar.winfo_height(), CANVAS_SIZE[1])
        width = max(bar.winfo_width(), _COLORBAR_WIDTH)
        strips = 64
        for index in range(strips):
            y0 = height - (index + 1) * height / strips
            y1 = height - index * height / strips
            shade = int(255 * (index + 0.5) / strips)
            bar.create_rectangle(4, y0, width - 16, y1, outline="",
                                 fill=f"#{shade:02x}{shade:02x}{shade:02x}")
        small = ("TkDefaultFont", 7)
        bar.create_text(width - 12, 6, text=f"{peak:.3g}", anchor="ne",
                        fill="white", font=small)
        bar.create_text(width - 12, height / 2, text=f"{peak / 2:.3g}",
                        anchor="e", fill="white", font=small)
        bar.create_text(width - 12, height - 6, text="0", anchor="se",
                        fill="white", font=small)

    def _clear_colorbar(self) -> None:
        self._colorbar.delete("all")

    def _draw_profiles(self, corrected_scalar_arr, unit: str) -> None:
        transform = self._perspective
        assert transform is not None
        cut_x = self._to_float(self._cut_x_var.get(), 0.0)
        cut_y = self._to_float(self._cut_y_var.get(), 0.0)
        profiles = extract_profiles(corrected_scalar_arr, transform.x_range,
                                    transform.y_range,
                                    at_x_mm=cut_x, at_y_mm=cut_y)
        self._h_label_var.set(f"Horizontal profile (Y = {cut_y:g} mm):")
        self._v_label_var.set(f"Vertical profile (X = {cut_x:g} mm):")
        mode = self._profile_info_var.get()
        # README: the Gaussian fit is not available in Streaming (compute
        # budget) — fall back to statistics.
        effective = ("stats" if (mode == "fit"
                                 and self._mode_var.get() == MODE_STREAMING)
                     else mode)
        for canvas, label_text, coords, values in (
                (self._h_profile_canvas, "Horizontal",
                 profiles.x_mm, profiles.horizontal),
                (self._v_profile_canvas, "Vertical",
                 profiles.y_mm, profiles.vertical)):
            overlay = None
            if effective == "stats":
                stats = profile_stats(coords, values)
                overlay = gaussian_overlay(
                    coords, baseline=float(values.min()),
                    amplitude=float(values.max() - values.min()),
                    center=stats["mean"], sigma=max(stats["sigma"], 1e-6))
                text = (f"{label_text}: center {stats['mean']:.2f} mm, "
                        f"width {stats['sigma']:.2f} mm [{unit}]")
            elif effective == "fit":
                fit = gaussian_fit(coords, values)
                overlay = gaussian_overlay(coords, **fit)
                text = (f"{label_text}: x0 {fit['center']:.2f} mm, "
                        f"FWHM {fit['sigma'] * 2.3548:.2f} mm, "
                        f"A {fit['amplitude']:.1f}, "
                        f"base {fit['baseline']:.1f} [{unit}]")
            else:
                text = f"{label_text} [{unit}]"
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
        small = ("TkDefaultFont", 7)

        def xy(coord: float, value: float):
            px = 4 + (coord - c0) / cspan * (width - 8)
            py = height - 16 - (value - vmin) / span * (height - 26)
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
        # Axis ticks: mm values along the bottom, value scale on the right.
        for tick in nice_ticks(c0, c1, target=4):
            px, _ = xy(tick, vmin)
            canvas.create_text(px, height - 2, text=f"{tick:g}", anchor="s",
                               fill="#aaaaaa", font=small)
        canvas.create_text(width - 4, 14, text=f"{vmax:.3g}", anchor="ne",
                           fill="#aaaaaa", font=small)
        canvas.create_text(width - 4, height - 16, text=f"{vmin:.3g}",
                           anchor="se", fill="#aaaaaa", font=small)

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
        """Stop any running acquisition, save defaults, dispose camera."""
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=5.0)
        self._save_session(session_config.default_path())
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
