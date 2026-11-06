"""Time Series mode: software-triggered frames at a user-defined Δt.

Exposure is manual (auto-exposure is not allowed here, README). The UI
coerces the user Δt up to ``minimum_interval_s``. Runs until ``stop_event``
is set or ``max_frames`` frames are captured. Every frame is appended to
both a classic multi-frame TIFF and an OME-TIFF.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from .base import AcquisitionError, FrameBundle, Processor, poll_frame_copy
from .naming import build_filename
from .processing import default_processor
from .writers import TiffStackWriter
from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE

#: Poll deadline per frame (seconds) before treating the camera as stalled.
_POLL_TIMEOUT_S = 5.0

#: Readout margin (seconds) added on top of exposure + frame time.
_READOUT_MARGIN_S = 0.010

#: Fallback minimum interval when the camera max fps is unknown.
_FALLBACK_MIN_INTERVAL_S = 0.100


def minimum_interval_s(exposure_us: float, fps_max: Optional[float]) -> float:
    """Smallest Δt that still lets the sensor be re-exposed each cycle.

    Δt_min = exposure + frame time (1000 / fps_max) + 10 ms readout margin.
    Falls back to 100 ms when the max fps is unknown.
    """
    exposure_s = max(float(exposure_us), 0.0) * 1e-6
    if fps_max is None or fps_max <= 0:
        return max(_FALLBACK_MIN_INTERVAL_S, exposure_s + _READOUT_MARGIN_S)
    return exposure_s + 1.0 / float(fps_max) + _READOUT_MARGIN_S


def camera_fps_max(camera) -> Optional[float]:
    """Upper bound of the camera's frame-rate control range, or None."""
    try:
        lo, hi = camera.frame_rate_control_value_range
        return float(hi) if hi and hi > 0 else None
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass
class TimeSeriesStats:
    frames: int = 0
    interval_s: float = 0.0


class TimeSeriesSession:
    def __init__(self, camera, interval_s: float, timestamp: str,
                 identifier: str,
                 save_dir: Optional[str] = None, save_enabled: bool = False,
                 max_frames: Optional[int] = None,
                 processor: Processor = default_processor,
                 on_frame: Optional[Callable[[FrameBundle, int], None]] = None,
                 ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self._camera = camera
        self._interval_s = float(interval_s)
        self._timestamp = timestamp
        self._identifier = identifier
        self._save_dir = save_dir
        self._save_enabled = save_enabled
        self._max_frames = max_frames
        self._processor = processor
        self._on_frame = on_frame

    def _arm_single_trigger(self) -> None:
        camera = self._camera
        camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        camera.frames_per_trigger_zero_for_unlimited = 1
        # Short SDK-level poll timeout; poll_frame_copy owns the deadline.
        camera.image_poll_timeout_ms = 1
        camera.arm(2)  # frames_to_buffer: ring-buffer depth (see SDK examples)

    def _paths(self) -> tuple:
        base = os.path.join(self._save_dir or os.getcwd(), "")
        classic = os.path.join(
            base, build_filename(self._timestamp, self._identifier, "tif"))
        ome = os.path.join(
            base, build_filename(self._timestamp, self._identifier + "_OME",
                                 "tif"))
        return classic, ome

    def run(self, stop_event: threading.Event) -> TimeSeriesStats:
        if self._max_frames is not None and self._max_frames <= 0:
            raise ValueError("max_frames must be > 0 when given")

        stats = TimeSeriesStats(interval_s=self._interval_s)
        classic_writer: Optional[TiffStackWriter] = None
        ome_writer: Optional[TiffStackWriter] = None
        if self._save_enabled:
            classic_path, ome_path = self._paths()
            classic_writer = TiffStackWriter(classic_path, ome=False)
            ome_writer = TiffStackWriter(ome_path, ome=True)

        self._arm_single_trigger()
        try:
            while not stop_event.is_set():
                started = _monotonic()
                self._camera.issue_software_trigger()
                raw16 = poll_frame_copy(
                    self._camera, self._interval_s + _POLL_TIMEOUT_S)
                if raw16 is None:
                    raise AcquisitionError(
                        "No frame received from camera (time series timed "
                        "out after trigger).")
                bundle = self._processor(raw16)
                stats.frames += 1

                if classic_writer is not None and ome_writer is not None:
                    _append_frame(classic_writer, ome_writer, bundle.raw16)

                if self._on_frame is not None:
                    self._on_frame(bundle, stats.frames)

                if (self._max_frames is not None
                        and stats.frames >= self._max_frames):
                    break

                # Sleep out the remainder of Δt (interruptible by stop).
                elapsed = _monotonic() - started
                remaining = self._interval_s - elapsed
                if remaining > 0:
                    stop_event.wait(remaining)
        finally:
            if self._camera.is_armed:
                self._camera.disarm()
            if classic_writer is not None:
                classic_writer.close()
            if ome_writer is not None:
                ome_writer.close()
        return stats


def _append_frame(classic_writer: TiffStackWriter,
                  ome_writer: TiffStackWriter, raw16) -> None:
    classic_writer.write(raw16)
    ome_writer.write(raw16)


def _monotonic() -> float:
    import time
    return time.monotonic()
