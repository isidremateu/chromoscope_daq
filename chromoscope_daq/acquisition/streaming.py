"""Streaming mode: free-run acquisition at a configured fps.

Every frame is displayed and (if saving) written to an mp4 by a dedicated
writer thread. Frames are dropped (and counted) when disk I/O is slower
than the configured fps; saving is prioritized over display. Runs until
``stop_event`` is set or ``duration_s`` elapses.

Auto-exposure is enabled by default in Streaming (README) and its exposure
writes are throttled to >= 300 ms apart.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from .autoexposure import AutoExposureController
from .base import (AcquisitionError, FrameBundle, Processor, poll_frame_copy,
                   set_gain_db)
from .naming import build_filename
from .processing import default_processor, to_bgr8
from .writers import Mp4Writer
from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE

#: Minimum stall grace (seconds): max silence since the last frame before
#: treating the camera as stalled.
_POLL_TIMEOUT_S = 5.0

#: Duration of one blocking poll (seconds). Short, so duration/stop checks
#: stay responsive; frames pending are returned immediately regardless.
_POLL_TICK_S = 0.05


@dataclass
class StreamingStats:
    frames: int = 0
    dropped_mp4: int = 0
    exposure_us: float = 0.0
    gain_db: float = 0.0


class StreamingSession:
    def __init__(self, camera, fps: float, timestamp: str, identifier: str,
                 save_dir: Optional[str] = None, save_enabled: bool = False,
                 duration_s: Optional[float] = None,
                 auto_exposure: Optional[AutoExposureController] = None,
                 processor: Processor = default_processor,
                 on_frame: Optional[Callable[[FrameBundle, int], None]] = None,
                 ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._camera = camera
        self._fps = float(fps)
        self._timestamp = timestamp
        self._identifier = identifier
        self._save_dir = save_dir
        self._save_enabled = save_enabled
        self._duration_s = duration_s
        self._auto = auto_exposure
        self._processor = processor
        self._on_frame = on_frame

    # -- camera helpers --------------------------------------------------

    def _arm_continuous(self) -> None:
        camera = self._camera
        camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        camera.frames_per_trigger_zero_for_unlimited = 0
        camera.image_poll_timeout_ms = 1
        if self._auto is not None:
            # Streaming owns the exposure when auto-exposure is on: start
            # from the controller's setting so a stale/persisted camera
            # exposure (possibly seconds long) cannot stall the free-run.
            camera.exposure_time_us = int(round(self._auto.exposure_us))
            self._auto.mark_written()
        camera.arm(2)  # frames_to_buffer: ring-buffer depth (see SDK examples)
        camera.issue_software_trigger()  # continuous self-triggering

    def _apply_auto_exposure(self, bundle: FrameBundle, frame_index: int,
                             stats: StreamingStats) -> None:
        if self._auto is None:
            return
        if not self._auto.should_update(frame_index):
            return
        if not self._auto.can_write_now():
            return
        if self._auto.update(bundle.mean_luminance):
            # SDK setters require integers (c_longlong / c_int).
            self._camera.exposure_time_us = int(round(self._auto.exposure_us))
            set_gain_db(self._camera, self._auto.gain_db)
            self._auto.mark_written()
        stats.exposure_us = self._auto.exposure_us
        stats.gain_db = self._auto.gain_db

    # -- main loop ---------------------------------------------------------

    def run(self, stop_event: threading.Event) -> StreamingStats:
        stats = StreamingStats()
        writer: Optional[Mp4Writer] = None
        self._arm_continuous()
        try:
            if self._save_enabled:
                path = self._mp4_path()
                # size: (width, height) of the 8-bit frames.
                writer = Mp4Writer(path, self._fps, self._frame_size())
            frame_interval = 1.0 / self._fps
            started = _monotonic()
            last_frame = started
            while not stop_event.is_set():
                loop_started = _monotonic()
                if (self._duration_s is not None
                        and loop_started - started >= self._duration_s):
                    break
                raw16 = poll_frame_copy(self._camera, _POLL_TICK_S)
                if raw16 is None:
                    if _monotonic() - last_frame >= self._stall_grace_s():
                        raise AcquisitionError(
                            "No frames from camera during streaming.")
                    continue
                last_frame = _monotonic()
                bundle = self._processor(raw16)
                stats.frames += 1
                if writer is not None:
                    writer.write(bundle.bgr8)
                # Saving already queued; display is the lower priority task
                # and runs after the enqueue.
                self._apply_auto_exposure(bundle, stats.frames, stats)
                if self._on_frame is not None:
                    self._on_frame(bundle, stats.frames)
                # Pace to the configured fps.
                elapsed = _monotonic() - loop_started
                remaining = frame_interval - elapsed
                if remaining > 0:
                    stop_event.wait(remaining)
            if writer is not None:
                stats.dropped_mp4 = writer.dropped
        finally:
            if self._camera.is_armed:
                self._camera.disarm()
            if writer is not None:
                writer.close()
        return stats

    # -- helpers ------------------------------------------------------------

    def _stall_grace_s(self) -> float:
        """Max silence (seconds) before declaring the camera stalled.

        Scales with the configured exposure: a camera exposing for N
        seconds cannot deliver a frame sooner, so the deadline grows with
        the exposure; the fixed ``_POLL_TIMEOUT_S`` floor still applies
        for typical short exposures.
        """
        if self._auto is not None:
            exposure_us = self._auto.exposure_us
        else:
            try:
                exposure_us = float(self._camera.exposure_time_us)
            except Exception:  # noqa: BLE001 - defensive
                exposure_us = 0.0
        return max(_POLL_TIMEOUT_S, exposure_us * 2e-6)

    def _frame_size(self) -> tuple:
        return (self._camera.image_width_pixels,
                self._camera.image_height_pixels)

    def _mp4_path(self) -> str:
        import os
        return os.path.join(
            self._save_dir or os.getcwd(),
            build_filename(self._timestamp, self._identifier, "mp4"))


def _monotonic() -> float:
    import time
    return time.monotonic()
