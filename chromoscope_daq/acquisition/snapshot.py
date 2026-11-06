"""Snapshot mode: one-time acquisition of a single frame.

Raw image saved as jpg or tif (user-selected format). Variant saving
(corrected/calibrated) is layered on top by the UI via ``on_frame``.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from .base import AcquisitionError, FrameBundle, Processor, poll_frame_copy
from .naming import build_basename
from .processing import default_processor, save_jpg8, save_tif16
from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE

#: Minimum poll deadline (seconds) for the single frame.
_POLL_TIMEOUT_S = 5.0

ImageFormat = str  # "jpg" or "tif"


def save_snapshot(bundle: FrameBundle, path_no_ext: str,
                  fmt: ImageFormat) -> str:
    """Save the raw snapshot; returns the file path written."""
    ext = "jpg" if fmt.lower().startswith("jpg") else "tif"
    path = f"{path_no_ext}.{ext}"
    if ext == "jpg":
        save_jpg8(path, bundle.bgr8)
    else:
        save_tif16(path, bundle.raw16)
    return path


class SnapshotSession:
    """Acquire exactly one frame via software trigger."""

    def __init__(self, camera, timestamp: str, identifier: str,
                 save_dir: Optional[str] = None, save_enabled: bool = False,
                 fmt: ImageFormat = "tif",
                 processor: Processor = default_processor,
                 on_frame: Optional[Callable[[FrameBundle], None]] = None,
                 ) -> None:
        self._camera = camera
        self._timestamp = timestamp
        self._identifier = identifier
        self._save_dir = save_dir
        self._save_enabled = save_enabled
        self._fmt = fmt
        self._processor = processor
        self._on_frame = on_frame

    def run(self) -> FrameBundle:
        camera = self._camera
        camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        camera.frames_per_trigger_zero_for_unlimited = 1
        # Short SDK-level poll timeout; poll_frame_copy owns the deadline.
        camera.image_poll_timeout_ms = 1
        camera.arm(2)  # frames_to_buffer: ring-buffer depth (see SDK examples)
        try:
            camera.issue_software_trigger()
            raw16 = poll_frame_copy(camera, self._poll_timeout_s())
            if raw16 is None:
                raise AcquisitionError("No frame received from camera "
                                       "(snapshot timed out).")
        finally:
            if camera.is_armed:
                camera.disarm()

        bundle = self._processor(raw16)

        if self._save_enabled:
            path_no_ext = os.path.join(
                self._save_dir or os.getcwd(),
                build_basename(self._timestamp, self._identifier))
            save_snapshot(bundle, path_no_ext, self._fmt)

        if self._on_frame is not None:
            self._on_frame(bundle)
        return bundle

    def _poll_timeout_s(self) -> float:
        """Poll deadline that scales with the configured exposure."""
        try:
            exposure_us = float(self._camera.exposure_time_us)
        except Exception:  # noqa: BLE001 - defensive; use the default
            exposure_us = 0.0
        return max(_POLL_TIMEOUT_S, exposure_us * 2e-6)
