"""Common types and helpers shared by the acquisition modes."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np


class AcquisitionError(RuntimeError):
    """Raised when an acquisition cannot proceed (camera errors, bad state)."""


class FrameBundle:
    """Processed view of one raw camera frame.

    ``raw16`` is the original 16-bit buffer from the camera (10-bit data,
    right-aligned). ``rgb16``/``bgr8`` are demosaiced color versions and
    ``luminance16`` the BT.709 luminance used for auto-exposure and
    statistics. ``mean_luminance`` is the scalar mean of ``luminance16``.
    """

    __slots__ = ("raw16", "rgb16", "luminance16", "bgr8", "mean_luminance")

    def __init__(self, raw16: np.ndarray, rgb16: np.ndarray,
                 luminance16: np.ndarray, bgr8: np.ndarray,
                 mean_luminance: float) -> None:
        self.raw16 = raw16
        self.rgb16 = rgb16
        self.luminance16 = luminance16
        self.bgr8 = bgr8
        self.mean_luminance = mean_luminance


# Injectable processor signature: raw16 -> FrameBundle
Processor = Callable[[np.ndarray], FrameBundle]

#: Wait between consecutive polls when no frame is pending, in seconds.
_POLL_SLEEP_S = 0.005


def poll_frame_copy(camera: Any, timeout_s: float) -> Optional[np.ndarray]:
    """Poll for one pending frame, waiting up to ``timeout_s`` seconds.

    The camera's ``image_poll_timeout_ms`` must be short (the sessions set
    it to ~1 ms) so each individual poll returns quickly; this helper loops
    until a frame arrives or the deadline expires. The buffer is always
    copied: the SDK only guarantees ``image_buffer`` is valid until the
    next poll. Returns ``None`` when no frame arrived within ``timeout_s``.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        frame = camera.get_pending_frame_or_null()
        if frame is not None:
            return np.copy(frame.image_buffer)
        if time.monotonic() >= deadline:
            return None
        time.sleep(_POLL_SLEEP_S)


def set_gain_db(camera: Any, gain_db: float) -> None:
    """Write a gain value given in dB to the camera.

    The SDK ``TLCamera.gain`` property is expressed in camera-specific gain
    index units (for the CS165CU1/M: 0..480, not dB), so the value is
    converted through ``convert_decibels_to_gain`` when the camera provides
    it. Duck-typed test doubles without the converter receive the raw dB
    value.
    """
    convert = getattr(camera, "convert_decibels_to_gain", None)
    camera.gain = convert(gain_db) if convert is not None else int(round(gain_db))


def monotonic() -> float:
    """Monotonic clock in seconds, injectable for tests."""
    return time.monotonic()
