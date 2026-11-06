"""Pure, testable auto-exposure controller per README.

Every N frames, scale the exposure time by L_target / L_mean (BT.709
luminance, target ~50 % of full scale), clamped to the camera's exposure
range. Gain ([0, 48] dB) is a secondary knob applied only when the exposure
is already clamped. Writes to the camera are throttled to at most one
exposure change per 300 ms (SDK scientific-CCD note).
"""

from __future__ import annotations

from math import log10
from typing import Optional, Tuple

#: Full scale of the 10-bit luminance.
FULL_SCALE = 1023.0
#: README target mean luminance: ~50 % of full scale.
TARGET_FRACTION = 0.5

#: Minimum interval between exposure writes, in seconds.
_MIN_WRITE_INTERVAL_S = 0.300
#: Gain step granularity (dB) when adjusting gain.
_GAIN_STEP_DB = 1.0


class AutoExposureController:
    """Stateful controller; feed it frame statistics, it returns settings."""

    def __init__(self, exposure_time_range_us: Tuple[float, float],
                 gain_range: Tuple[float, float],
                 target_luminance: float = TARGET_FRACTION * FULL_SCALE,
                 every_n_frames: int = 10,
                 tolerance: float = 0.05,
                 min_write_interval_s: float = _MIN_WRITE_INTERVAL_S,
                 max_step_factor: float = 8.0) -> None:
        if every_n_frames < 1:
            raise ValueError("every_n_frames must be >= 1")
        if max_step_factor <= 1.0:
            raise ValueError("max_step_factor must be > 1")
        self._exposure_range = exposure_time_range_us
        self._gain_range = gain_range
        self._target = float(target_luminance)
        self._every_n = every_n_frames
        self._tolerance = float(tolerance)
        self._min_interval = min_write_interval_s
        # Per-update cap on the exposure change factor. Without it, a
        # near-black frame (ratio ~500x) would jump the exposure to the
        # range maximum in one step and starve the camera for tens of
        # seconds; capping converges over a few gated frames instead.
        self._max_step = float(max_step_factor)
        # Current commanded settings.
        self.exposure_us = float(exposure_time_range_us[0])
        self.gain_db = float(gain_range[0])
        self._last_write_time: Optional[float] = None  # monotonic seconds
        self._frame_index = 0

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _now() -> float:
        import time
        return time.monotonic()

    def _gain_headroom(self, ratio: float) -> float:
        """Gain change (dB) equivalent to ``ratio``, limited to the range."""
        if ratio <= 0.0:
            return 0.0
        wanted_db = 20.0 * log10(ratio)
        low, high = self._gain_range
        target = min(max(self.gain_db + wanted_db, low), high)
        # Quantize to _GAIN_STEP_DB within range.
        lo_q = _ceil_step(low)
        hi_q = _floor_step(high)
        return min(max(_round_step(target), lo_q), hi_q) - self.gain_db

    # -- main interface --------------------------------------------------

    def should_update(self, frame_index: int) -> bool:
        """True when ``frame_index`` is an N-th frame (update checkpoint)."""
        return frame_index % self._every_n == 0

    def update(self, mean_luminance: float) -> bool:
        """Feed one mean luminance; adjust exposure/gain if needed.

        Returns True when new settings were produced (caller should write
        them to the camera, respecting the 300 ms throttle).
        """
        if mean_luminance <= 0.0:
            return False
        ratio = self._target / mean_luminance
        lo = self._tolerance
        if 1.0 - lo <= ratio <= 1.0 + lo:
            return False
        # Cap the per-update change factor.
        effective = min(max(ratio, 1.0 / self._max_step), self._max_step)

        new_exposure = self.exposure_us * effective
        exp_lo, exp_hi = self._exposure_range
        clamped = min(max(new_exposure, exp_lo), exp_hi)
        exposure_changed = abs(clamped - self.exposure_us) > 1e-9
        self.exposure_us = clamped

        gain_changed = False
        if abs(new_exposure - clamped) > 1e-9 and ratio > 0.0:
            # Exposure is clamped: use gain as the secondary knob, moving it
            # in the same direction as the requested ratio.
            delta_db = self._gain_headroom(effective)
            if abs(delta_db) >= _GAIN_STEP_DB / 2.0:
                self.gain_db = self.gain_db + delta_db
                gain_changed = True

        return exposure_changed or gain_changed

    def can_write_now(self, now: Optional[float] = None) -> bool:
        """Respect the SDK's >=300 ms spacing between exposure writes."""
        if self._last_write_time is None:
            return True
        t = self._now() if now is None else now
        return (t - self._last_write_time) >= self._min_interval

    def mark_written(self, now: Optional[float] = None) -> None:
        self._last_write_time = self._now() if now is None else now


def _floor_step(value: float) -> float:
    return float(int(value / _GAIN_STEP_DB)) * _GAIN_STEP_DB


def _ceil_step(value: float) -> float:
    return float(-int(-value / _GAIN_STEP_DB)) * _GAIN_STEP_DB


def _round_step(value: float) -> float:
    return float(round(value / _GAIN_STEP_DB)) * _GAIN_STEP_DB
