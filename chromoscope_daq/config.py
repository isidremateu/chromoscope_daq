"""Application configuration (README 'Configuration settings').

Pure data + validation + camera application — no GUI or SDK dependencies
beyond the acquisition helpers, so it is testable without hardware.

Camera knobs written to the hardware: gain (dB -> SDK index units), black
level, and exposure (manual mode only — in auto mode Streaming owns the
exposure). Color space and white balance gains are *processing* options
(the SDK always delivers raw Bayer); they select the frame processor via
:func:`chromoscope_daq.acquisition.make_processor`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from .acquisition import camera_fps_max, minimum_interval_s
from .acquisition.base import set_gain_db
from .acquisition.processing import COLOR_SPACES, UNPROCESSED, WB_DEFAULTS

#: Allowed gain range in dB (README).
GAIN_RANGE_DB = (0.0, 48.0)
#: Allowed black level range (README).
BLACK_LEVEL_RANGE = (0, 511)
#: Allowed per-channel white balance gain range (README).
WB_GAIN_RANGE = (0.0, 10.0)

__all__ = [
    "Configuration",
    "ConfigurationError",
    "GAIN_RANGE_DB",
    "BLACK_LEVEL_RANGE",
    "WB_GAIN_RANGE",
    "WB_DEFAULTS",
    "COLOR_SPACES",
    "max_fps",
]


class ConfigurationError(ValueError):
    """A configuration value is outside its allowed range."""


def _check_range(name: str, value, lo, hi) -> None:
    if not lo <= value <= hi:
        raise ConfigurationError(
            f"{name} = {value} outside allowed range [{lo}, {hi}]")


def max_fps(exposure_us: float, camera) -> float:
    """Highest streaming frame rate consistent with the exposure time.

    README: "the maximum frame rate is determined by the exposure time."
    A frame needs exposure + readout (derived from the camera's maximum
    frame rate, as in ``minimum_interval_s``), so the ceiling is the
    reciprocal of that minimum interval.
    """
    return 1.0 / minimum_interval_s(exposure_us, camera_fps_max(camera))


@dataclass
class Configuration:
    """User-facing acquisition settings."""

    #: Auto-exposure feedback loop (default in Streaming, README).
    auto_exposure: bool = True
    #: Fixed exposure in manual mode (required for Time Series).
    exposure_us: float = 1_000.0
    #: Streaming frame rate; coerce down to ``max_fps`` at run time.
    fps: float = 30.0
    gain_db: float = 0.0
    black_level: int = 0
    color_space: str = UNPROCESSED
    #: White balance gains (R, G, B) for the sRGB color spaces.
    wb_gains: Tuple[float, float, float] = WB_DEFAULTS

    def validate(self) -> None:
        """Raise :class:`ConfigurationError` on any out-of-range value."""
        _check_range("gain_db", self.gain_db, *GAIN_RANGE_DB)
        _check_range("black_level", self.black_level, *BLACK_LEVEL_RANGE)
        if self.color_space not in COLOR_SPACES:
            raise ConfigurationError(
                f"unknown color space {self.color_space!r}; "
                f"expected one of {COLOR_SPACES}")
        for index, gain in enumerate(self.wb_gains):
            _check_range(f"white balance gain {index}", gain,
                         *WB_GAIN_RANGE)
        if self.fps <= 0:
            raise ConfigurationError("fps must be > 0")
        if self.exposure_us < 0:
            raise ConfigurationError("exposure_us must be >= 0")

    def apply_to_camera(self, camera) -> None:
        """Write the camera-resident settings to the hardware.

        Gain is converted from dB to the SDK's index units. Exposure is
        written only in manual mode; in auto mode the Streaming session
        owns the exposure (and writes its own starting value at arm).
        """
        self.validate()
        set_gain_db(camera, self.gain_db)
        camera.black_level = int(self.black_level)
        if not self.auto_exposure:
            camera.exposure_time_us = int(round(self.exposure_us))

    def frozen_manual(self, exposure_us: Optional[float] = None,
                      gain_db: Optional[float] = None) -> "Configuration":
        """Copy configured for Time Series (manual exposure required).

        README: switching Streaming -> Time Series freezes the current
        auto-exposure values as the manual settings for the run — pass
        the controller's values as ``exposure_us`` / ``gain_db``.
        """
        return replace(self,
                       auto_exposure=False,
                       exposure_us=(self.exposure_us
                                    if exposure_us is None else exposure_us),
                       gain_db=(self.gain_db
                                if gain_db is None else gain_db))
