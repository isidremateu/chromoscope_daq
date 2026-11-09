"""Intensity calibration (README 'Intensity calibration').

Transforms luminance (BT.709, computed from the R, G, B channels) into
particle flux units via one of four user-defined functions:

- linear:      y = a * x + b
- quadratic:   y = a * x^2 + b * x + c
- power law:   y = a * x^b + c
- logarithmic: y = a * ln(x + b) + c

Definitions are saved/loaded as json files. Domain edge cases (x + b <= 0
for the logarithm, x = 0 with a negative exponent for the power law) are
clamped to finite values instead of producing inf/NaN.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: Available calibration function types (README).
LINEAR = "linear"
QUADRATIC = "quadratic"
POWER_LAW = "power law"
LOGARITHMIC = "logarithmic"
CALIBRATION_TYPES = (LINEAR, QUADRATIC, POWER_LAW, LOGARITHMIC)

#: Number of coefficients per function type.
COEFF_COUNTS = {LINEAR: 2, QUADRATIC: 3, POWER_LAW: 3, LOGARITHMIC: 3}

_FILE_TYPE = "chromoscope_calibration"
_VERSION = 1

__all__ = ["IntensityCalibration", "CalibrationError", "CALIBRATION_TYPES",
           "COEFF_COUNTS"]


class CalibrationError(ValueError):
    """Invalid calibration definition or file."""


@dataclass
class IntensityCalibration:
    """One calibration function: y = f(luminance) in flux units.

    The function is defined at a reference exposure time
    (``exposure_norm_us``); when the current exposure differs, the flux
    is normalized: flux = f(luminance) * exposure_norm / exposure_current.
    """

    kind: str
    coefficients: Tuple[float, ...]
    exposure_norm_us: float = 1000.0

    def __post_init__(self) -> None:
        self.kind = str(self.kind)
        self.coefficients = tuple(float(c) for c in self.coefficients)
        self.exposure_norm_us = float(self.exposure_norm_us)

    def validate(self) -> None:
        if self.kind not in CALIBRATION_TYPES:
            raise CalibrationError(
                f"unknown calibration type {self.kind!r}; "
                f"expected one of {CALIBRATION_TYPES}")
        expected = COEFF_COUNTS[self.kind]
        if len(self.coefficients) != expected:
            raise CalibrationError(
                f"{self.kind} calibration needs {expected} coefficients, "
                f"got {len(self.coefficients)}")
        for coefficient in self.coefficients:
            if not math.isfinite(coefficient):
                raise CalibrationError("coefficients must be finite")
        if not math.isfinite(self.exposure_norm_us) or self.exposure_norm_us <= 0:
            raise CalibrationError("exposure_norm_us must be > 0")

    def apply(self, luminance,
              exposure_us: Optional[float] = None) -> np.ndarray:
        """Evaluate the calibration; returns a float64 array like the input.

        When ``exposure_us`` (the current exposure) is given, the result
        is normalized by ``exposure_norm_us / exposure_us`` so the flux
        stays consistent across exposure changes.
        """
        self.validate()
        x = np.asarray(luminance, dtype=np.float64)
        if self.kind == LINEAR:
            a, b = self.coefficients
            result = a * x + b
        elif self.kind == QUADRATIC:
            a, b, c = self.coefficients
            result = a * x * x + b * x + c
        elif self.kind == POWER_LAW:
            a, b, c = self.coefficients
            if b < 0:
                # 0**negative would be inf; clamp the term at the origin.
                term = np.where(x > 0.0,
                                np.power(np.where(x > 0.0, x, 1.0), b), 0.0)
            else:
                term = np.power(x, b)
            result = a * term + c
        else:
            # logarithmic: x + b must stay positive for ln; clamp to a
            # tiny positive value so the result stays finite.
            a, b, c = self.coefficients
            result = a * np.log(np.maximum(x + b, 1e-12)) + c
        if exposure_us is not None:
            current = float(exposure_us)
            if current > 0:
                result = result * (self.exposure_norm_us / current)
        return result

    # -- json persistence -------------------------------------------------

    def to_dict(self) -> dict:
        return {"type": _FILE_TYPE, "version": _VERSION,
                "kind": self.kind,
                "coefficients": list(self.coefficients),
                "exposure_norm_us": self.exposure_norm_us}

    @classmethod
    def from_dict(cls, data: dict) -> "IntensityCalibration":
        if not isinstance(data, dict):
            raise CalibrationError("not a json object")
        if data.get("type") != _FILE_TYPE:
            raise CalibrationError(
                f"not a calibration file (type={data.get('type')!r})")
        try:
            return cls(kind=data["kind"],
                       coefficients=tuple(data["coefficients"]),
                       # Older files without the field keep the default.
                       exposure_norm_us=data.get("exposure_norm_us", 1000.0))
        except KeyError as error:
            raise CalibrationError(f"missing field {error}") from error
        except (TypeError, ValueError) as error:
            raise CalibrationError(f"malformed field: {error}") from error

    def save_json(self, path: str) -> None:
        self.validate()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "IntensityCalibration":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(f"cannot read {path}: {error}") from error
        calibration = cls.from_dict(data)
        calibration.validate()
        return calibration
