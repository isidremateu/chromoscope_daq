"""Beam profiles extracted from the corrected image (README 'Display modes').

The vertical profile is the luminance along the X = 0 mm column of the
perspective-corrected image; the horizontal profile runs along Y = 0 mm.
Both are expressed in mm coordinates on the corrected scale. Statistics
(mean/sigma) and a Gaussian + baseline fit summarize each profile for
beam steering. Statistics and fit parameters are for information only —
they are never saved to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = ["BeamProfiles", "extract_profiles", "profile_stats",
           "gaussian_overlay", "gaussian_fit", "nice_ticks"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def nice_ticks(lo: float, hi: float, target: int = 6) -> list:
    """Rounded tick values covering [lo, hi] with about ``target`` steps.

    Steps are 1/2/2.5/5 x 10^n; used for the axis grid and profile tick
    labels.
    """
    import math

    span = hi - lo
    if span <= 0:
        return [lo]
    rough = span / max(target, 1)
    exponent = math.floor(math.log10(rough)) if rough > 0 else 0
    step = 10.0 ** exponent
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = factor * 10.0 ** exponent
        if span / step <= target:
            break
    first = math.ceil(lo / step - 1e-9)
    ticks = []
    index = first
    while index * step <= hi + 1e-9:
        ticks.append(round(index * step, 12))
        index += 1
    return ticks


@dataclass
class BeamProfiles:
    """Profiles sampled from one corrected image, in mm coordinates."""

    #: mm coordinate of each horizontal-profile sample (left -> right).
    x_mm: np.ndarray
    #: Luminance along the Y = 0 mm row.
    horizontal: np.ndarray
    #: mm coordinate of each vertical-profile sample (top -> bottom).
    y_mm: np.ndarray
    #: Luminance along the X = 0 mm column.
    vertical: np.ndarray


def extract_profiles(corrected_luma: np.ndarray,
                     x_range: tuple, y_range: tuple,
                     at_x_mm: float = 0.0, at_y_mm: float = 0.0,
                     ) -> BeamProfiles:
    """Extract the two profiles at the given corrected-scale coordinates.

    Coordinates clamped into the axis ranges (the profile sits at the
    nearest edge when 0 mm is outside the display range). Sample
    coordinates use pixel centers, consistent with
    ``PerspectiveTransform.mm_to_output``.
    """
    height, width = corrected_luma.shape
    x_min, x_max = x_range
    y_min, y_max = y_range
    col = int(_clamp((at_x_mm - x_min) / (x_max - x_min) * width,
                     0, width - 1))
    row = int(_clamp((y_max - at_y_mm) / (y_max - y_min) * height,
                     0, height - 1))
    x_mm = x_min + (np.arange(width) + 0.5) * (x_max - x_min) / width
    y_mm = y_max - (np.arange(height) + 0.5) * (y_max - y_min) / height
    return BeamProfiles(
        x_mm=x_mm,
        horizontal=corrected_luma[row, :].astype(np.float64),
        y_mm=y_mm,
        vertical=corrected_luma[:, col].astype(np.float64),
    )


def profile_stats(coords: np.ndarray,
                  values: np.ndarray) -> Dict[str, float]:
    """Intensity-weighted mean position and position spread (mm).

    The README overlays a Gaussian on the profile from the mean/sigma:
    those are the beam center and width along the profile axis, computed
    as luminance-weighted moments over the coordinates.
    """
    x = np.asarray(coords, dtype=np.float64)
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    # Remove the pedestal (black level / background) first: a uniform
    # offset would pull the weighted center toward the profile middle.
    v = v - v.min()
    total = float(v.sum())
    if total <= 0.0:
        return {"mean": 0.0, "sigma": 0.0}
    center = float((x * v).sum() / total)
    sigma = float(np.sqrt((v * (x - center) ** 2).sum() / total))
    return {"mean": center, "sigma": sigma}


def gaussian_overlay(coords: np.ndarray, baseline: float, amplitude: float,
                     center: float, sigma: float) -> np.ndarray:
    """Gaussian evaluated on ``coords`` (for overlay drawing)."""
    coords = np.asarray(coords, dtype=np.float64)
    return baseline + amplitude * np.exp(
        -((coords - center) ** 2) / (2.0 * sigma ** 2))


def gaussian_fit(coords: np.ndarray, values: np.ndarray,
                 iterations: int = 8) -> Dict[str, float]:
    """Gaussian + constant baseline fit (dependency-free).

    Iterative moment estimation: subtract the baseline, compute
    center/sigma as weighted moments, re-estimate the baseline from
    outside a +-2.5 sigma window, repeat. Good to ~1% on clean beams —
    the README only requires the parameters for information during
    operation, without uncertainties.
    """
    x = np.asarray(coords, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    baseline = float(np.percentile(y, 10))
    center = float(x[int(np.argmax(y))])
    sigma = max((x.max() - x.min()) / 6.0, 1e-9)
    z = np.clip(y - baseline, 0.0, None)
    for _ in range(iterations):
        z = np.clip(y - baseline, 0.0, None)
        total = float(z.sum())
        if total <= 0.0:
            break
        center = float((x * z).sum() / total)
        variance = float((z * (x - center) ** 2).sum() / total)
        sigma = max(np.sqrt(variance), 1e-9)
        mask = np.abs(x - center) <= 2.5 * sigma
        if mask.sum() < 5:
            break
        if (~mask).any():
            baseline = float(np.median(y[~mask]))
        else:
            # Window saturated the profile: use the data floor so the
            # next iteration can separate beam from baseline.
            baseline = float(y.min())
    amplitude = float(z.max()) if z.size else 0.0
    return {"baseline": baseline, "amplitude": amplitude,
            "center": center, "sigma": sigma}
