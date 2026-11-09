"""Variant saving: corrected / calibrated copies next to the raw data.

README rules:
- Raw images are always saved; variants are additional.
- Variants use the same file format as the raw data of the current mode,
  with two exceptions: the intensity-calibrated variant is always a
  32-bit float TIFF, and variant saving is not available in Streaming
  (raw mp4 only).
- Corrected images without intensity calibration keep the 10-bit-in-16-bit
  rule (uint16 values 0-1023).
- Profiles are never saved.

File naming: the raw name plus a `_corrected` / `_calibrated` suffix
(`_OME` appended for OME-TIFFs, mirroring the raw Time Series files).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .acquisition import FrameBundle, TiffStackWriter, build_filename
from .acquisition.processing import (save_float32_tif, save_jpg8, save_tif16,
                                     to_bgr8)
from .calibration import IntensityCalibration
from .perspective import PerspectiveTransform

CORRECTED_SUFFIX = "_corrected"
CALIBRATED_SUFFIX = "_calibrated"

__all__ = ["save_snapshot_variants", "TimeSeriesVariantWriters",
           "corrected_rgb16", "corrected_scalar", "CORRECTED_SUFFIX",
           "CALIBRATED_SUFFIX"]


def corrected_rgb16(bundle: FrameBundle,
                    transform: PerspectiveTransform) -> np.ndarray:
    """Perspective-corrected color frame (uint16, 10-bit values)."""
    return transform.apply(bundle.rgb16)


def corrected_scalar(bundle: FrameBundle, transform: PerspectiveTransform,
                     calibration: Optional[IntensityCalibration],
                     exposure_us: Optional[float] = None) -> np.ndarray:
    """Corrected luminance, or normalized flux when a calibration is given.

    ``exposure_us`` is the current exposure time; the flux is normalized
    by the calibration's reference exposure (see IntensityCalibration.apply).
    """
    luma = transform.apply(bundle.luminance16)
    if calibration is None:
        return luma
    return calibration.apply(luma, exposure_us)


def save_snapshot_variants(bundle: FrameBundle,
                           transform: PerspectiveTransform,
                           calibration: Optional[IntensityCalibration],
                           path_no_ext: str, fmt: str,
                           save_corrected: bool,
                           save_calibrated: bool,
                           exposure_us: Optional[float] = None
                           ) -> List[str]:
    """Save the Snapshot-mode variants; returns the written paths."""
    saved: List[str] = []
    if save_corrected:
        corrected = corrected_rgb16(bundle, transform)
        ext = "jpg" if fmt.lower().startswith("jpg") else "tif"
        path = f"{path_no_ext}{CORRECTED_SUFFIX}.{ext}"
        if ext == "jpg":
            save_jpg8(path, to_bgr8(corrected))
        else:
            save_tif16(path, corrected)
        saved.append(path)
    if save_calibrated:
        flux = corrected_scalar(bundle, transform, calibration, exposure_us)
        path = f"{path_no_ext}{CALIBRATED_SUFFIX}.tif"
        save_float32_tif(path, flux)
        saved.append(path)
    return saved


class TimeSeriesVariantWriters:
    """Multi-frame variant writers for Time Series runs.

    Mirrors the raw formats: each enabled variant is written as a classic
    multi-frame TIFF plus an OME-TIFF. The calibrated variant stores
    float32 flux values; the corrected variant uint16 (10-bit rule).
    """

    def __init__(self, timestamp: str, identifier: str, save_dir: str,
                 transform: PerspectiveTransform,
                 calibration: Optional[IntensityCalibration],
                 save_corrected: bool, save_calibrated: bool,
                 exposure_us: Optional[float] = None) -> None:
        self._transform = transform
        self._calibration = calibration
        self._exposure_us = exposure_us
        self._corrected: Optional[tuple] = None
        self._calibrated: Optional[tuple] = None
        if save_corrected:
            classic = os.path.join(
                save_dir, build_filename(timestamp,
                                         identifier + CORRECTED_SUFFIX,
                                         "tif"))
            ome = os.path.join(
                save_dir, build_filename(timestamp,
                                         identifier + CORRECTED_SUFFIX + "_OME",
                                         "tif"))
            self._corrected = (TiffStackWriter(classic, ome=False),
                               TiffStackWriter(ome, ome=True))
        if save_calibrated:
            classic = os.path.join(
                save_dir, build_filename(timestamp,
                                         identifier + CALIBRATED_SUFFIX,
                                         "tif"))
            ome = os.path.join(
                save_dir, build_filename(timestamp,
                                         identifier + CALIBRATED_SUFFIX + "_OME",
                                         "tif"))
            self._calibrated = (TiffStackWriter(classic, ome=False),
                                TiffStackWriter(ome, ome=True))

    def write(self, bundle: FrameBundle) -> None:
        if self._corrected is not None:
            corrected = corrected_rgb16(bundle, self._transform)
            self._corrected[0].write(corrected)
            self._corrected[1].write(corrected)
        if self._calibrated is not None:
            flux = corrected_scalar(bundle, self._transform,
                                    self._calibration,
                                    self._exposure_us).astype(np.float32)
            self._calibrated[0].write(flux)
            self._calibrated[1].write(flux)

    def close(self) -> None:
        for pair in (self._corrected, self._calibrated):
            if pair is not None:
                pair[0].close()
                pair[1].close()
