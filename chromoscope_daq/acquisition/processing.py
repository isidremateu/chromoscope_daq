"""Raw-frame processing: demosaic, BT.709 luminance, 8-bit conversion, savers."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import FrameBundle

#: Bayer pattern of the CS165CU1/M as interpreted by cv2.cvtColor.
#: Configurable so it can be fixed without touching acquisition logic.
BAYER_CODE = cv2.COLOR_BAYER_BG2RGB

#: BT.709 luminance weights (README): L = 0.2126 R + 0.7152 G + 0.0722 B.
_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722

#: 10-bit data stored in 16-bit buffers; JPG/MP4 are 8-bit via integer /4.
BIT_DEPTH = 10
_SATURATION = (1 << BIT_DEPTH) - 1  # 1023
_TO_8BIT_SHIFT = BIT_DEPTH - 8  # >> 2 == integer // 4


def demosaic_bayer(raw16: np.ndarray) -> np.ndarray:
    """Demosaic a 16-bit Bayer frame to RGB (uint16)."""
    return cv2.cvtColor(raw16, BAYER_CODE)


def luminance_bt709(rgb16: np.ndarray) -> np.ndarray:
    """BT.709 luminance of an RGB uint16 image, clipped to [0, 1023]."""
    rgb = rgb16.astype(np.float32)
    luma = (_LUMA_R * rgb[..., 0] + _LUMA_G * rgb[..., 1]
            + _LUMA_B * rgb[..., 2])
    return np.clip(luma, 0.0, float(_SATURATION))


def to_bgr8(rgb16: np.ndarray) -> np.ndarray:
    """RGB uint16 (10-bit) -> BGR uint8 via integer division by 4."""
    rgb8 = (rgb16 >> _TO_8BIT_SHIFT).astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)


def default_processor(raw16: np.ndarray) -> FrameBundle:
    """Default Unprocessed-color-space pipeline: demosaic -> luminance -> 8-bit."""
    rgb16 = demosaic_bayer(raw16)
    luma16 = luminance_bt709(rgb16).astype(np.uint16)
    bgr8 = to_bgr8(rgb16)
    return FrameBundle(raw16, rgb16, luma16, bgr8, float(luma16.mean()))


def save_tif16(path: str, raw16: np.ndarray) -> None:
    """Save a raw 16-bit frame as TIFF (10-bit data right-aligned in uint16)."""
    import tifffile

    tifffile.imwrite(path, raw16)


def save_jpg8(path: str, bgr8: np.ndarray) -> None:
    """Save an 8-bit BGR frame as JPEG."""
    cv2.imwrite(path, bgr8, [cv2.IMWRITE_JPEG_QUALITY, 95])


def save_float32_tif(path: str, image: np.ndarray) -> None:
    """Save an intensity-calibrated image as 32-bit float TIFF."""
    import tifffile

    tifffile.imwrite(path, np.asarray(image, dtype=np.float32))
