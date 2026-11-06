"""Pure display helpers: frame conversion and view geometry.

No tkinter here — the geometry math and frame conversion are unit
testable without a display. The main window applies them to PIL/tkinter
objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..acquisition import FrameBundle

__all__ = ["bundle_to_rgb8", "normalize_roi", "ViewGeometry", "compute_view",
           "crop_for_view"]


def bundle_to_rgb8(bundle: FrameBundle) -> np.ndarray:
    """RGB uint8 view of a frame bundle (PIL/tkinter use RGB order)."""
    return cv2.cvtColor(bundle.bgr8, cv2.COLOR_BGR2RGB)


def normalize_roi(roi: Optional[Tuple[int, int, int, int]],
                  image_size: Tuple[int, int],
                  min_size: int = 2) -> Optional[Tuple[int, int, int, int]]:
    """Clamp a ROI (x0, y0, x1, y1) to the image; None if degenerate."""
    if roi is None:
        return None
    width, height = image_size
    x0, y0, x1, y1 = roi
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return (x0, y0, x1, y1)


@dataclass
class ViewGeometry:
    """Mapping between image pixels and canvas pixels for the current view.

    The source crop (``crop_x``..) is displayed at uniform ``scale``,
    centered in the canvas by ``offset_x``/``offset_y``.
    """

    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    scale: float
    offset_x: float
    offset_y: float

    def canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.offset_x) / self.scale + self.crop_x,
                (y - self.offset_y) / self.scale + self.crop_y)

    def image_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.crop_x) * self.scale + self.offset_x,
                (y - self.crop_y) * self.scale + self.offset_y)


def compute_view(image_size: Tuple[int, int],
                 canvas_size: Tuple[int, int],
                 roi: Optional[Tuple[int, int, int, int]] = None
                 ) -> ViewGeometry:
    """Geometry for displaying the full image or an ROI zoom, fitted."""
    image_w, image_h = image_size
    canvas_w, canvas_h = canvas_size
    if roi is not None:
        crop_x, crop_y, x1, y1 = roi
        crop_w, crop_h = x1 - crop_x, y1 - crop_y
    else:
        crop_x, crop_y, crop_w, crop_h = 0, 0, image_w, image_h
    scale = min(canvas_w / crop_w, canvas_h / crop_h)
    return ViewGeometry(
        crop_x=crop_x, crop_y=crop_y, crop_w=crop_w, crop_h=crop_h,
        scale=scale,
        offset_x=(canvas_w - crop_w * scale) / 2.0,
        offset_y=(canvas_h - crop_h * scale) / 2.0)


def crop_for_view(rgb8: np.ndarray, geom: ViewGeometry) -> np.ndarray:
    """Source crop of the display image for the given view geometry."""
    return rgb8[geom.crop_y:geom.crop_y + geom.crop_h,
                geom.crop_x:geom.crop_x + geom.crop_w]
