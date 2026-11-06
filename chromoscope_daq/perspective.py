"""Perspective correction (README 'Perspective correction menu').

The user defines 4 source points on an (uncorrected) image and, for each,
the destination coordinates in mm on the corrected plane, plus the X and
Y axis ranges of the output. The transformation is computed with
``cv2.getPerspectiveTransform`` and applied with ``cv2.warpPerspective``.

Output geometry (README): the output image has the same size as the
input; the axis ranges map linearly onto that output grid — x increases
to the right (left edge = x_min), y increases upward (bottom edge =
y_min). Destinations outside the axis ranges fall outside the output
canvas and are therefore black (README: missing values filled with
black).

Matrices (the full definition: points + ranges) are saved/loaded as
json files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

Point2D = Tuple[float, float]

#: Marker in the json file (rejects files that are not ours).
_FILE_TYPE = "chromoscope_perspective"
_VERSION = 1

__all__ = ["PerspectiveError", "PerspectiveTransform", "load_image_rgb8"]


class PerspectiveError(ValueError):
    """Invalid perspective definition or file."""


def _check_points(points, name: str) -> None:
    if len(points) != 4:
        raise PerspectiveError(f"{name}: exactly 4 points are required "
                               f"(got {len(points)})")
    for point in points:
        if len(point) != 2:
            raise PerspectiveError(f"{name}: each point needs (x, y)")


@dataclass
class PerspectiveTransform:
    """Definition + computation of the perspective correction."""

    #: 4 points in input-image pixel coordinates.
    source_points: Tuple[Point2D, ...]
    #: Destination of each source point, in mm.
    dest_points: Tuple[Point2D, ...]
    #: Output X axis range [mm]: (min, max), left edge = min.
    x_range: Tuple[float, float]
    #: Output Y axis range [mm]: (min, max), bottom edge = min.
    y_range: Tuple[float, float]

    def __post_init__(self) -> None:
        self.source_points = tuple((float(x), float(y))
                                   for x, y in self.source_points)
        self.dest_points = tuple((float(x), float(y))
                                 for x, y in self.dest_points)
        self.x_range = (float(self.x_range[0]), float(self.x_range[1]))
        self.y_range = (float(self.y_range[0]), float(self.y_range[1]))

    # -- validation -----------------------------------------------------

    def validate(self) -> None:
        _check_points(self.source_points, "source points")
        _check_points(self.dest_points, "destination points")
        if self.x_range[0] >= self.x_range[1]:
            raise PerspectiveError(
                f"X range {self.x_range} must have min < max")
        if self.y_range[0] >= self.y_range[1]:
            raise PerspectiveError(
                f"Y range {self.y_range} must have min < max")
        if len({(round(x, 6), round(y, 6))
                for x, y in self.source_points}) != 4:
            raise PerspectiveError("source points must be distinct")

    # -- mapping --------------------------------------------------------

    @staticmethod
    def mm_to_output(point_mm: Point2D, x_range: Tuple[float, float],
                     y_range: Tuple[float, float],
                     size: Tuple[int, int]) -> Tuple[float, float]:
        """Map an mm coordinate onto the output pixel grid.

        x_min maps to column 0, y_min maps to the bottom row (axis up).
        """
        width, height = size
        x_mm, y_mm = point_mm
        px = (x_mm - x_range[0]) / (x_range[1] - x_range[0]) * width
        py = (y_range[1] - y_mm) / (y_range[1] - y_range[0]) * height
        return (px, py)

    def compute_matrix(self, size: Tuple[int, int]) -> np.ndarray:
        """3x3 matrix mapping input pixels to output pixels.

        ``size`` is the output (width, height); per README it equals the
        input image size.
        """
        self.validate()
        src = np.asarray(self.source_points, dtype=np.float32)
        dst = np.asarray(
            [self.mm_to_output(point, self.x_range, self.y_range, size)
             for point in self.dest_points], dtype=np.float32)
        try:
            return cv2.getPerspectiveTransform(src, dst)
        except cv2.error as error:
            raise PerspectiveError(
                f"degenerate point selection: {error}") from error

    def apply(self, image: np.ndarray,
              size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Correct ``image``; output size defaults to the image size."""
        out_size = size if size is not None else (image.shape[1],
                                                  image.shape[0])
        matrix = self.compute_matrix(out_size)
        return cv2.warpPerspective(image, matrix, out_size)

    # -- json persistence ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": _FILE_TYPE,
            "version": _VERSION,
            "source_points": [list(point) for point in self.source_points],
            "dest_points_mm": [list(point) for point in self.dest_points],
            "x_range_mm": list(self.x_range),
            "y_range_mm": list(self.y_range),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerspectiveTransform":
        if not isinstance(data, dict):
            raise PerspectiveError("not a json object")
        if data.get("type") != _FILE_TYPE:
            raise PerspectiveError(
                f"not a perspective correction file (type={data.get('type')!r})")
        try:
            return cls(
                source_points=tuple(map(tuple, data["source_points"])),
                dest_points=tuple(map(tuple, data["dest_points_mm"])),
                x_range=tuple(data["x_range_mm"]),
                y_range=tuple(data["y_range_mm"]),
            )
        except KeyError as error:
            raise PerspectiveError(f"missing field {error}") from error
        except (TypeError, ValueError) as error:
            raise PerspectiveError(f"malformed field: {error}") from error

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PerspectiveTransform":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise PerspectiveError(f"cannot read {path}: {error}") from error
        return cls.from_dict(data)


def load_image_rgb8(path: str) -> np.ndarray:
    """Load an image from disk as RGB uint8 for display / point picking.

    16-bit TIFFs (our raw snapshots, 10-bit right-aligned) are converted
    by integer division by 4 like every other 8-bit output; grayscale
    images are replicated to RGB.
    """
    extension = path.lower().rsplit(".", 1)[-1]
    if extension in ("tif", "tiff"):
        import tifffile

        image = tifffile.imread(path)
        if image.ndim == 3:  # (H, W, C)
            image = image[..., :3]
        if image.dtype == np.uint16:
            image = (image >> 2).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(image)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PerspectiveError(f"cannot read image {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.dtype != np.uint8:
        image = (image.astype(np.float32) / 4.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
