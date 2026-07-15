"""Reusable padded square regions for aligned per-object pipeline crops."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SquareROI:
    """A square crop in full-image coordinates, including outside padding."""

    x: int
    y: int
    size: int
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("ROI size must be positive")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_width, self.image_height

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Return the unclamped PIL-style crop box."""
        return self.x, self.y, self.x + self.size, self.y + self.size

    @property
    def clipped_box(self) -> tuple[int, int, int, int]:
        """Return the part of the ROI that intersects the source image."""
        left, top, right, bottom = self.box
        return (
            max(0, left),
            max(0, top),
            min(self.image_width, right),
            min(self.image_height, bottom),
        )

    @property
    def padding(self) -> tuple[int, int, int, int]:
        """Return crop padding as ``(left, top, right, bottom)``."""
        left, top, right, bottom = self.box
        return (
            max(0, -left),
            max(0, -top),
            max(0, right - self.image_width),
            max(0, bottom - self.image_height),
        )

    @property
    def inner_box(self) -> tuple[int, int, int, int]:
        """Return crop coordinates occupied by real source-image pixels."""
        pad_left, pad_top, pad_right, pad_bottom = self.padding
        return (
            pad_left,
            pad_top,
            self.size - pad_right,
            self.size - pad_bottom,
        )


def square_roi_from_support(
    support: np.ndarray, *, context_ratio: float
) -> SquareROI:
    """Build an expanded square ROI around non-empty full-image support."""
    if support.ndim != 2:
        raise ValueError("support must be a two-dimensional mask")
    if not math.isfinite(context_ratio) or context_ratio < 0:
        raise ValueError("context_ratio must be finite and non-negative")

    positive_y, positive_x = np.nonzero(support)
    if positive_x.size == 0:
        raise ValueError("support mask must be non-empty")

    x0, x1 = int(positive_x.min()), int(positive_x.max()) + 1
    y0, y1 = int(positive_y.min()), int(positive_y.max()) + 1
    tight_side = max(x1 - x0, y1 - y0)
    size = max(1, math.ceil(tight_side * (1 + 2 * context_ratio)))
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2

    return SquareROI(
        x=math.floor(center_x - size / 2),
        y=math.floor(center_y - size / 2),
        size=size,
        image_width=support.shape[1],
        image_height=support.shape[0],
    )


def crop_array(
    array: np.ndarray, roi: SquareROI, *, fill_value=0
) -> np.ndarray:
    """Crop and pad a NumPy image or mask using ``roi`` geometry."""
    if array.ndim < 2 or array.shape[:2] != (
        roi.image_height,
        roi.image_width,
    ):
        raise ValueError("array dimensions must match the ROI source image")

    result = np.full(
        (roi.size, roi.size, *array.shape[2:]),
        fill_value,
        dtype=array.dtype,
    )
    source_left, source_top, source_right, source_bottom = roi.clipped_box
    crop_left, crop_top, crop_right, crop_bottom = roi.inner_box
    result[crop_top:crop_bottom, crop_left:crop_right] = array[
        source_top:source_bottom,
        source_left:source_right,
    ]
    return result


def crop_image(image: Image.Image, roi: SquareROI) -> Image.Image:
    """Crop a PIL image with zero padding outside its boundaries."""
    if image.size != roi.image_size:
        raise ValueError("image dimensions must match the ROI source image")
    return image.crop(roi.box)


def restore_array(crop: np.ndarray, roi: SquareROI) -> np.ndarray:
    """Remove ROI padding and paste a crop into full-image coordinates."""
    if crop.ndim < 2 or crop.shape[:2] != (roi.size, roi.size):
        raise ValueError("crop dimensions must match the square ROI")

    restored = np.zeros(
        (roi.image_height, roi.image_width, *crop.shape[2:]),
        dtype=crop.dtype,
    )
    source_left, source_top, source_right, source_bottom = roi.inner_box
    dest_left, dest_top, dest_right, dest_bottom = roi.clipped_box
    restored[dest_top:dest_bottom, dest_left:dest_right] = crop[
        source_top:source_bottom,
        source_left:source_right,
    ]
    return restored
