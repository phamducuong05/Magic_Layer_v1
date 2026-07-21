"""Safety validation for model reconstruction outputs."""

import math
from typing import Any

import numpy as np
from PIL import Image

from ...core.layerd_refine import expand_mask
from ..roi import SquareROI


class ReconstructionValidationError(ValueError):
    """A model result cannot safely enter later grouping stages."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _real_image_pixels(roi: SquareROI) -> np.ndarray:
    """Return crop pixels that map to the source image rather than padding."""
    # Create a mask marking coordinates of the actual original image within the cropped square.
    # When the crop bounds extend outside the source image, the empty outer areas are padded 
    # with black. We mask them out so we only validate changes on real source pixels.
    real_pixels = np.zeros((roi.size, roi.size), dtype=bool)
    left, top, right, bottom = roi.inner_box
    real_pixels[top:bottom, left:right] = True
    return real_pixels


def _permitted_reconstruction_region(
    hard_mask: np.ndarray,
    *,
    blend_allowance_ratio: float,
) -> np.ndarray:
    """Expand a hard mask by a crop-scale allowance for model blending."""
    if (
        not math.isfinite(blend_allowance_ratio)
        or blend_allowance_ratio < 0
    ):
        raise ValueError(
            "blend_allowance_ratio must be finite and non-negative"
        )
    
    # NOTE: This is a SECOND-STAGE expansion of the reconstruction mask.
    # - 1st Expansion (in mask_preparation.py): Done geometrically to determine what area to inpaint.
    # - 2nd Expansion (here): Done during validation to create a "safety buffer" (tolerance zone).
    # Since models like HD-Painter or Poisson blending can leak/blend colors slightly outside the 
    # original mask boundaries, checking strict equality against the original hard_mask would cause 
    # false validation failures. This expansion allows a small blending boundary margin.
    radius = math.ceil(hard_mask.shape[0] * blend_allowance_ratio)
    if radius == 0:
        return hard_mask.astype(bool)
    kernel_size = 2 * radius + 1
    return expand_mask(
        hard_mask.astype(bool), (kernel_size, kernel_size)
    ).astype(bool)


def _extreme_hole_matches_visible_object(
    hole_pixels: np.ndarray,
    visible_pixels: np.ndarray,
) -> bool:
    """Allow flat black/white output only when visible object RGB supports it."""
    if visible_pixels.size == 0:
        return False
    
    # Catching generator failures where the model output is completely flat/blank:
    # 1. If inpainted region is completely black (max intensity <= 1):
    #    Only allow this if the original visible object itself is extremely dark (95th percentile <= 8).
    if np.max(hole_pixels) <= 1:
        return bool(np.percentile(visible_pixels, 95) <= 8)
        
    # 2. If inpainted region is completely white (min intensity >= 254):
    #    Only allow this if the original visible object itself is extremely bright (5th percentile >= 247).
    if np.min(hole_pixels) >= 254:
        return bool(np.percentile(visible_pixels, 5) >= 247)
        
    return True


def validate_reconstruction_result(
    result: Any,
    *,
    source_crop: Image.Image,
    hard_mask: np.ndarray,
    completion_hole: np.ndarray,
    modal_mask: np.ndarray,
    roi: SquareROI,
    blend_allowance_ratio: float,
) -> Image.Image:
    """Return one safe RGB crop or raise a stage-specific validation error."""
    if not isinstance(result, Image.Image):
        raise ReconstructionValidationError(
            "result_type", "reconstruction result must be one PIL image"
        )
    if result.size != source_crop.size:
        raise ReconstructionValidationError(
            "crop_size_restoration",
            "expected reconstruction size "
            f"{source_crop.size}, got {result.size}",
        )
    if result.mode not in {"RGB", "RGBA", "L"}:
        raise ReconstructionValidationError(
            "result_channels",
            f"unsupported reconstruction image mode {result.mode!r}",
        )

    try:
        rgb_result = result.convert("RGB")
        result_array = np.asarray(rgb_result, dtype=np.uint8)
    except Exception as exc:
        raise ReconstructionValidationError(
            "result_channels",
            f"could not convert reconstruction to RGB: {exc}",
        ) from exc
    if result_array.shape != (*hard_mask.shape, 3):
        raise ReconstructionValidationError(
            "result_channels",
            f"invalid reconstructed RGB shape {result_array.shape}",
        )

    source_array = np.asarray(source_crop.convert("RGB"), dtype=np.uint8)
    real_pixels = _real_image_pixels(roi)
    permitted = _permitted_reconstruction_region(
        hard_mask,
        blend_allowance_ratio=blend_allowance_ratio,
    )

    # HD-Painter resizes the complete crop during super resolution and applies
    # Poisson blending around the inpaint mask. Both operations can alter RGB
    # outside the requested region even when the input hard mask is correct.
    # Enforce the pipeline contract by restoring those pixels instead of
    # rejecting an otherwise usable reconstruction.
    result_array = result_array.copy()
    result_array[~permitted] = source_array[~permitted]
    rgb_result = Image.fromarray(result_array, mode="RGB")

    usable_hole = completion_hole.astype(bool) & real_pixels
    if not np.any(usable_hole):
        raise ReconstructionValidationError(
            "completion_hole", "completion-hole crop is empty"
        )
    hole_pixels = result_array[usable_hole]
    visible_pixels = source_array[modal_mask.astype(bool) & real_pixels]
    if not _extreme_hole_matches_visible_object(
        hole_pixels, visible_pixels
    ):
        raise ReconstructionValidationError(
            "completion_hole",
            "completion-hole RGB is an unsupported blank extreme",
        )
    return rgb_result
