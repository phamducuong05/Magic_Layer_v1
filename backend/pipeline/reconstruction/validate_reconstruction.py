"""Safety validation for model reconstruction outputs."""

import math
from typing import Any

import numpy as np
import cv2
from PIL import Image

from ...core.layerd_refine import expand_mask, refine_with_reference_mask
from ..roi import SquareROI


class ReconstructionValidationError(ValueError):
    """A model result cannot safely enter later grouping stages."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _palette_from_visible_target(
    source: np.ndarray,
    modal_mask: np.ndarray,
    *,
    max_colors: int,
) -> np.ndarray:
    """Extract a compact anti-alias-tolerant palette from visible target RGB."""
    pixels = source[modal_mask.astype(bool)]
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    quantized = pixels // 16
    _, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )
    selected = np.argsort(counts)[::-1][:max_colors]
    palette = [
        np.rint(pixels[inverse == index].mean(axis=0)).astype(np.uint8)
        for index in selected
    ]
    return np.asarray(palette, dtype=np.uint8)


def _lab_distances(
    pixels: np.ndarray, references: np.ndarray
) -> np.ndarray:
    """Return each RGB pixel's distance to its nearest reference color."""
    if pixels.size == 0 or references.size == 0:
        return np.empty(0, dtype=np.float32)
    pixel_lab = cv2.cvtColor(
        pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)
    reference_lab = cv2.cvtColor(
        references.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)
    delta = pixel_lab[:, None, :] - reference_lab[None, :, :]
    delta[..., 1:] *= 0.5
    return np.linalg.norm(delta, axis=-1).min(axis=1)


def reconstruction_color_metrics(
    result: Image.Image,
    *,
    source_crop: Image.Image,
    composition_mask: np.ndarray,
    modal_mask: np.ndarray,
    occluder_mask: np.ndarray,
) -> dict[str, float | None]:
    """Measure color consistency without hard-rejecting generated content."""
    result_array = np.asarray(result.convert("RGB"), dtype=np.uint8)
    source_array = np.asarray(source_crop.convert("RGB"), dtype=np.uint8)
    generated = result_array[composition_mask.astype(bool)]
    target_palette = _palette_from_visible_target(
        source_array, modal_mask, max_colors=16
    )
    occluder_palette = _palette_from_visible_target(
        source_array, occluder_mask, max_colors=16
    )
    target_distance = _lab_distances(generated, target_palette)
    occluder_distance = _lab_distances(generated, occluder_palette)
    source_delta = np.linalg.norm(
        generated.astype(np.float32)
        - source_array[composition_mask.astype(bool)].astype(np.float32),
        axis=1,
    )
    result_gray = cv2.cvtColor(result_array, cv2.COLOR_RGB2GRAY)
    source_gray = cv2.cvtColor(source_array, cv2.COLOR_RGB2GRAY)
    generated_detail = np.abs(
        cv2.Laplacian(result_gray, cv2.CV_32F)
    )[composition_mask.astype(bool)]
    visible_detail = np.abs(
        cv2.Laplacian(source_gray, cv2.CV_32F)
    )[modal_mask.astype(bool)]
    detail_ratio = (
        float(generated_detail.mean() / max(visible_detail.mean(), 1e-6))
        if generated_detail.size and visible_detail.size
        else None
    )
    return {
        "target_lab_distance": (
            float(np.median(target_distance))
            if target_distance.size
            else None
        ),
        "occluder_lab_distance": (
            float(np.median(occluder_distance))
            if occluder_distance.size
            else None
        ),
        "source_persistence_ratio": (
            float(np.mean(source_delta < 8.0))
            if source_delta.size
            else None
        ),
        "generated_detail_ratio": detail_ratio,
    }


def refine_reconstruction_colors(
    result: Image.Image,
    *,
    source_crop: Image.Image,
    composition_mask: np.ndarray,
    modal_mask: np.ndarray,
    strength: float,
    max_colors: int = 16,
) -> Image.Image:
    """Refine write-back RGB from flat colors in the original target modal."""
    result_array = np.asarray(result.convert("RGB"), dtype=np.uint8).copy()
    source_array = np.asarray(source_crop.convert("RGB"), dtype=np.uint8)
    refined = refine_with_reference_mask(
        result_array,
        edit_mask=composition_mask.astype(bool),
        reference_image=source_array,
        reference_mask=modal_mask.astype(bool),
        max_num_colors=max_colors,
        strength=strength,
    )
    return Image.fromarray(refined, mode="RGB")


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
