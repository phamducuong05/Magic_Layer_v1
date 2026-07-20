"""Occlusion roles, reconstruction masks, and hidden-RGB reconstruction."""

from collections.abc import Callable, Sequence
import logging
import math
from typing import Any

import numpy as np
from PIL import Image

from ..core.layerd_refine import expand_mask
from ..core.occlusion import (
    OverlapPair,
    PairDecision,
    assign_pair_roles,
    effective_hole_area,
)
from .roi import SquareROI, crop_array, crop_image, square_roi_from_support
from .types import DetectedObject


logger = logging.getLogger(__name__)


class ReconstructionValidationError(ValueError):
    """A model result cannot safely enter later grouping stages."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def apply_pair_decisions(
    objects: Sequence[DetectedObject], decisions: Sequence[PairDecision]
) -> None:
    """Record each decisive pair's occluder on its occluded object."""
    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.occluder_ids.clear()

    for decision in decisions:
        if decision.ambiguous:
            continue
        objects_by_id[decision.occluded_id].occluder_ids.add(
            decision.occluder_id
        )


def build_reconstruction_masks(
    objects: Sequence[DetectedObject], kernel_size: tuple[int, int]
) -> None:
    """Build constrained masks for objects with assigned occluders."""
    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.reconstruction_mask = None
        if (
            not detected.occluder_ids
            or detected.amodal_mask is None
            or detected.completion_hole_mask is None
        ):
            continue

        occluder_union = np.zeros_like(detected.modal_mask, dtype=bool)
        for occluder_id in detected.occluder_ids:
            occluder_union |= objects_by_id[occluder_id].modal_mask > 0

        expanded_support = expand_mask(
            detected.amodal_mask > 0, kernel_size
        ).astype(bool)
        relevant_occluder = (
            occluder_union
            & expanded_support
            & ~(detected.modal_mask > 0)
        )
        detected.reconstruction_mask = (
            detected.completion_hole_mask | relevant_occluder
        )


def prepare_raw_reconstruction_masks(
    objects: Sequence[DetectedObject],
    retained_pairs: Sequence[OverlapPair],
    kernel_size: tuple[int, int],
    *,
    minimum_hole_area_pixels: int,
    minimum_hole_area_ratio: float,
    tie_tolerance_ratio: float,
) -> list[PairDecision]:
    """Decide pairwise depth and build one reconstruction mask per raw object."""
    effective_areas: dict[str, int] = {}
    for detected in objects:
        raw_area = detected.completion_hole_area
        if raw_area is None:
            continue

        effective_area = effective_hole_area(
            raw_area,
            int(np.count_nonzero(detected.modal_mask)),
            minimum_pixels=minimum_hole_area_pixels,
            minimum_modal_ratio=minimum_hole_area_ratio,
        )
        detected.effective_completion_hole_area = effective_area
        effective_areas[detected.object_id] = effective_area

    decisions = assign_pair_roles(
        retained_pairs,
        effective_areas,
        tie_tolerance_ratio=tie_tolerance_ratio,
    )
    apply_pair_decisions(objects, decisions)
    build_reconstruction_masks(objects, kernel_size)
    return decisions


def _real_image_pixels(roi: SquareROI) -> np.ndarray:
    """Return crop pixels that map to the source image rather than padding."""
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
    if np.max(hole_pixels) <= 1:
        return bool(np.percentile(visible_pixels, 95) <= 8)
    if np.min(hole_pixels) >= 254:
        return bool(np.percentile(visible_pixels, 5) >= 247)
    return True


def _validate_reconstruction_result(
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
    protected = real_pixels & ~permitted
    if np.any(result_array[protected] != source_array[protected]):
        raise ReconstructionValidationError(
            "permitted_region",
            "reconstruction changed source pixels outside the permitted region",
        )

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


def _store_reconstruction_failure(
    detected: DetectedObject,
    *,
    stage: str,
    reason: str,
) -> None:
    """Clear only one invalid result and retain its mask diagnostics."""
    detected.reconstruction_canvas = None
    detected.reconstruction_roi = None
    detected.reconstruction_failure_stage = stage
    detected.reconstruction_failure_reason = reason
    logger.warning(
        "Object reconstruction rejected for %s at %s: %s",
        detected.object_id,
        stage,
        reason,
    )


def reconstruct_objects(
    image: Image.Image,
    objects: Sequence[DetectedObject],
    reconstruct: Callable[[Image.Image, Image.Image, str], Image.Image],
    *,
    context_ratio: float,
    blend_allowance_ratio: float = 0.0,
) -> None:
    """Reconstruct and validate hidden RGB independently for each raw object."""
    for detected in objects:
        detected.reconstruction_canvas = None
        detected.reconstruction_roi = None
        detected.reconstruction_failure_stage = None
        detected.reconstruction_failure_reason = None
        reconstruction_mask = detected.reconstruction_mask
        if reconstruction_mask is None or not np.any(reconstruction_mask):
            continue
        try:
            if detected.amodal_mask is None:
                raise ReconstructionValidationError(
                    "input_preparation", "amodal support mask is missing"
                )
            if detected.completion_hole_mask is None:
                raise ReconstructionValidationError(
                    "input_preparation", "completion-hole mask is missing"
                )
            roi = square_roi_from_support(
                detected.amodal_mask > 0,
                context_ratio=context_ratio,
            )
            source_crop = crop_image(image.convert("RGB"), roi)
            mask_crop = crop_array(reconstruction_mask.astype(bool), roi)
            hole_crop = crop_array(
                detected.completion_hole_mask.astype(bool), roi
            )
            modal_crop = crop_array(detected.modal_mask > 0, roi)
            mask_image = Image.fromarray(
                mask_crop.astype(np.uint8) * 255,
                mode="L",
            )
            prompt = (
                f"Continue the hidden parts of the {detected.semantic_class}, "
                "preserving its visible appearance and surrounding context."
            )
            reconstructed = reconstruct(source_crop, mask_image, prompt)
            validated = _validate_reconstruction_result(
                reconstructed,
                source_crop=source_crop,
                hard_mask=mask_crop,
                completion_hole=hole_crop,
                modal_mask=modal_crop,
                roi=roi,
                blend_allowance_ratio=blend_allowance_ratio,
            )
        except Exception as exc:
            _store_reconstruction_failure(
                detected,
                stage=str(getattr(exc, "stage", "inference")),
                reason=str(exc) or type(exc).__name__,
            )
            continue

        detected.reconstruction_canvas = validated
        detected.reconstruction_roi = roi
