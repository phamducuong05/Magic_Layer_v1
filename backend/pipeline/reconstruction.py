"""Occlusion roles, reconstruction masks, and hidden-RGB reconstruction."""

from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from ..core.layerd_refine import expand_mask
from ..core.occlusion import (
    OverlapPair,
    PairDecision,
    assign_pair_roles,
    effective_hole_area,
)
from .roi import crop_array, crop_image, square_roi_from_support
from .types import DetectedObject


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


def reconstruct_objects(
    image: Image.Image,
    objects: Sequence[DetectedObject],
    reconstruct: Callable[[Image.Image, Image.Image, str], Image.Image],
    *,
    context_ratio: float,
) -> None:
    """Reconstruct hidden RGB in one padded square crop per selected object."""
    for detected in objects:
        detected.reconstruction_canvas = None
        detected.reconstruction_roi = None
        reconstruction_mask = detected.reconstruction_mask
        if reconstruction_mask is None or not np.any(reconstruction_mask):
            continue

        roi = square_roi_from_support(
            detected.amodal_mask > 0,
            context_ratio=context_ratio,
        )
        source_crop = crop_image(image.convert("RGB"), roi)
        mask_crop = crop_array(reconstruction_mask.astype(bool), roi)
        mask_image = Image.fromarray(
            mask_crop.astype(np.uint8) * 255,
            mode="L",
        )
        prompt = (
            f"Continue the hidden parts of the {detected.semantic_class}, "
            "preserving its visible appearance and surrounding context."
        )
        reconstructed = reconstruct(source_crop, mask_image, prompt)
        expected_size = (roi.size, roi.size)
        if reconstructed.size != expected_size:
            reconstructed = reconstructed.resize(
                expected_size, Image.Resampling.LANCZOS
            )
        detected.reconstruction_canvas = reconstructed.convert("RGB")
        detected.reconstruction_roi = roi
