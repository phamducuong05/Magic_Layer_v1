"""Occlusion role assignment and reconstruction mask construction."""

from collections.abc import Sequence

import cv2
import numpy as np

from ...core.layerd_refine import expand_mask
from ...core.logging import get_logger, log_event
from ...core.occlusion import (
    OverlapPair,
    PairDecision,
    assign_directional_pair_roles,
    effective_hole_area,
)
from ..roi import SquareROI, square_roi_from_support
from ..types import DetectedObject


logger = get_logger(__name__)


def _mask_inside_roi(shape: tuple[int, int], roi: SquareROI) -> np.ndarray:
    """Return the source-image pixels covered by a possibly padded ROI."""
    result = np.zeros(shape, dtype=bool)
    left, top, right, bottom = roi.clipped_box
    result[top:bottom, left:right] = True
    return result


def _directional_reconstruction_mask(
    target: DetectedObject,
    occluder: DetectedObject,
    *,
    composition_margin_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a directional write-back mask anchored to real overlap evidence."""
    hole = target.completion_hole_mask.astype(bool)
    occluder_modal = occluder.modal_mask > 0
    exact_seed = hole & occluder_modal
    if not np.any(exact_seed):
        return exact_seed, np.zeros_like(hole)

    if composition_margin_pixels:
        radius = composition_margin_pixels
        occluder_support = expand_mask(
            occluder_modal,
            (2 * radius + 1, 2 * radius + 1),
        ).astype(bool)
    else:
        occluder_support = occluder_modal
    candidate = hole & occluder_support

    _, labels = cv2.connectedComponents(
        candidate.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(candidate)
    for label in np.unique(labels[exact_seed]):
        if label != 0:
            filtered |= labels == label
    filtered &= target.amodal_mask > 0
    filtered &= ~(target.modal_mask > 0)
    return exact_seed, filtered


def apply_pair_decisions(
    objects: Sequence[DetectedObject], decisions: Sequence[PairDecision]
) -> None:
    """Record every valid directional occluder on its target object."""
    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.occluder_ids.clear()
        detected.occluder_classes.clear()

    for decision in decisions:
        directions = decision.reconstruction_directions
        if not directions and not decision.ambiguous:
            directions = ((decision.occluded_id, decision.occluder_id),)
        if not directions:
            log_event(
                logger,
                "depth_ordering",
                "pair_assignment",
                first_id=decision.first_id,
                second_id=decision.second_id,
                decision="skip",
                reason="no_directional_completion_overlap",
            )
            continue
        for occluded_id, occluder_id in directions:
            occluded = objects_by_id[occluded_id]
            occluder = objects_by_id[occluder_id]
            occluded.occluder_ids.add(occluder_id)
            occluded.occluder_classes.add(occluder.semantic_class)
            log_event(
                logger,
                "depth_ordering",
                "pair_assignment",
                first_id=decision.first_id,
                second_id=decision.second_id,
                decision="assign",
                occluded_id=occluded_id,
                occluder_id=occluder_id,
            )


def build_reconstruction_masks(
    objects: Sequence[DetectedObject],
    kernel_size: tuple[int, int],
    *,
    generation_mask_dilation_pixels: int = 0,
    generation_mask_closing_pixels: int = 0,
    support_margin_pixels: int | None = None,
    composition_margin_pixels: int | None = None,
    context_ratio: float = 0.0,
) -> None:
    """Build separate directional write-back and model-generation masks."""
    for value in (
        generation_mask_dilation_pixels,
        generation_mask_closing_pixels,
    ):
        if value < 0:
            raise ValueError("reconstruction morphology settings must be non-negative")
    if support_margin_pixels is None:
        support_margin_pixels = max(kernel_size) // 2
    if composition_margin_pixels is None:
        composition_margin_pixels = support_margin_pixels
    if support_margin_pixels < 0 or composition_margin_pixels < 0:
        raise ValueError("reconstruction margin settings must be non-negative")

    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.reconstruction_seed_mask = None
        detected.reconstruction_mask = None
        detected.reconstruction_generation_seed_mask = None
        detected.reconstruction_generation_mask = None
        detected.reconstruction_occluder_mask = None
        detected.reconstruction_input_roi = None

        if (
            not detected.occluder_ids
            or detected.amodal_mask is None
            or detected.completion_hole_mask is None
        ):
            missing = []
            if not detected.occluder_ids:
                missing.append("no_assigned_occluder")
            if detected.amodal_mask is None:
                missing.append("missing_amodal_mask")
            if detected.completion_hole_mask is None:
                missing.append("missing_completion_hole_mask")
            log_event(
                logger,
                "reconstruction_mask",
                "decision",
                object_id=detected.object_id,
                decision="skip",
                reason=",".join(missing),
            )
            continue

        occluder_union = np.zeros_like(detected.modal_mask, dtype=bool)
        exact_seed = np.zeros_like(detected.modal_mask, dtype=bool)
        composition_mask = np.zeros_like(detected.modal_mask, dtype=bool)
        for occluder_id in detected.occluder_ids:
            occluder = objects_by_id[occluder_id]
            occluder_union |= occluder.modal_mask > 0
            directional_seed, directional_mask = (
                _directional_reconstruction_mask(
                    detected,
                    occluder,
                    composition_margin_pixels=composition_margin_pixels,
                )
            )
            exact_seed |= directional_seed
            composition_mask |= directional_mask

        if not np.any(composition_mask):
            log_event(
                logger,
                "reconstruction_mask",
                "decision",
                object_id=detected.object_id,
                decision="skip",
                reason="no_filtered_directional_hole",
            )
            continue

        # Freeze ROI before adding the full occluder so the crop stays centered
        # on the target rather than growing to the occluder's complete bounds.
        roi = square_roi_from_support(
            (detected.amodal_mask > 0) | composition_mask,
            context_ratio=context_ratio,
        )
        roi_mask = _mask_inside_roi(composition_mask.shape, roi)
        relevant_occluder = (
            occluder_union & roi_mask & ~(detected.modal_mask > 0)
        )
        generation_seed = composition_mask | relevant_occluder
        generation_mask = generation_seed.copy()
        if generation_mask_closing_pixels:
            radius = generation_mask_closing_pixels
            closing_kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
            generation_mask = cv2.morphologyEx(
                generation_mask.astype(np.uint8),
                cv2.MORPH_CLOSE,
                closing_kernel,
            ).astype(bool)
        if generation_mask_dilation_pixels:
            radius = generation_mask_dilation_pixels
            generation_mask = expand_mask(
                generation_mask,
                (2 * radius + 1, 2 * radius + 1),
            ).astype(bool)
        generation_mask &= roi_mask
        generation_mask &= ~(detected.modal_mask > 0)
        generation_mask |= composition_mask

        detected.reconstruction_seed_mask = exact_seed
        detected.reconstruction_mask = composition_mask
        detected.reconstruction_generation_seed_mask = generation_seed
        detected.reconstruction_generation_mask = generation_mask
        detected.reconstruction_occluder_mask = relevant_occluder
        detected.reconstruction_input_roi = roi
        log_event(
            logger,
            "reconstruction_mask",
            "decision",
            object_id=detected.object_id,
            decision="created",
            occluder_ids=sorted(detected.occluder_ids),
            completion_hole_pixels=int(
                np.count_nonzero(detected.completion_hole_mask)
            ),
            directional_seed_pixels=int(np.count_nonzero(exact_seed)),
            relevant_occluder_pixels=int(
                np.count_nonzero(relevant_occluder)
            ),
            composition_pixels=int(np.count_nonzero(composition_mask)),
            generation_pixels=int(np.count_nonzero(generation_mask)),
            roi=(roi.x, roi.y, roi.size),
        )


def prepare_raw_reconstruction_masks(
    objects: Sequence[DetectedObject],
    retained_pairs: Sequence[OverlapPair],
    kernel_size: tuple[int, int],
    *,
    minimum_hole_area_pixels: int,
    minimum_hole_area_ratio: float,
    tie_tolerance_ratio: float,
    generation_mask_dilation_pixels: int = 0,
    generation_mask_closing_pixels: int = 0,
    support_margin_pixels: int | None = None,
    composition_margin_pixels: int | None = None,
    context_ratio: float = 0.0,
) -> list[PairDecision]:
    """Filter directional holes, decide depth, and build raw-object masks."""
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
        log_event(
            logger,
            "depth_ordering",
            "object_hole",
            object_id=detected.object_id,
            raw_hole_area=raw_area,
            effective_hole_area=effective_area,
            modal_area=int(np.count_nonzero(detected.modal_mask)),
            decision="retain" if effective_area else "suppress_as_noise",
        )

    if composition_margin_pixels is None:
        composition_margin_pixels = (
            max(kernel_size) // 2
            if support_margin_pixels is None
            else support_margin_pixels
        )
    objects_by_id = {detected.object_id: detected for detected in objects}
    directional_areas: dict[OverlapPair, int] = {}
    for first_id, second_id in retained_pairs:
        first = objects_by_id[first_id]
        second = objects_by_id[second_id]
        for target, occluder in ((first, second), (second, first)):
            _, directional_mask = _directional_reconstruction_mask(
                target,
                occluder,
                composition_margin_pixels=composition_margin_pixels,
            )
            raw_directional_area = int(np.count_nonzero(directional_mask))
            directional_areas[(target.object_id, occluder.object_id)] = (
                effective_hole_area(
                    raw_directional_area,
                    int(np.count_nonzero(target.modal_mask)),
                    minimum_pixels=minimum_hole_area_pixels,
                    minimum_modal_ratio=minimum_hole_area_ratio,
                )
            )

    decisions = assign_directional_pair_roles(
        retained_pairs,
        directional_areas,
        tie_tolerance_ratio=tie_tolerance_ratio,
    )
    for decision in decisions:
        log_event(
            logger,
            "depth_ordering",
            "pair_decision",
            first_id=decision.first_id,
            second_id=decision.second_id,
            first_hole_area=effective_areas.get(decision.first_id, 0),
            second_hole_area=effective_areas.get(decision.second_id, 0),
            first_hidden_by_second=decision.first_hidden_by_second_area,
            second_hidden_by_first=decision.second_hidden_by_first_area,
            reconstruction_directions=decision.reconstruction_directions,
            decision="ambiguous" if decision.ambiguous else "ordered",
            occluded_id=decision.occluded_id,
            occluder_id=decision.occluder_id,
        )
    apply_pair_decisions(objects, decisions)
    build_reconstruction_masks(
        objects,
        kernel_size,
        generation_mask_dilation_pixels=generation_mask_dilation_pixels,
        generation_mask_closing_pixels=generation_mask_closing_pixels,
        support_margin_pixels=support_margin_pixels,
        composition_margin_pixels=composition_margin_pixels,
        context_ratio=context_ratio,
    )
    return decisions
