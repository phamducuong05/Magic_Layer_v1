"""Cross-class overlap linking and explicit-model amodal completion."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from ...core.occlusion import (
    ObjectBounds,
    OverlapPair,
    find_cross_class_overlaps,
)
from ...core.logging import get_logger, log_event
from ..types import DetectedObject
from .validate_completion import (
    _store_modal_fallback,
    _store_valid_completion,
    _validated_amodal_mask,
)


logger = get_logger(__name__)


def link_overlap_partners(
    objects: Sequence[DetectedObject],
    pairs: Sequence[OverlapPair] | None = None,
) -> list[OverlapPair]:
    """Link prefiltered external pairs, or discover legacy overlaps."""
    for detected in objects:
        detected.overlap_partner_ids.clear()

    linked_pairs = (
        list(pairs)
        if pairs is not None
        else find_cross_class_overlaps(
            [
                ObjectBounds(
                    object_id=detected.object_id,
                    semantic_class=detected.semantic_class,
                    bbox=detected.original_modal_bbox,
                )
                for detected in objects
            ]
        )
    )
    objects_by_id = {detected.object_id: detected for detected in objects}
    for first_id, second_id in linked_pairs:
        if first_id not in objects_by_id or second_id not in objects_by_id:
            raise ValueError(
                "overlap pair references unknown objects: "
                f"{first_id!r}, {second_id!r}"
            )
        objects_by_id[first_id].overlap_partner_ids.add(second_id)
        objects_by_id[second_id].overlap_partner_ids.add(first_id)
        log_event(
            logger,
            "overlap_detection",
            "pair_decision",
            first_id=first_id,
            second_id=second_id,
            decision="completion_candidate",
            reason="external_cross_class_bbox_overlap",
        )

    return linked_pairs


def get_completion_candidates(
    objects: Sequence[DetectedObject],
) -> list[DetectedObject]:
    """Return overlapping objects once each, preserving object order."""
    candidates = [
        detected for detected in objects if detected.overlap_partner_ids
    ]
    for detected in objects:
        log_event(
            logger,
            "completion",
            "object_decision",
            object_id=detected.object_id,
            decision=(
                "complete" if detected.overlap_partner_ids else "bypass"
            ),
            overlap_partner_ids=sorted(detected.overlap_partner_ids),
            reason=(
                "cross_class_bbox_overlap"
                if detected.overlap_partner_ids
                else "no_cross_class_bbox_overlap"
            ),
        )
    return candidates


def complete_objects(
    image: Image.Image,
    candidates: Sequence[DetectedObject],
    completion_model: Any,
    *,
    max_area_growth_ratio: float,
    max_bbox_growth_ratio: float,
) -> None:
    """Complete candidates, validating each output with safe modal fallback."""
    if not candidates:
        return

    for detected in candidates:
        detected.completion_failure_stage = None
        detected.completion_failure_reason = None

    try:
        outputs = completion_model.complete(
            image,
            [detected.modal_mask for detected in candidates],
            [detected.bbox for detected in candidates],
        )
    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        logger.warning(
            "Shared completion inference failed; using modal fallback.",
            exc_info=True,
        )
        for detected in candidates:
            _store_modal_fallback(
                detected, stage="inference", reason=reason
            )
            log_event(
                logger,
                "completion",
                "object_decision",
                object_id=detected.object_id,
                decision="modal_fallback",
                reason=reason,
                failure_stage="inference",
            )
        return

    if not isinstance(outputs, Sequence) or len(outputs) != len(candidates):
        reason = (
            f"returned {len(outputs)} outputs for {len(candidates)} candidates"
            if isinstance(outputs, Sequence)
            else "returned a non-sequence output"
        )
        logger.warning(
            "Completion returned %s outputs for %d candidates; using modal "
            "fallback for the batch.",
            len(outputs) if isinstance(outputs, Sequence) else "non-sequence",
            len(candidates),
        )
        for detected in candidates:
            _store_modal_fallback(
                detected, stage="output_contract", reason=reason
            )
            log_event(
                logger,
                "completion",
                "object_decision",
                object_id=detected.object_id,
                decision="modal_fallback",
                reason=reason,
                failure_stage="output_contract",
            )
        return

    for detected, output in zip(candidates, outputs):
        amodal_mask = _validated_amodal_mask(
            detected,
            output,
            max_area_growth_ratio=max_area_growth_ratio,
            max_bbox_growth_ratio=max_bbox_growth_ratio,
        )
        if amodal_mask is None:
            logger.warning(
                "Invalid completion output for object %s; using modal fallback.",
                detected.object_id,
            )
            _store_modal_fallback(detected)
            log_event(
                logger,
                "completion",
                "object_decision",
                object_id=detected.object_id,
                decision="modal_fallback",
                reason=(
                    detected.completion_failure_reason
                    or "completion_validation_failed"
                ),
                failure_stage=detected.completion_failure_stage,
            )
            continue
        _store_valid_completion(detected, amodal_mask)
        log_event(
            logger,
            "completion",
            "object_decision",
            object_id=detected.object_id,
            decision="accepted",
            modal_area=int(np.count_nonzero(detected.modal_mask)),
            amodal_area=int(np.count_nonzero(amodal_mask)),
            completion_hole_area=detected.completion_hole_area,
        )


def filter_pairs_by_amodal_overlap(
    objects: Sequence[DetectedObject],
    pairs: Sequence[OverlapPair],
) -> list[OverlapPair]:
    """Keep pairs whose two validated amodal masks overlap by at least a pixel."""
    objects_by_id = {detected.object_id: detected for detected in objects}
    retained: list[OverlapPair] = []

    for first_id, second_id in pairs:
        first_amodal = objects_by_id[first_id].amodal_mask
        second_amodal = objects_by_id[second_id].amodal_mask
        overlaps = (
            first_amodal is not None
            and second_amodal is not None
            and np.any((first_amodal > 0) & (second_amodal > 0))
        )
        if overlaps:
            retained.append((first_id, second_id))
            log_event(
                logger,
                "amodal_overlap_validation",
                "pair_decision",
                first_id=first_id,
                second_id=second_id,
                decision="retain",
                reason="validated_amodal_masks_overlap",
            )
            continue

        logger.debug(
            "Validated amodal masks for %s and %s do not overlap; "
            "skipping depth ordering and reconstruction for this pair.",
            first_id,
            second_id,
        )
        log_event(
            logger,
            "amodal_overlap_validation",
            "pair_decision",
            first_id=first_id,
            second_id=second_id,
            decision="reject",
            reason="validated_amodal_masks_do_not_overlap",
        )

    return retained
