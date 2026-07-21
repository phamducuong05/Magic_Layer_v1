"""Cross-class overlap linking and explicit-model amodal completion."""

from collections.abc import Sequence
import logging
from typing import Any

import numpy as np
from PIL import Image

from ...core.occlusion import (
    ObjectBounds,
    OverlapPair,
    find_cross_class_overlaps,
)
from ..types import DetectedObject
from .validate_completion import (
    _store_modal_fallback,
    _store_valid_completion,
    _validated_amodal_mask,
)


logger = logging.getLogger(__name__)


def link_overlap_partners(
    objects: Sequence[DetectedObject],
) -> list[OverlapPair]:
    """Record positive-area cross-class box overlaps on both objects."""
    for detected in objects:
        detected.overlap_partner_ids.clear()

    pairs = find_cross_class_overlaps(
        [
            ObjectBounds(
                object_id=detected.object_id,
                semantic_class=detected.semantic_class,
                bbox=detected.original_modal_bbox,
            )
            for detected in objects
        ]
    )
    objects_by_id = {detected.object_id: detected for detected in objects}
    for first_id, second_id in pairs:
        objects_by_id[first_id].overlap_partner_ids.add(second_id)
        objects_by_id[second_id].overlap_partner_ids.add(first_id)

    return pairs


def get_completion_candidates(
    objects: Sequence[DetectedObject],
) -> list[DetectedObject]:
    """Return overlapping objects once each, preserving object order."""
    return [detected for detected in objects if detected.overlap_partner_ids]


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

    try:
        outputs = completion_model.complete(
            image,
            [detected.modal_mask for detected in candidates],
            [detected.bbox for detected in candidates],
        )
    except Exception:
        logger.warning(
            "Shared completion inference failed; using modal fallback.",
            exc_info=True,
        )
        for detected in candidates:
            _store_modal_fallback(detected)
        return

    if not isinstance(outputs, Sequence) or len(outputs) != len(candidates):
        logger.warning(
            "Completion returned %s outputs for %d candidates; using modal "
            "fallback for the batch.",
            len(outputs) if isinstance(outputs, Sequence) else "non-sequence",
            len(candidates),
        )
        for detected in candidates:
            _store_modal_fallback(detected)
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
            continue
        _store_valid_completion(detected, amodal_mask)


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
            continue

        logger.info(
            "Validated amodal masks for %s and %s do not overlap; "
            "skipping depth ordering and reconstruction for this pair.",
            first_id,
            second_id,
        )

    return retained
