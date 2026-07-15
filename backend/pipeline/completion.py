"""Cross-class overlap linking and explicit-model amodal completion."""

from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..core.occlusion import (
    ObjectBounds,
    OverlapPair,
    find_cross_class_overlaps,
)
from .types import DetectedObject


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
                bbox=detected.bbox,
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
) -> None:
    """Complete the supplied candidates with the supplied model adapter."""
    if not candidates:
        return

    amodal_masks = completion_model.complete(
        image,
        [detected.modal_mask for detected in candidates],
        [detected.bbox for detected in candidates],
    )
    for detected, amodal_mask in zip(candidates, amodal_masks):
        detected.amodal_mask = amodal_mask
        detected.completion_hole_mask = (amodal_mask > 0) & (
            detected.modal_mask == 0
        )
        detected.completion_hole_area = int(
            np.count_nonzero(detected.completion_hole_mask)
        )
