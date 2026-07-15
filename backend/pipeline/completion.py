"""Cross-class overlap linking and explicit-model amodal completion."""

from collections.abc import Sequence
import logging
from typing import Any

import numpy as np
from PIL import Image

from ..core.occlusion import (
    ObjectBounds,
    OverlapPair,
    find_cross_class_overlaps,
)
from .types import DetectedObject


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


def _store_modal_fallback(detected: DetectedObject) -> None:
    """Store a safe completed state that cannot create a reconstruction hole."""
    modal_mask = detected.modal_mask > 0
    detected.amodal_mask = modal_mask
    detected.completion_hole_mask = np.zeros_like(modal_mask, dtype=bool)
    detected.completion_hole_area = 0


def _bbox_area(mask: np.ndarray) -> int:
    """Return the tight positive-pixel bounding-box area of a binary mask."""
    positive_y, positive_x = np.nonzero(mask)
    if positive_x.size == 0:
        return 0
    width = int(positive_x.max() - positive_x.min() + 1)
    height = int(positive_y.max() - positive_y.min() + 1)
    return width * height


def _validated_amodal_mask(
    detected: DetectedObject,
    output: Any,
    *,
    max_area_growth_ratio: float,
    max_bbox_growth_ratio: float,
) -> np.ndarray | None:
    """Return a canonical mask when one model output passes all limits."""
    if not isinstance(output, np.ndarray):
        return None
    if not (
        np.issubdtype(output.dtype, np.bool_)
        or np.issubdtype(output.dtype, np.integer)
        or np.issubdtype(output.dtype, np.floating)
    ):
        return None
    if output.shape != detected.modal_mask.shape:
        return None
    if not np.all(np.isfinite(output)):
        return None

    amodal_mask = output > 0
    modal_mask = detected.modal_mask > 0
    modal_area = int(np.count_nonzero(modal_mask))
    amodal_area = int(np.count_nonzero(amodal_mask))
    if modal_area == 0 or amodal_area == 0:
        return None
    if not np.all(amodal_mask[modal_mask]):
        return None
    if amodal_area / modal_area > max_area_growth_ratio:
        return None

    modal_bbox_area = _bbox_area(modal_mask)
    amodal_bbox_area = _bbox_area(amodal_mask)
    if (
        modal_bbox_area == 0
        or amodal_bbox_area / modal_bbox_area > max_bbox_growth_ratio
    ):
        return None
    return amodal_mask


def _store_valid_completion(
    detected: DetectedObject, amodal_mask: np.ndarray
) -> None:
    detected.amodal_mask = amodal_mask
    detected.completion_hole_mask = amodal_mask & ~(
        detected.modal_mask > 0
    )
    detected.completion_hole_area = int(
        np.count_nonzero(detected.completion_hole_mask)
    )


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
