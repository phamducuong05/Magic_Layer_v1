"""Validation helpers for amodal completion outputs."""

import logging
from typing import Any

import numpy as np

from ...core.helpers import _bbox_area
from ...core.layerd_refine import divide_mask_to_connected_components
from ..types import DetectedObject


logger = logging.getLogger(__name__)


def _store_modal_fallback(detected: DetectedObject) -> None:
    """Store a safe completed state that cannot create a reconstruction hole."""
    modal_mask = detected.modal_mask > 0
    detected.amodal_mask = modal_mask
    detected.completion_hole_mask = np.zeros_like(modal_mask, dtype=bool)
    detected.completion_hole_area = 0


def _validated_amodal_mask(
    detected: DetectedObject,
    output: Any,
    *,
    max_area_growth_ratio: float,
    max_bbox_growth_ratio: float,
) -> np.ndarray | None:
    """Return a canonical mask when one model output passes all limits."""
    label = detected.display_label
    if not isinstance(output, np.ndarray):
        logger.warning(
            "[%s] Amodal output rejected: not a numpy array (got %s)",
            label, type(output).__name__,
        )
        return None
    if not (
        np.issubdtype(output.dtype, np.bool_)
        or np.issubdtype(output.dtype, np.integer)
        or np.issubdtype(output.dtype, np.floating)
    ):
        logger.warning(
            "[%s] Amodal output rejected: unsupported dtype %s",
            label, output.dtype,
        )
        return None
    if output.shape != detected.modal_mask.shape:
        logger.warning(
            "[%s] Amodal output rejected: shape mismatch "
            "(output=%s, modal_mask=%s)",
            label, output.shape, detected.modal_mask.shape,
        )
        return None
    if not np.all(np.isfinite(output)):
        logger.warning(
            "[%s] Amodal output rejected: contains NaN or Inf values",
            label,
        )
        return None

    amodal_mask = output > 0
    modal_mask = detected.modal_mask > 0

    # Filter disconnected components
    # Split the amodal mask into connected components and discard any
    # component that does not overlap with the original modal mask.
    components = divide_mask_to_connected_components(amodal_mask)
    if components:
        valid_components = [
            comp for comp in components
            if np.any(comp & modal_mask)
        ]
        discarded_count = len(components) - len(valid_components)
        if discarded_count > 0:
            logger.info(
                "[%s] Discarded %d/%d disconnected amodal components "
                "(not overlapping with modal mask)",
                label, discarded_count, len(components),
            )
        if not valid_components:
            logger.warning(
                "[%s] Amodal output rejected: no connected component "
                "overlaps with the modal mask",
                label,
            )
            return None
        # Rebuild amodal mask from valid components only
        amodal_mask = np.zeros_like(amodal_mask, dtype=bool)
        for comp in valid_components:
            amodal_mask |= comp

    modal_area = int(np.count_nonzero(modal_mask))
    amodal_area = int(np.count_nonzero(amodal_mask))
    if modal_area == 0 or amodal_area == 0:
        logger.warning(
            "[%s] Amodal output rejected: empty mask "
            "(modal_area=%d, amodal_area=%d)",
            label, modal_area, amodal_area,
        )
        return None
    if not np.all(amodal_mask[modal_mask]):
        missing_pixels = int(np.count_nonzero(modal_mask & ~amodal_mask))
        logger.warning(
            "[%s] Amodal output rejected: amodal mask does not fully cover "
            "the modal mask (%d modal pixels missing)",
            label, missing_pixels,
        )
        return None
    area_ratio = amodal_area / modal_area
    if area_ratio > max_area_growth_ratio:
        logger.warning(
            "[%s] Amodal output rejected: area growth ratio %.2f exceeds "
            "limit %.2f (modal=%d, amodal=%d)",
            label, area_ratio, max_area_growth_ratio, modal_area, amodal_area,
        )
        return None

    modal_bbox_area = _bbox_area(modal_mask)
    amodal_bbox_area = _bbox_area(amodal_mask)
    if modal_bbox_area == 0:
        logger.warning(
            "[%s] Amodal output rejected: modal bounding box area is 0",
            label,
        )
        return None
    bbox_ratio = amodal_bbox_area / modal_bbox_area
    if bbox_ratio > max_bbox_growth_ratio:
        logger.warning(
            "[%s] Amodal output rejected: bbox growth ratio %.2f exceeds "
            "limit %.2f (modal_bbox=%d, amodal_bbox=%d)",
            label, bbox_ratio, max_bbox_growth_ratio,
            modal_bbox_area, amodal_bbox_area,
        )
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
