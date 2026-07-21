"""Hidden-RGB reconstruction execution for occluded objects."""

from collections.abc import Callable, Sequence
import logging

import numpy as np
from PIL import Image

from ..roi import SquareROI, crop_array, crop_image, square_roi_from_support
from ..types import DetectedObject
from .validate_reconstruction import (
    ReconstructionValidationError,
    validate_reconstruction_result,
)


logger = logging.getLogger(__name__)


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
            validated = validate_reconstruction_result(
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
