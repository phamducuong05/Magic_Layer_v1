"""Hidden-RGB reconstruction execution for occluded objects."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ...core.logging import get_logger, log_event
from ..roi import SquareROI, crop_array, crop_image, square_roi_from_support
from ..types import DetectedObject
from .validate_reconstruction import (
    ReconstructionValidationError,
    validate_reconstruction_result,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class _PreparedReconstruction:
    detected: DetectedObject
    roi: SquareROI
    source_crop: Image.Image
    mask_image: Image.Image
    mask_crop: np.ndarray
    hole_crop: np.ndarray
    modal_crop: np.ndarray
    prompt: str


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
    log_event(
        logger,
        "object_reconstruction",
        "object_decision",
        object_id=detected.object_id,
        decision="modal_rgb_fallback",
        failure_stage=stage,
        reason=reason,
    )


def reconstruct_objects(
    image: Image.Image,
    objects: Sequence[DetectedObject],
    reconstruct: Callable[[Image.Image, Image.Image, str], Image.Image],
    *,
    reconstruct_many: Callable[
        [Sequence[tuple[Image.Image, Image.Image, str]]],
        Sequence[Image.Image | Exception],
    ]
    | None = None,
    context_ratio: float,
    blend_allowance_ratio: float = 0.0,
) -> None:
    """Reconstruct and validate hidden RGB independently for each raw object."""
    prepared: list[_PreparedReconstruction] = []
    for detected in objects:
        detected.reconstruction_canvas = None
        detected.reconstruction_roi = None
        detected.reconstruction_failure_stage = None
        detected.reconstruction_failure_reason = None
        reconstruction_mask = detected.reconstruction_mask
        if reconstruction_mask is None or not np.any(reconstruction_mask):
            log_event(
                logger,
                "object_reconstruction",
                "object_decision",
                object_id=detected.object_id,
                decision="bypass",
                reason="no_reconstruction_mask",
            )
            continue
        log_event(
            logger,
            "object_reconstruction",
            "object_decision",
            object_id=detected.object_id,
            decision="run",
            reconstruction_pixels=int(np.count_nonzero(reconstruction_mask)),
        )
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
            log_event(
                logger,
                "object_reconstruction",
                "input_prepared",
                object_id=detected.object_id,
                roi=(roi.x, roi.y, roi.size),
                crop_size=source_crop.size,
                hard_mask_pixels=int(np.count_nonzero(mask_crop)),
            )
            prepared.append(
                _PreparedReconstruction(
                    detected=detected,
                    roi=roi,
                    source_crop=source_crop,
                    mask_image=mask_image,
                    mask_crop=mask_crop,
                    hole_crop=hole_crop,
                    modal_crop=modal_crop,
                    prompt=prompt,
                )
            )
        except Exception as exc:
            _store_reconstruction_failure(
                detected,
                stage=str(getattr(exc, "stage", "inference")),
                reason=str(exc) or type(exc).__name__,
            )
            continue

    if reconstruct_many is None:
        outcomes: Sequence[Image.Image | Exception] = []
        single_outcomes: list[Image.Image | Exception] = []
        for item in prepared:
            try:
                single_outcomes.append(
                    reconstruct(
                        item.source_crop,
                        item.mask_image,
                        item.prompt,
                    )
                )
            except Exception as exc:
                single_outcomes.append(exc)
        outcomes = single_outcomes
    else:
        try:
            outcomes = reconstruct_many(
                [
                    (item.source_crop, item.mask_image, item.prompt)
                    for item in prepared
                ]
            )
        except Exception as exc:
            outcomes = [exc] * len(prepared)

    if len(outcomes) != len(prepared):
        mismatch = ReconstructionValidationError(
            "inference",
            "batch reconstruction returned an unexpected result count",
        )
        outcomes = [mismatch] * len(prepared)

    for item, reconstructed in zip(prepared, outcomes):
        detected = item.detected
        if isinstance(reconstructed, Exception):
            _store_reconstruction_failure(
                detected,
                stage=str(getattr(reconstructed, "stage", "inference")),
                reason=str(reconstructed) or type(reconstructed).__name__,
            )
            continue
        try:
            validated = validate_reconstruction_result(
                reconstructed,
                source_crop=item.source_crop,
                hard_mask=item.mask_crop,
                completion_hole=item.hole_crop,
                modal_mask=item.modal_crop,
                roi=item.roi,
                blend_allowance_ratio=blend_allowance_ratio,
            )
        except Exception as exc:
            _store_reconstruction_failure(
                detected,
                stage=str(getattr(exc, "stage", "validation")),
                reason=str(exc) or type(exc).__name__,
            )
            continue

        detected.reconstruction_canvas = validated
        detected.reconstruction_roi = item.roi
        log_event(
            logger,
            "object_reconstruction",
            "object_decision",
            object_id=detected.object_id,
            decision="accepted",
            roi=(item.roi.x, item.roi.y, item.roi.size),
            output_size=validated.size,
        )
