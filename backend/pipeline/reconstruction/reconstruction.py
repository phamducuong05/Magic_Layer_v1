"""Hidden-RGB reconstruction execution for occluded objects."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image

from ...core.logging import get_logger, log_event
from ..roi import SquareROI, crop_array, crop_image, square_roi_from_support
from ..types import DetectedObject
from .validate_reconstruction import (
    ReconstructionValidationError,
    reconstruction_color_metrics,
    validate_reconstruction_result,
)
from .artifacts import artifact_path


logger = get_logger(__name__)


@dataclass(frozen=True)
class _PreparedReconstruction:
    detected: DetectedObject
    roi: SquareROI
    source_crop: Image.Image
    mask_image: Image.Image
    mask_crop: np.ndarray
    accepted_rgb_crop: np.ndarray | None
    composition_crop: np.ndarray
    hole_crop: np.ndarray
    modal_crop: np.ndarray
    prompt: str


DEFAULT_PROMPT_TEMPLATE = (
    "Reconstruct only the hidden continuation of the {target} behind "
    "{occluders}, matching the visible target's color, texture, lighting, "
    "and geometry. Do not recreate {occluders}."
)


def _artifact_directory(root: str | Path, object_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", object_id)
    directory = Path(root) / safe_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _store_reconstruction_failure(
    detected: DetectedObject,
    *,
    stage: str,
    reason: str,
) -> None:
    """Clear only one invalid result and retain its mask diagnostics."""
    detected.raw_reconstruction_canvas = None
    detected.reconstruction_canvas = None
    detected.reconstruction_roi = None
    detected.reconstruction_write_alpha = None
    detected.reconstruction_write_mask = None
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
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    style_hint: str = "",
    diagnostics_directory: str | Path | None = None,
    debug_artifacts_provider: Callable[
        [], Sequence[dict[str, Image.Image]]
    ]
    | None = None,
) -> None:
    """Reconstruct and validate hidden RGB independently for each raw object."""
    prepared: list[_PreparedReconstruction] = []
    for detected in objects:
        detected.raw_reconstruction_canvas = None
        detected.reconstruction_canvas = None
        detected.reconstruction_roi = None
        detected.reconstruction_evidence_alpha = None
        detected.reconstruction_extension_mask = None
        detected.reconstruction_write_alpha = None
        detected.reconstruction_write_mask = None
        detected.reconstruction_support_mask = None
        detected.reconstruction_failure_stage = None
        detected.reconstruction_failure_reason = None
        reconstruction_mask = detected.reconstruction_mask
        generation_mask = (
            detected.reconstruction_generation_mask
            if detected.reconstruction_generation_mask is not None
            else reconstruction_mask
        )
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
            roi = detected.reconstruction_input_roi
            if roi is None:
                roi = square_roi_from_support(
                    (detected.amodal_mask > 0)
                    | reconstruction_mask.astype(bool),
                    context_ratio=context_ratio,
                )
            source_crop = crop_image(image.convert("RGB"), roi)
            mask_crop = crop_array(generation_mask.astype(bool), roi)
            accepted_rgb_mask = detected.reconstruction_accepted_rgb_mask
            accepted_rgb_crop = (
                crop_array(accepted_rgb_mask.astype(bool), roi)
                if accepted_rgb_mask is not None
                else None
            )
            composition_crop = crop_array(
                reconstruction_mask.astype(bool), roi
            )
            hole_crop = crop_array(
                detected.completion_hole_mask.astype(bool), roi
            )
            modal_crop = crop_array(detected.modal_mask > 0, roi)
            mask_image = Image.fromarray(
                mask_crop.astype(np.uint8) * 255,
                mode="L",
            )
            occluders = " and ".join(sorted(detected.occluder_classes)) or (
                "the foreground occluder"
            )
            prompt = prompt_template.format(
                target=detected.semantic_class,
                occluders=occluders,
            )
            if style_hint.strip():
                prompt = f"{prompt.rstrip()} {style_hint.strip()}"
            log_event(
                logger,
                "object_reconstruction",
                "input_prepared",
                object_id=detected.object_id,
                roi=(roi.x, roi.y, roi.size),
                crop_size=source_crop.size,
                hard_mask_pixels=int(np.count_nonzero(mask_crop)),
                composition_mask_pixels=int(
                    np.count_nonzero(composition_crop)
                ),
                prompt=prompt,
            )
            if diagnostics_directory is not None:
                directory = _artifact_directory(
                    diagnostics_directory, detected.object_id
                )
                source_crop.save(artifact_path(directory, "source"))
                _save_mask(
                    artifact_path(directory, "composition_mask"),
                    composition_crop,
                )
                _save_mask(
                    artifact_path(directory, "generation_mask"),
                    mask_crop,
                )
                real_pixels = np.zeros_like(mask_crop)
                inner_left, inner_top, inner_right, inner_bottom = (
                    roi.inner_box
                )
                real_pixels[
                    inner_top:inner_bottom,
                    inner_left:inner_right,
                ] = True
                diagnostic_accepted = (
                    accepted_rgb_crop
                    if accepted_rgb_crop is not None
                    else mask_crop
                )
                protected_mask = detected.reconstruction_protected_mask
                target_bbox_mask = (
                    detected.reconstruction_target_bbox_mask
                )
                completed_amodal = (
                    crop_array(detected.amodal_mask.astype(bool), roi)
                    if detected.amodal_mask is not None
                    else modal_crop
                )
                if target_bbox_mask is not None:
                    diagnostic_bbox = crop_array(
                        target_bbox_mask.astype(bool), roi
                    )
                else:
                    diagnostic_bbox = np.zeros_like(mask_crop)
                    bbox_y, bbox_x = np.nonzero(completed_amodal)
                    if bbox_x.size:
                        diagnostic_bbox[
                            int(bbox_y.min()) : int(bbox_y.max()) + 1,
                            int(bbox_x.min()) : int(bbox_x.max()) + 1,
                        ] = True
                _save_mask(
                    artifact_path(directory, "initial_modal_mask"),
                    modal_crop,
                )
                _save_mask(
                    artifact_path(directory, "completed_amodal_mask"),
                    completed_amodal,
                )
                _save_mask(
                    artifact_path(directory, "amodal_bbox_mask"),
                    diagnostic_bbox,
                )
                foreign_inside = (
                    detected.reconstruction_foreign_modal_inside_bbox
                )
                foreign_outside = (
                    detected.reconstruction_foreign_modal_outside_bbox
                )
                replacement_domain = (
                    detected.reconstruction_replacement_domain_mask
                )
                replaceable_foreign_inside = (
                    detected.reconstruction_replaceable_foreign_inside
                )
                protected_foreign_inside = (
                    detected.reconstruction_protected_foreign_inside
                )
                foreign_protection = (
                    detected.reconstruction_foreign_protection_mask
                )
                _save_mask(
                    artifact_path(directory, "accepted_model_rgb_mask"),
                    diagnostic_accepted,
                )
                _save_mask(
                    artifact_path(directory, "protected_source_pixels"),
                    crop_array(protected_mask.astype(bool), roi)
                    if protected_mask is not None
                    else real_pixels & ~diagnostic_accepted,
                )
                _save_mask(
                    artifact_path(directory, "roi_real_pixels"),
                    real_pixels,
                )
                _save_mask(
                    artifact_path(directory, "target_bbox_mask"),
                    diagnostic_bbox,
                )
                _save_mask(
                    artifact_path(directory, "target_modal_protected"),
                    modal_crop,
                )
                _save_mask(
                    artifact_path(directory, "foreign_modal_inside_bbox"),
                    crop_array(foreign_inside.astype(bool), roi)
                    if foreign_inside is not None
                    else np.zeros_like(mask_crop),
                )
                _save_mask(
                    artifact_path(directory, "foreign_modal_outside_bbox"),
                    crop_array(foreign_outside.astype(bool), roi)
                    if foreign_outside is not None
                    else np.zeros_like(mask_crop),
                )
                _save_mask(
                    artifact_path(directory, "replacement_domain"),
                    crop_array(replacement_domain.astype(bool), roi)
                    if replacement_domain is not None
                    else np.zeros_like(mask_crop),
                )
                _save_mask(
                    artifact_path(directory, "replaceable_foreign_inside"),
                    crop_array(
                        replaceable_foreign_inside.astype(bool), roi
                    )
                    if replaceable_foreign_inside is not None
                    else np.zeros_like(mask_crop),
                )
                _save_mask(
                    artifact_path(directory, "protected_foreign_inside"),
                    crop_array(protected_foreign_inside.astype(bool), roi)
                    if protected_foreign_inside is not None
                    else np.zeros_like(mask_crop),
                )
                _save_mask(
                    artifact_path(directory, "foreign_protection"),
                    crop_array(foreign_protection.astype(bool), roi)
                    if foreign_protection is not None
                    else np.zeros_like(mask_crop),
                )
                _save_mask(
                    artifact_path(directory, "completion_hole"),
                    hole_crop,
                )
                seed_mask = detected.reconstruction_seed_mask
                _save_mask(
                    artifact_path(directory, "directional_seed"),
                    crop_array(seed_mask.astype(bool), roi)
                    if seed_mask is not None
                    else np.zeros_like(mask_crop),
                )
                _save_mask(
                    artifact_path(directory, "filtered_reconstruction_mask"),
                    composition_crop,
                )
                generation_seed = (
                    detected.reconstruction_generation_seed_mask
                )
                _save_mask(
                    artifact_path(directory, "generation_before_dilation"),
                    crop_array(generation_seed.astype(bool), roi)
                    if generation_seed is not None
                    else composition_crop,
                )
                _save_mask(
                    artifact_path(directory, "generation_after_dilation"),
                    mask_crop,
                )
                occluder_mask = detected.reconstruction_occluder_mask
                _save_mask(
                    artifact_path(directory, "occluder_mask"),
                    crop_array(
                        occluder_mask.astype(bool), roi
                    ) if occluder_mask is not None else np.zeros_like(mask_crop),
                )
                _save_mask(
                    artifact_path(directory, "full_occluder_in_roi"),
                    crop_array(occluder_mask.astype(bool), roi)
                    if occluder_mask is not None
                    else np.zeros_like(mask_crop),
                )
            prepared.append(
                _PreparedReconstruction(
                    detected=detected,
                    roi=roi,
                    source_crop=source_crop,
                    mask_image=mask_image,
                    mask_crop=mask_crop,
                    accepted_rgb_crop=accepted_rgb_crop,
                    composition_crop=composition_crop,
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

    debug_artifacts: Sequence[dict[str, Image.Image]] = ()
    if diagnostics_directory is not None and debug_artifacts_provider:
        try:
            debug_artifacts = debug_artifacts_provider()
        except Exception as exc:
            logger.warning(
                "Could not collect HD-Painter debug artifacts: %s", exc
            )

    for index, (item, reconstructed) in enumerate(zip(prepared, outcomes)):
        detected = item.detected
        if diagnostics_directory is not None and index < len(debug_artifacts):
            directory = _artifact_directory(
                diagnostics_directory, detected.object_id
            )
            for artifact_name in ("base_output_512", "sr_output"):
                artifact = debug_artifacts[index].get(artifact_name)
                if isinstance(artifact, Image.Image):
                    artifact.save(artifact_path(directory, artifact_name))
        if isinstance(reconstructed, Exception):
            _store_reconstruction_failure(
                detected,
                stage=str(getattr(reconstructed, "stage", "inference")),
                reason=str(reconstructed) or type(reconstructed).__name__,
            )
            continue
        try:
            if diagnostics_directory is not None and isinstance(
                reconstructed, Image.Image
            ):
                directory = _artifact_directory(
                    diagnostics_directory, detected.object_id
                )
                reconstructed.save(artifact_path(directory, "model_output"))
            validated = validate_reconstruction_result(
                reconstructed,
                source_crop=item.source_crop,
                hard_mask=item.mask_crop,
                accepted_model_rgb_mask=item.accepted_rgb_crop,
                completion_hole=item.composition_crop,
                modal_mask=item.modal_crop,
                roi=item.roi,
                blend_allowance_ratio=blend_allowance_ratio,
            )
            raw_reconstructed = reconstructed.convert("RGB")
            occluder_crop = crop_array(
                detected.reconstruction_occluder_mask.astype(bool),
                item.roi,
            ) if detected.reconstruction_occluder_mask is not None else (
                np.zeros_like(item.mask_crop)
            )
            color_metrics = reconstruction_color_metrics(
                validated,
                source_crop=item.source_crop,
                composition_mask=(
                    item.accepted_rgb_crop
                    if item.accepted_rgb_crop is not None
                    else item.composition_crop
                ),
                modal_mask=item.modal_crop,
                occluder_mask=occluder_crop,
            )
            log_event(
                logger,
                "object_reconstruction",
                "color_assessment",
                object_id=detected.object_id,
                **color_metrics,
            )
        except Exception as exc:
            _store_reconstruction_failure(
                detected,
                stage=str(getattr(exc, "stage", "validation")),
                reason=str(exc) or type(exc).__name__,
            )
            continue

        detected.raw_reconstruction_canvas = raw_reconstructed
        detected.reconstruction_canvas = validated
        detected.reconstruction_roi = item.roi
        if detected.reconstruction_accepted_rgb_mask is not None:
            detected.reconstruction_write_mask = (
                detected.reconstruction_accepted_rgb_mask.astype(bool).copy()
            )
            detected.reconstruction_write_alpha = (
                detected.reconstruction_write_mask.astype(np.float64)
            )
        if diagnostics_directory is not None:
            directory = _artifact_directory(
                diagnostics_directory, detected.object_id
            )
            validated.save(artifact_path(directory, "validated_output"))
        log_event(
            logger,
            "object_reconstruction",
            "object_decision",
            object_id=detected.object_id,
            decision="accepted",
            roi=(item.roi.x, item.roi.y, item.roi.size),
            output_size=validated.size,
        )

    accepted_count = sum(
        detected.reconstruction_canvas is not None for detected in objects
    )
    failed_count = sum(
        detected.reconstruction_failure_stage is not None for detected in objects
    )
    log_event(
        logger,
        "object_reconstruction",
        "summary",
        level="INFO",
        requested=len(prepared),
        accepted=accepted_count,
        failed=failed_count,
        bypassed=len(objects) - len(prepared),
        artifacts=diagnostics_directory,
    )
