"""Explicit-callable hard-mask to soft-alpha refinement."""

from collections.abc import Callable, Sequence
from pathlib import Path
import math
import re

import cv2
import numpy as np
import torch
from PIL import Image

from ..core.helpers import _inference_context
from ..core.logging import get_logger, log_event
from .roi import (
    SquareROI,
    crop_array,
    crop_image,
    restore_array,
    square_roi_from_support,
)
from .types import DetectedObject, GroupedObject
from .reconstruction.artifacts import artifact_path

THRESHOLD_ALPHA = 0.005
logger = get_logger(__name__)


def _reconstruction_artifact_directory(
    root: str | Path, object_id: str
) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", object_id)
    directory = Path(root) / safe_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _save_alpha(path: Path, alpha: np.ndarray) -> None:
    Image.fromarray(
        np.rint(np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8),
        mode="L",
    ).save(path)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool).copy()
    kernel = np.ones((2 * radius + 1,) * 2, dtype=np.uint8)
    return cv2.dilate(
        mask.astype(np.uint8), kernel, iterations=1
    ).astype(bool)


def _resize_alpha(alpha: torch.Tensor, size: int) -> np.ndarray:
    if not isinstance(alpha, torch.Tensor) or alpha.ndim != 2:
        raise ValueError("matting output must be a two-dimensional tensor")
    if tuple(alpha.shape) != (size, size):
        alpha = torch.nn.functional.interpolate(
            alpha[None, None].to(torch.float32),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return np.clip(
        alpha.detach().cpu().to(torch.float64).numpy(),
        0.0,
        1.0,
    )


def refine_reconstruction_supports(
    image: Image.Image,
    objects: Sequence[DetectedObject],
    matte: Callable[[Image.Image], torch.Tensor],
    *,
    alpha_low_threshold: float,
    alpha_high_threshold: float,
    change_threshold: float,
    connection_margin_pixels: int,
    max_extension_area_ratio: float,
    min_component_area_pixels: int = 0,
    generation_evidence_margin_pixels: int = 0,
    require_generation_evidence: bool = True,
    alpha_write_epsilon: float = 0.01,
    alpha_feather_pixels: int = 0,
    fallback_to_validated_output: bool = True,
    diagnostics_directory: str | Path | None = None,
) -> None:
    """Extract target extension directly from raw HD-Painter RGB."""
    if not 0.0 <= alpha_low_threshold <= alpha_high_threshold <= 1.0:
        raise ValueError(
            "support alpha thresholds must satisfy 0 <= low <= high <= 1"
        )
    if not math.isfinite(change_threshold) or change_threshold < 0:
        raise ValueError("support change threshold must be non-negative")
    if connection_margin_pixels < 0:
        raise ValueError("support connection margin must be non-negative")
    if min_component_area_pixels < 0:
        raise ValueError("support minimum component area must be non-negative")
    if generation_evidence_margin_pixels < 0:
        raise ValueError(
            "support generation evidence margin must be non-negative"
        )
    if not math.isfinite(alpha_write_epsilon) or not (
        0.0 <= alpha_write_epsilon <= 1.0
    ):
        raise ValueError("support alpha write epsilon must be in [0, 1]")
    if alpha_feather_pixels < 0:
        raise ValueError("support alpha feather radius must be non-negative")
    if (
        not math.isfinite(max_extension_area_ratio)
        or max_extension_area_ratio < 0
    ):
        raise ValueError("support maximum extension ratio must be non-negative")

    source = image.convert("RGB")
    for target in objects:
        target.reconstruction_evidence_alpha = None
        target.reconstruction_extension_mask = None
        target.reconstruction_support_mask = None

        reconstruction_mask = target.reconstruction_mask
        raw_canvas = (
            target.raw_reconstruction_canvas
            if target.raw_reconstruction_canvas is not None
            else target.reconstruction_canvas
        )
        roi = target.reconstruction_roi
        if (
            raw_canvas is None
            or roi is None
            or reconstruction_mask is None
            or not np.any(reconstruction_mask)
        ):
            continue

        modal = target.modal_mask > 0
        fallback_write_mask = (
            target.reconstruction_accepted_rgb_mask.astype(bool)
            if target.reconstruction_accepted_rgb_mask is not None
            else reconstruction_mask.astype(bool)
        )
        target.reconstruction_write_mask = fallback_write_mask.copy()
        target.reconstruction_write_alpha = fallback_write_mask.astype(
            np.float64
        )
        target.reconstruction_support_mask = (
            modal | reconstruction_mask.astype(bool)
            if fallback_to_validated_output
            else modal.copy()
        )

        if (
            roi.image_size != source.size
            or raw_canvas.size != (roi.size, roi.size)
        ):
            if not fallback_to_validated_output:
                target.reconstruction_canvas = None
                target.reconstruction_roi = None
                target.reconstruction_write_alpha = np.zeros_like(
                    modal, dtype=np.float64
                )
                target.reconstruction_write_mask = np.zeros_like(modal)
                target.reconstruction_support_mask = modal.copy()
            log_event(
                logger,
                "reconstruction_support",
                "object_decision",
                object_id=target.object_id,
                decision="fallback",
                reason="invalid_reconstruction_roi_or_canvas",
            )
            continue

        source_crop = np.asarray(crop_image(source, roi), dtype=np.uint8)
        raw_reconstructed_crop = np.asarray(
            raw_canvas.convert("RGB"), dtype=np.uint8
        )
        modal_crop = crop_array(modal, roi).astype(bool)
        valid_roi_crop = crop_array(
            np.ones_like(modal, dtype=bool), roi
        ).astype(bool)
        if target.reconstruction_accepted_target_bbox is not None:
            accepted_bbox = crop_array(
                target.reconstruction_accepted_target_bbox.astype(bool),
                roi,
            ).astype(bool)
        elif target.reconstruction_accepted_rgb_mask is not None:
            accepted_bbox = crop_array(
                target.reconstruction_accepted_rgb_mask.astype(bool),
                roi,
            ).astype(bool)
        else:
            accepted_bbox = valid_roi_crop.copy()
        accepted_bbox &= valid_roi_crop

        masked_input = np.zeros_like(raw_reconstructed_crop)
        masked_input[accepted_bbox] = raw_reconstructed_crop[accepted_bbox]

        try:
            alpha_crop = _resize_alpha(
                matte(Image.fromarray(masked_input, mode="RGB")),
                roi.size,
            )
        except Exception as exc:
            logger.warning(
                "Could not refine reconstruction support for %s: %s",
                target.object_id,
                exc,
            )
            log_event(
                logger,
                "reconstruction_support",
                "object_decision",
                object_id=target.object_id,
                decision="fallback",
                reason="matting_failed",
            )
            if not fallback_to_validated_output:
                target.reconstruction_canvas = None
                target.reconstruction_roi = None
                target.reconstruction_write_alpha = np.zeros_like(
                    modal, dtype=np.float64
                )
                target.reconstruction_write_mask = np.zeros_like(modal)
                target.reconstruction_support_mask = modal.copy()
            continue

        if target.reconstruction_foreign_protection_mask is not None:
            foreign_protection = (
                target.reconstruction_foreign_protection_mask.astype(bool)
            )
        elif target.reconstruction_protected_mask is not None:
            foreign_protection = (
                target.reconstruction_protected_mask.astype(bool) & ~modal
            )
        else:
            foreign_protection = np.zeros_like(modal)
        foreign_protection_crop = crop_array(
            foreign_protection, roi
        ).astype(bool)

        generation_mask = (
            target.reconstruction_generation_mask.astype(bool)
            if target.reconstruction_generation_mask is not None
            else reconstruction_mask.astype(bool)
        )
        generation_evidence = _dilate_mask(
            crop_array(generation_mask, roi).astype(bool),
            generation_evidence_margin_pixels,
        )
        target_domain = generation_evidence | modal_crop

        connection_anchor = _dilate_mask(
            modal_crop, connection_margin_pixels
        )

        candidate = (
            (alpha_crop >= alpha_low_threshold)
            & valid_roi_crop
            & accepted_bbox
        )
        strong_foreground = alpha_crop >= alpha_high_threshold
        change_distance = np.mean(
            np.abs(
                raw_reconstructed_crop.astype(np.int16)
                - source_crop.astype(np.int16)
            ),
            axis=2,
        )
        changed_by_model = change_distance >= change_threshold

        accepted = np.zeros_like(candidate)
        component_count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                candidate.astype(np.uint8), connectivity=8
            )
        )
        modal_area = int(np.count_nonzero(modal))
        max_extension_area = int(
            math.floor(modal_area * max_extension_area_ratio)
        )
        rejected_area_components = 0
        for label in range(1, component_count):
            component = labels == label
            extension_part = component & ~modal_crop
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            extension_area = int(np.count_nonzero(extension_part))
            if component_area < min_component_area_pixels:
                continue
            if not np.any(component & target_domain):
                continue
            if not np.any(component & connection_anchor):
                continue
            if not np.any(component & strong_foreground):
                continue
            if not np.any(extension_part & changed_by_model):
                continue
            if (
                require_generation_evidence
                and not np.any(extension_part & generation_evidence)
            ):
                continue
            if extension_area > max_extension_area:
                rejected_area_components += 1
                continue
            accepted |= component

        extension_support = (
            accepted
            & ~modal_crop
            & target_domain
            & accepted_bbox
            & valid_roi_crop
        )
        extension_alpha = (
            alpha_crop * extension_support.astype(np.float64)
        )
        if alpha_feather_pixels and np.any(extension_alpha):
            kernel_size = 2 * alpha_feather_pixels + 1
            feather_domain = _dilate_mask(
                extension_support, alpha_feather_pixels
            )
            extension_alpha = cv2.GaussianBlur(
                extension_alpha,
                (kernel_size, kernel_size),
                sigmaX=0,
            )
            extension_alpha *= feather_domain
        extension_alpha[modal_crop] = 0.0
        extension_alpha[~target_domain] = 0.0
        extension_alpha[~accepted_bbox] = 0.0
        extension_alpha[~valid_roi_crop] = 0.0
        write_crop = extension_alpha > alpha_write_epsilon

        alpha_full = restore_array(alpha_crop, roi)
        write_alpha_full = restore_array(extension_alpha, roi)
        extension_full = restore_array(write_crop, roi).astype(bool)
        target.reconstruction_evidence_alpha = alpha_full
        target.reconstruction_write_alpha = write_alpha_full
        target.reconstruction_write_mask = extension_full.copy()
        target.reconstruction_extension_mask = extension_full
        target.reconstruction_support_mask = modal | extension_full

        composed_crop = np.rint(
            raw_reconstructed_crop.astype(np.float64)
            * extension_alpha[..., None]
            + source_crop.astype(np.float64)
            * (1.0 - extension_alpha[..., None])
        ).clip(0, 255).astype(np.uint8)
        if np.any(extension_full):
            # Group composition performs the soft-alpha blend exactly once.
            target.reconstruction_canvas = raw_canvas.convert("RGB")
        else:
            if fallback_to_validated_output and target.reconstruction_canvas is not None:
                pass  # Giữ nguyên validated canvas đã lưu trước đó
            else:
                target.reconstruction_canvas = None
                target.reconstruction_roi = None

        if diagnostics_directory is not None:
            directory = _reconstruction_artifact_directory(
                diagnostics_directory, target.object_id
            )
            Image.fromarray(masked_input, mode="RGB").save(
                artifact_path(directory, "birefnet_masked_input")
            )
            _save_alpha(
                artifact_path(directory, "raw_birefnet_alpha"),
                alpha_crop,
            )
            _save_mask(
                artifact_path(directory, "target_connection_anchor"),
                connection_anchor,
            )
            _save_mask(
                artifact_path(directory, "birefnet_candidate"),
                candidate,
            )
            _save_mask(
                artifact_path(directory, "changed_by_model"),
                changed_by_model,
            )
            _save_mask(
                artifact_path(directory, "generation_evidence"),
                generation_evidence,
            )
            _save_mask(
                artifact_path(directory, "accepted_target_component"),
                accepted,
            )
            _save_alpha(
                artifact_path(directory, "reconstruction_extension_alpha"),
                extension_alpha,
            )
            _save_mask(
                artifact_path(directory, "reconstruction_extension_mask"),
                write_crop,
            )
            _save_alpha(
                artifact_path(directory, "reconstruction_write_alpha"),
                extension_alpha,
            )
            _save_mask(
                artifact_path(directory, "reconstruction_write_mask"),
                crop_array(target.reconstruction_write_mask, roi),
            )
            Image.fromarray(
                composed_crop, mode="RGB"
            ).save(artifact_path(directory, "composed_reconstruction"))
            _save_mask(
                artifact_path(directory, "final_reconstruction_support"),
                crop_array(target.reconstruction_support_mask, roi),
            )

        log_event(
            logger,
            "reconstruction_support",
            "object_decision",
            object_id=target.object_id,
            decision=(
                "extended" if np.any(extension_full) else "keep_modal_prior"
            ),
            alpha_candidate_pixels=int(np.count_nonzero(candidate)),
            protected_pixels_excluded=int(
                np.count_nonzero(foreign_protection_crop)
            ),
            extension_pixels=int(np.count_nonzero(extension_full)),
            max_extension_pixels=max_extension_area,
            rejected_area_components=rejected_area_components,
            write_pixels=int(
                np.count_nonzero(target.reconstruction_write_mask)
            ),
            support_pixels=int(
                np.count_nonzero(target.reconstruction_support_mask)
            ),
        )


def _expand_composed_source(
    source_image: Image.Image,
    composed_source: Image.Image,
    composed_roi: SquareROI,
    matting_roi: SquareROI,
) -> Image.Image:
    """Add original context around an authoritative composed group crop."""
    expanded = np.asarray(
        crop_image(source_image, matting_roi), dtype=np.uint8
    ).copy()
    composed = np.asarray(composed_source.convert("RGB"), dtype=np.uint8)

    left = max(0, composed_roi.x, matting_roi.x)
    top = max(0, composed_roi.y, matting_roi.y)
    right = min(
        source_image.width,
        composed_roi.x + composed_roi.size,
        matting_roi.x + matting_roi.size,
    )
    bottom = min(
        source_image.height,
        composed_roi.y + composed_roi.size,
        matting_roi.y + matting_roi.size,
    )
    if left < right and top < bottom:
        expanded[
            top - matting_roi.y : bottom - matting_roi.y,
            left - matting_roi.x : right - matting_roi.x,
        ] = composed[
            top - composed_roi.y : bottom - composed_roi.y,
            left - composed_roi.x : right - composed_roi.x,
        ]
    return Image.fromarray(expanded, mode="RGB")


def refine_masks(
    image_np: np.ndarray,
    raw_masks: Sequence[np.ndarray],
    matte: Callable[[Image.Image], torch.Tensor],
) -> list[np.ndarray]:
    """Convert hard masks into soft alpha mattes using the supplied callable."""
    soft_alphas: list[np.ndarray] = []

    with torch.inference_mode(), _inference_context():
        for mask in raw_masks:
            guided_image = image_np.copy()
            guided_image[mask == 0] = 0
            alpha = matte(Image.fromarray(guided_image))

            support = cv2.dilate(
                mask, np.ones((5, 5), dtype=np.uint8), iterations=1
            ) > 0
            valid = (alpha > THRESHOLD_ALPHA) & torch.from_numpy(support).to(
                alpha.device
            )
            alpha[~valid] = 0.0
            soft_alphas.append(
                np.clip(alpha.cpu().to(torch.float64).numpy(), 0.0, 1.0)
            )

    return soft_alphas


def refine_objects(
    image: Image.Image,
    objects: Sequence[GroupedObject],
    matte: Callable[[Image.Image], torch.Tensor],
    *,
    context_ratio: float,
    support_dilation_pixels: int,
) -> None:
    """Matte final groups from aligned original or reconstructed RGB crops."""
    if support_dilation_pixels < 0:
        raise ValueError("support_dilation_pixels must be non-negative")

    source_image = image.convert("RGB")
    dilation_size = 2 * support_dilation_pixels + 1
    dilation_kernel = np.ones(
        (dilation_size, dilation_size), dtype=np.uint8
    )
    with torch.inference_mode(), _inference_context():
        for group in objects:
            if group.has_reconstruction:
                if group.composed_source is None or group.composed_roi is None:
                    raise ValueError(
                        f"reconstructed group {group.group_id} has no "
                        "composed source"
                    )
                support = group.effective_support_mask
                roi = square_roi_from_support(
                    support,
                    context_ratio=context_ratio,
                )
                source_crop = _expand_composed_source(
                    source_image,
                    group.composed_source,
                    group.composed_roi,
                    roi,
                )
                source_kind = "composed_reconstructed_rgb"
            else:
                support = group.modal_mask > 0
                roi = square_roi_from_support(
                    support,
                    context_ratio=context_ratio,
                )
                source_crop = crop_image(source_image, roi)
                source_kind = "original_modal_rgb"

            log_event(
                logger,
                "matting",
                "group_decision",
                group_id=group.group_id,
                decision="run",
                source=source_kind,
                support=("amodal" if group.has_reconstruction else "modal"),
                roi=(roi.x, roi.y, roi.size),
            )

            if roi.image_size != source_image.size:
                raise ValueError(
                    f"matting ROI for {group.group_id} does not match "
                    "the source image"
                )
            if source_crop.size != (roi.size, roi.size):
                raise ValueError(
                    f"matting source for {group.group_id} does not match "
                    "its ROI"
                )

            alpha = matte(source_crop)
            if not isinstance(alpha, torch.Tensor) or alpha.ndim != 2:
                raise ValueError(
                    "matting output must be a two-dimensional tensor"
                )
            if tuple(alpha.shape) != (roi.size, roi.size):
                alpha = torch.nn.functional.interpolate(
                    alpha[None, None].to(torch.float32),
                    size=(roi.size, roi.size),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

            alpha_crop = np.clip(
                alpha.detach().cpu().to(torch.float64).numpy(),
                0.0,
                1.0,
            )
            support_crop = crop_array(support, roi).astype(np.uint8)
            valid_support = cv2.dilate(
                support_crop,
                dilation_kernel,
                iterations=1,
            ).astype(bool)
            alpha_crop[
                (alpha_crop <= THRESHOLD_ALPHA) | ~valid_support
            ] = 0.0

            group.matting_source = source_crop
            group.matting_roi = roi
            group.soft_alpha = restore_array(alpha_crop, roi)
            log_event(
                logger,
                "matting",
                "group_result",
                group_id=group.group_id,
                decision="accepted",
                alpha_pixels=int(
                    np.count_nonzero(group.soft_alpha > THRESHOLD_ALPHA)
                ),
            )
