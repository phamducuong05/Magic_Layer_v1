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
    diagnostics_directory: str | Path | None = None,
) -> None:
    """Extend incomplete amodal support using reconstructed RGB evidence.

    BiRefNet supplies class-agnostic foreground confidence on each accepted
    reconstruction crop. The model-generation mask remains an inference input
    only; foreground evidence may use the wider post-inference accepted RGB
    region when it connects to the visible target and was changed by the model.
    """
    if not 0.0 <= alpha_low_threshold <= alpha_high_threshold <= 1.0:
        raise ValueError(
            "support alpha thresholds must satisfy 0 <= low <= high <= 1"
        )
    if not math.isfinite(change_threshold) or change_threshold < 0:
        raise ValueError("support change threshold must be non-negative")
    if connection_margin_pixels < 0:
        raise ValueError("support connection margin must be non-negative")
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
        canvas = target.reconstruction_canvas
        roi = target.reconstruction_roi
        if (
            canvas is None
            or roi is None
            or reconstruction_mask is None
            or not np.any(reconstruction_mask)
        ):
            continue

        modal = target.modal_mask > 0
        accepted_rgb_mask = (
            target.reconstruction_accepted_rgb_mask.astype(bool)
            if target.reconstruction_accepted_rgb_mask is not None
            else reconstruction_mask.astype(bool)
        )
        target.reconstruction_write_mask = accepted_rgb_mask.copy()
        target.reconstruction_support_mask = modal.copy()

        if roi.image_size != source.size or canvas.size != (roi.size, roi.size):
            log_event(
                logger,
                "reconstruction_support",
                "object_decision",
                object_id=target.object_id,
                decision="fallback",
                reason="invalid_reconstruction_roi_or_canvas",
            )
            continue

        try:
            alpha_crop = _resize_alpha(matte(canvas.convert("RGB")), roi.size)
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
            continue

        source_crop = np.asarray(crop_image(source, roi), dtype=np.uint8)
        reconstructed_crop = np.asarray(
            canvas.convert("RGB"), dtype=np.uint8
        )
        modal_crop = crop_array(modal, roi).astype(bool)
        accepted_rgb_crop = crop_array(
            accepted_rgb_mask, roi
        ).astype(bool)
        protected_mask = (
            target.reconstruction_protected_mask.astype(bool)
            if target.reconstruction_protected_mask is not None
            else ~accepted_rgb_mask
        )
        protected_crop = crop_array(protected_mask, roi).astype(bool)

        core_crop = modal_crop
        if connection_margin_pixels:
            radius = connection_margin_pixels
            kernel = np.ones((2 * radius + 1,) * 2, dtype=np.uint8)
            connection_anchor = cv2.dilate(
                core_crop.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
        else:
            connection_anchor = core_crop

        candidate = (
            (alpha_crop >= alpha_low_threshold)
            & accepted_rgb_crop
        )
        strong_foreground = alpha_crop >= alpha_high_threshold
        change_distance = np.mean(
            np.abs(
                reconstructed_crop.astype(np.int16)
                - source_crop.astype(np.int16)
            ),
            axis=2,
        )
        changed_by_model = change_distance >= change_threshold

        accepted = np.zeros_like(candidate)
        _, labels = cv2.connectedComponents(
            candidate.astype(np.uint8), connectivity=8
        )
        for label in range(1, int(labels.max()) + 1):
            component = labels == label
            if (
                np.any(component & connection_anchor)
                and np.any(component & strong_foreground)
                and np.any(component & changed_by_model)
            ):
                accepted |= component

        extension_crop = accepted & ~core_crop

        alpha_full = restore_array(alpha_crop, roi)
        extension_full = restore_array(extension_crop, roi).astype(bool)
        target.reconstruction_evidence_alpha = alpha_full
        target.reconstruction_extension_mask = extension_full
        target.reconstruction_support_mask = modal | extension_full

        if diagnostics_directory is not None:
            directory = _reconstruction_artifact_directory(
                diagnostics_directory, target.object_id
            )
            Image.fromarray(
                np.rint(alpha_crop * 255).astype(np.uint8), mode="L"
            ).save(directory / "reconstruction_birefnet_alpha.png")
            _save_mask(
                directory / "reconstruction_extension_mask.png",
                extension_crop,
            )
            _save_mask(
                directory / "birefnet_candidate.png",
                candidate,
            )
            _save_mask(
                directory / "reconstruction_write_mask.png",
                crop_array(target.reconstruction_write_mask, roi),
            )
            _save_mask(
                directory / "final_reconstruction_support.png",
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
                np.count_nonzero(protected_crop)
            ),
            extension_pixels=int(np.count_nonzero(extension_full)),
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
