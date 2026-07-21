"""Explicit-callable hard-mask to soft-alpha refinement."""

from collections.abc import Callable, Sequence

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
from .types import GroupedObject

THRESHOLD_ALPHA = 0.005
logger = get_logger(__name__)


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
