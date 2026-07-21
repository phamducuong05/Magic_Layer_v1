"""Final background removal with a background-only inpainter."""

from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from ..core.layerd_refine import expand_mask, refine_background
from ..core.logging import get_logger, log_event
from .layers import BG_REFINE_NUM_COLORS, BG_REFINE_OUTER_RATIO
from .matting import THRESHOLD_ALPHA
from .types import DetectedObject, GroupedObject


VISIBLE_ALPHA_DILATION = (3, 3)
logger = get_logger(__name__)


def _visible_soft_alpha(
    modal_mask: np.ndarray,
    soft_alpha: np.ndarray,
) -> np.ndarray:
    """Keep alpha coverage only on, or immediately beside, visible pixels."""
    visible_support = expand_mask(
        modal_mask > 0, VISIBLE_ALPHA_DILATION
    ).astype(bool)
    return np.where(visible_support, soft_alpha, 0.0)


def generate_background_from_masks(
    image: Image.Image,
    raw_masks: Sequence[np.ndarray],
    soft_alphas: Sequence[np.ndarray],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
) -> Image.Image:
    """Inpaint all components using hard masks and soft-alpha coverage."""
    union_mask = np.logical_or.reduce([mask > 0 for mask in raw_masks])
    for alpha in soft_alphas:
        union_mask |= alpha > THRESHOLD_ALPHA
    union_mask = expand_mask(union_mask, kernel_size).astype(bool)
    log_event(
        logger,
        "background_inpainting",
        "mask_prepared",
        raw_mask_count=len(raw_masks),
        soft_alpha_count=len(soft_alphas),
        inpaint_pixels=int(np.count_nonzero(union_mask)),
        kernel_size=kernel_size,
    )
    final_mask = Image.fromarray(union_mask.astype(np.uint8) * 255, mode="L")

    background = background_inpaint(image, final_mask)
    log_event(
        logger,
        "background_inpainting",
        "model_result",
        decision="accepted",
        output_size=background.size,
    )
    if background.size != image.size:
        background = background.resize(image.size, Image.Resampling.LANCZOS)

    background_np = np.asarray(background.convert("RGB"), dtype=np.uint8)
    background_np = refine_background(
        background_np,
        union_mask,
        n_outer_ratio=BG_REFINE_OUTER_RATIO,
        max_num_colors=BG_REFINE_NUM_COLORS,
    )
    return Image.fromarray(background_np, mode="RGB")


def generate_final_background(
    image: Image.Image,
    objects: Sequence[DetectedObject | GroupedObject],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
) -> Image.Image:
    """Inpaint visible object coverage once on the original source image."""
    log_event(
        logger,
        "background_inpainting",
        "decision",
        decision="remove_visible_modal_coverage",
        object_count=len(objects),
        reason="hidden_amodal_rgb_must_not_affect_final_background",
    )
    return generate_background_from_masks(
        image,
        [detected.modal_mask for detected in objects],
        [
            _visible_soft_alpha(
                detected.modal_mask,
                detected.soft_alpha,
            )
            for detected in objects
            if detected.soft_alpha is not None
        ],
        kernel_size,
        background_inpaint,
    )
