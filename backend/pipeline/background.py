"""Final background removal with a background-only inpainter."""

from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from ..core.layerd_refine import expand_mask, refine_background
from .layers import BG_REFINE_NUM_COLORS, BG_REFINE_OUTER_RATIO
from .matting import THRESHOLD_ALPHA
from .types import DetectedObject


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
    final_mask = Image.fromarray(union_mask.astype(np.uint8) * 255, mode="L")

    background = background_inpaint(image, final_mask)
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
    objects: Sequence[DetectedObject],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
) -> Image.Image:
    """Generate a background from per-object modal masks and soft alphas."""
    return generate_background_from_masks(
        image,
        [detected.modal_mask for detected in objects],
        [
            detected.soft_alpha
            for detected in objects
            if detected.soft_alpha is not None
        ],
        kernel_size,
        background_inpaint,
    )
