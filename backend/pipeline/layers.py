"""RGBA object-layer rendering with a background-only inpainter."""

import logging
from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from ..core.helpers import _bbox_from_mask, _image_to_base64
from ..core.layerd_refine import refine_background
from ..core.refine import build_inpaint_mask, refine_alpha_with_colors
from .matting import THRESHOLD_ALPHA
from .types import DetectedObject, ObjectLayer

logger = logging.getLogger(__name__)

BG_REFINE_NUM_COLORS = 10
BG_REFINE_OUTER_RATIO = 0.2


def extract_layers(
    image: Image.Image,
    image_np: np.ndarray,
    soft_alphas: Sequence[np.ndarray],
    labels: Sequence[str],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
) -> list[ObjectLayer]:
    """Create RGBA crops from aligned alpha and label sequences."""
    layers: list[ObjectLayer] = []

    for alpha, label in zip(soft_alphas, labels):
        hard_mask = alpha > THRESHOLD_ALPHA
        bbox = _bbox_from_mask(hard_mask.astype(np.uint8))
        if bbox is None:
            continue

        inpaint_mask = build_inpaint_mask(image_np, hard_mask, kernel_size)
        mask_image = Image.fromarray(
            (inpaint_mask > 0).astype(np.uint8) * 255, mode="L"
        )
        component_background = background_inpaint(image, mask_image)
        if component_background.size != image.size:
            component_background = component_background.resize(
                image.size, Image.Resampling.LANCZOS
            )
        background_np = np.asarray(
            component_background.convert("RGB"), dtype=np.uint8
        )
        background_np = refine_background(
            background_np,
            inpaint_mask.astype(bool),
            n_outer_ratio=BG_REFINE_OUTER_RATIO,
            max_num_colors=BG_REFINE_NUM_COLORS,
        )
        refined_alpha, foreground_rgb = refine_alpha_with_colors(
            image_np,
            background_np,
            alpha.copy(),
            hard_mask,
            kernel_size,
        )

        x, y, layer_width, layer_height = bbox
        rgb_crop = foreground_rgb[y : y + layer_height, x : x + layer_width]
        alpha_crop = np.rint(
            refined_alpha[y : y + layer_height, x : x + layer_width] * 255
        ).astype(np.uint8)
        rgba_image = Image.fromarray(
            np.dstack((rgb_crop, alpha_crop)), mode="RGBA"
        )
        layers.append(
            ObjectLayer(
                keyword=label,
                png_base64=_image_to_base64(rgba_image),
                x=x,
                y=y,
                width=layer_width,
                height=layer_height,
            )
        )
        logger.info("[Layer] '%s' bbox=%s", label, bbox)

    return layers


def extract_object_layers(
    image: Image.Image,
    image_np: np.ndarray,
    objects: Sequence[DetectedObject],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
) -> list[ObjectLayer]:
    """Render objects that have a stored soft alpha."""
    ready_objects = [
        detected for detected in objects if detected.soft_alpha is not None
    ]
    return extract_layers(
        image,
        image_np,
        [detected.soft_alpha for detected in ready_objects],
        [detected.display_label for detected in ready_objects],
        kernel_size,
        background_inpaint,
    )
