"""RGBA object-layer rendering with a background-only inpainter."""

import logging
from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from ..core.helpers import _bbox_from_mask, _image_to_base64
from ..core.layerd_refine import refine_background
from ..core.refine import build_inpaint_mask, refine_alpha_with_colors
from .matting import THRESHOLD_ALPHA
from .roi import crop_array
from .types import GroupedObject, ObjectLayer

logger = logging.getLogger(__name__)

BG_REFINE_NUM_COLORS = 10
BG_REFINE_OUTER_RATIO = 0.2


def extract_object_layers(
    objects: Sequence[GroupedObject],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
) -> list[ObjectLayer]:
    """Render final groups from their exact matting RGB source and ROI."""
    layers: list[ObjectLayer] = []

    for group in objects:
        if group.soft_alpha is None:
            continue
        if group.matting_source is None or group.matting_roi is None:
            raise ValueError(
                f"group {group.group_id} has alpha but no matting source"
            )

        roi = group.matting_roi
        source = (
            group.matting_source
            if group.matting_source.mode == "RGB"
            else group.matting_source.convert("RGB")
        )
        if source.size != (roi.size, roi.size):
            raise ValueError(
                f"matting source for {group.group_id} does not match its ROI"
            )

        source_rgb = np.asarray(source, dtype=np.uint8)
        alpha = crop_array(group.soft_alpha, roi).astype(np.float64)
        selected_support = (
            group.amodal_mask > 0
            if group.has_reconstruction
            else group.modal_mask > 0
        )
        support_crop = crop_array(selected_support, roi).astype(bool)
        hard_mask = support_crop | (alpha > THRESHOLD_ALPHA)
        if not np.any(hard_mask):
            continue

        inpaint_mask = build_inpaint_mask(
            source_rgb,
            hard_mask,
            kernel_size,
        ).astype(bool)
        mask_image = Image.fromarray(
            inpaint_mask.astype(np.uint8) * 255,
            mode="L",
        )
        component_background = background_inpaint(source, mask_image)
        if component_background.size != source.size:
            component_background = component_background.resize(
                source.size,
                Image.Resampling.LANCZOS,
            )
        background_rgb = np.asarray(
            component_background.convert("RGB"), dtype=np.uint8
        )
        background_rgb = refine_background(
            background_rgb,
            inpaint_mask,
            n_outer_ratio=BG_REFINE_OUTER_RATIO,
            max_num_colors=BG_REFINE_NUM_COLORS,
        )
        refined_alpha, foreground_rgb = refine_alpha_with_colors(
            source_rgb,
            background_rgb,
            alpha.copy(),
            hard_mask,
            kernel_size,
        )
        refined_alpha = np.clip(refined_alpha, 0.0, 1.0)

        real_pixels = np.zeros((roi.size, roi.size), dtype=bool)
        left, top, right, bottom = roi.inner_box
        real_pixels[top:bottom, left:right] = True
        refined_alpha[~real_pixels] = 0.0
        local_bbox = _bbox_from_mask(
            (refined_alpha > THRESHOLD_ALPHA).astype(np.uint8)
        )
        if local_bbox is None:
            continue

        local_x, local_y, layer_width, layer_height = local_bbox
        rgb_crop = foreground_rgb[
            local_y : local_y + layer_height,
            local_x : local_x + layer_width,
        ]
        alpha_crop = np.rint(
            refined_alpha[
                local_y : local_y + layer_height,
                local_x : local_x + layer_width,
            ]
            * 255
        ).astype(np.uint8)
        rgba_image = Image.fromarray(
            np.dstack((rgb_crop, alpha_crop)), mode="RGBA"
        )
        global_x = roi.x + local_x
        global_y = roi.y + local_y
        layers.append(
            ObjectLayer(
                keyword=group.display_label,
                png_base64=_image_to_base64(rgba_image),
                x=global_x,
                y=global_y,
                width=layer_width,
                height=layer_height,
            )
        )
        logger.info(
            "[Layer] '%s' bbox=%s",
            group.display_label,
            (global_x, global_y, layer_width, layer_height),
        )

    return layers
