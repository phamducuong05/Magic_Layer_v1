"""RGBA object-layer rendering with a background-only inpainter."""

from collections.abc import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from ..core.helpers import _bbox_from_mask, _image_to_base64
from ..core.layerd_refine import refine_background
from ..core.logging import get_logger, log_event
from ..core.refine import build_inpaint_mask, refine_alpha_with_colors
from .matting import THRESHOLD_ALPHA, recover_missing_member_alpha
from .roi import crop_array
from .types import GroupedObject, ObjectLayer

logger = get_logger(__name__)

BG_REFINE_NUM_COLORS = 5
BG_REFINE_OUTER_RATIO = 2


def _clean_final_alpha(
    alpha: np.ndarray,
    support: np.ndarray,
    *,
    threshold: float,
    min_component_area_pixels: int,
) -> np.ndarray:
    """Remove weak alpha fringe and tiny components detached from support."""
    cleaned = np.clip(alpha.astype(np.float64), 0.0, 1.0)
    cleaned[cleaned < threshold] = 0.0
    if min_component_area_pixels <= 0 or not np.any(cleaned):
        return cleaned

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0.0).astype(np.uint8),
        connectivity=8,
    )
    for label in range(1, component_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area_pixels and not np.any(
            component & support
        ):
            cleaned[component] = 0.0
    return cleaned


def extract_object_layers(
    objects: Sequence[GroupedObject],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
    *,
    final_alpha_threshold: float = THRESHOLD_ALPHA,
    final_min_component_area_pixels: int = 0,
    alpha_presence_threshold: float = 0.05,
    min_member_alpha_coverage_ratio: float = 0.95,
) -> list[ObjectLayer]:
    """Render final groups from their exact matting RGB source and ROI."""
    if not np.isfinite(final_alpha_threshold) or not (
        0.0 <= final_alpha_threshold <= 1.0
    ):
        raise ValueError("final alpha threshold must be in [0, 1]")
    if final_min_component_area_pixels < 0:
        raise ValueError(
            "final minimum component area must be non-negative"
        )

    layers: list[ObjectLayer] = []

    for group in objects:
        if group.soft_alpha is None:
            log_event(
                logger,
                "layer_extraction",
                "group_decision",
                group_id=group.group_id,
                decision="skip",
                reason="missing_soft_alpha",
            )
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
        selected_support = group.effective_support_mask
        support_crop = crop_array(selected_support, roi).astype(bool)
        hard_mask = support_crop | (alpha > THRESHOLD_ALPHA)
        if not np.any(hard_mask):
            log_event(
                logger,
                "layer_extraction",
                "group_decision",
                group_id=group.group_id,
                decision="skip",
                reason="empty_hard_mask",
            )
            continue

        log_event(
            logger,
            "layer_extraction",
            "group_decision",
            group_id=group.group_id,
            decision="run",
            source=(
                "composed_reconstructed_rgb"
                if group.has_reconstruction
                else "original_modal_rgb"
            ),
            hard_mask_pixels=int(np.count_nonzero(hard_mask)),
        )

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
        refined_alpha = _clean_final_alpha(
            refined_alpha,
            support_crop,
            threshold=final_alpha_threshold,
            min_component_area_pixels=(
                final_min_component_area_pixels
            ),
        )
        refined_alpha = recover_missing_member_alpha(
            refined_alpha,
            group,
            roi,
            presence_threshold=alpha_presence_threshold,
            min_coverage_ratio=min_member_alpha_coverage_ratio,
            stage="final_alpha",
        )

        real_pixels = np.zeros((roi.size, roi.size), dtype=bool)
        left, top, right, bottom = roi.inner_box
        real_pixels[top:bottom, left:right] = True
        refined_alpha[~real_pixels] = 0.0
        local_bbox = _bbox_from_mask(
            (refined_alpha > THRESHOLD_ALPHA).astype(np.uint8)
        )
        if local_bbox is None:
            log_event(
                logger,
                "layer_extraction",
                "group_decision",
                group_id=group.group_id,
                decision="skip",
                reason="empty_refined_alpha",
            )
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
        logger.debug(
            "[Layer] '%s' bbox=%s",
            group.display_label,
            (global_x, global_y, layer_width, layer_height),
        )
        log_event(
            logger,
            "layer_extraction",
            "group_result",
            group_id=group.group_id,
            decision="layer_created",
            bbox=(global_x, global_y, layer_width, layer_height),
            alpha_pixels=int(np.count_nonzero(alpha_crop)),
        )

    return layers
