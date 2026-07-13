"""Orchestrates text-guided component extraction and background inpainting."""

import logging
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from .core.helpers import (
    _bbox_from_mask,
    _calc_kernel_size,
    _image_to_base64,
    _inference_context,
    _merge_overlapping_masks,
    _normalise_mask,
)
from .core.refine import build_inpaint_mask, refine_alpha_with_colors
from .core.layerd_refine import expand_mask, refine_background
from .models import model_manager

logger = logging.getLogger(__name__)

_THRESHOLD_ALPHA = 0.005
_BG_REFINE_NUM_COLORS = 10
_BG_REFINE_OUTER_RATIO = 0.2


@dataclass
class ObjectLayer:
    keyword: str
    png_base64: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class ProcessResult:
    background_base64: str
    original_width: int
    original_height: int
    layers: List[ObjectLayer] = field(default_factory=list)


def _extract_raw_masks(
    image: Image.Image, keywords: List[str]
) -> tuple[List[np.ndarray], List[str]]:
    """Run SAM3 text prompting and return full-resolution masks and labels."""
    processor = model_manager.get_segmentation_model().get_processor()
    raw_masks: List[np.ndarray] = []
    labels: List[str] = []

    with torch.inference_mode(), _inference_context():
        state = processor.set_image(image)
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue

            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=keyword)
            masks = state.get("masks")
            if masks is None or len(masks) == 0:
                logger.warning("[SAM3] No object found for '%s'", keyword)
                continue

            keyword_masks = [
                _normalise_mask(mask, image.size) for mask in masks
            ]
            keyword_masks = _merge_overlapping_masks(keyword_masks)

            for index, mask in enumerate(keyword_masks):
                raw_masks.append(mask)
                label = (
                    f"{keyword}_{index}"
                    if len(keyword_masks) > 1
                    else keyword
                )
                labels.append(label)
                logger.info("[SAM3] grouped instance '%s'", label)

    return raw_masks, labels


def _refine_masks(
    image_np: np.ndarray, raw_masks: List[np.ndarray]
) -> List[np.ndarray]:
    """Convert SAM3 hard masks into soft alpha mattes with BiRefNet."""
    matting_model = model_manager.get_matting_model().process
    soft_alphas: List[np.ndarray] = []

    with torch.inference_mode(), _inference_context():
        for mask in raw_masks:
            guided_image = image_np.copy()
            guided_image[mask == 0] = 0
            alpha = matting_model(Image.fromarray(guided_image))

            support = cv2.dilate(
                mask, np.ones((5, 5), dtype=np.uint8), iterations=1
            ) > 0
            valid = (alpha > _THRESHOLD_ALPHA) & torch.from_numpy(support).to(
                alpha.device
            )
            alpha[~valid] = 0.0
            soft_alphas.append(
                np.clip(alpha.cpu().to(torch.float64).numpy(), 0.0, 1.0)
            )

    return soft_alphas


def _extract_object_layers(
    image: Image.Image,
    image_np: np.ndarray,
    soft_alphas: List[np.ndarray],
    labels: List[str],
    kernel_size: tuple[int, int],
) -> List[ObjectLayer]:
    """Refine component colors and alpha mattes, then create RGBA crops."""
    inpaint = model_manager.get_inpainting_model().process
    layers: List[ObjectLayer] = []

    for alpha, label in zip(soft_alphas, labels):
        hard_mask = alpha > _THRESHOLD_ALPHA
        bbox = _bbox_from_mask(hard_mask.astype(np.uint8))
        if bbox is None:
            continue

        inpaint_mask = build_inpaint_mask(image_np, hard_mask, kernel_size)
        mask_image = Image.fromarray(
            (inpaint_mask > 0).astype(np.uint8) * 255, mode="L"
        )
        component_background = inpaint(image, mask_image)
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
            n_outer_ratio=_BG_REFINE_OUTER_RATIO,
            max_num_colors=_BG_REFINE_NUM_COLORS,
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


def _generate_final_background(
    image: Image.Image,
    raw_masks: List[np.ndarray],
    kernel_size: tuple[int, int],
) -> Image.Image:
    """Inpaint all detected components from one expanded union mask."""
    union_mask = np.logical_or.reduce([mask > 0 for mask in raw_masks])
    union_mask = expand_mask(union_mask, kernel_size).astype(bool)
    final_mask = Image.fromarray(union_mask.astype(np.uint8) * 255, mode="L")

    inpaint = model_manager.get_inpainting_model().process
    background = inpaint(image, final_mask)
    if background.size != image.size:
        background = background.resize(image.size, Image.Resampling.LANCZOS)

    background_np = np.asarray(background.convert("RGB"), dtype=np.uint8)
    background_np = refine_background(
        background_np,
        union_mask,
        n_outer_ratio=_BG_REFINE_OUTER_RATIO,
        max_num_colors=_BG_REFINE_NUM_COLORS,
    )
    return Image.fromarray(background_np, mode="RGB")


def process_image(image: Image.Image, keywords: List[str]) -> ProcessResult:
    """Coordinate segmentation, matting, layer extraction, and inpainting."""
    # Normalize once so every model and NumPy operation shares the same RGB data.
    image = image.convert("RGB")
    width, height = image.size
    image_np = np.asarray(image, dtype=np.uint8)

    # Stage 1: text-guided SAM3 segmentation.
    raw_masks, labels = _extract_raw_masks(image, keywords)
    if not raw_masks:
        logger.warning("No objects were detected; returning the original background.")
        return ProcessResult(
            background_base64=_image_to_base64(image),
            original_width=width,
            original_height=height,
        )

    # Stage 2: turn hard SAM3 masks into edge-aware soft alpha mattes.
    kernel_size = _calc_kernel_size(image_np)
    soft_alphas = _refine_masks(image_np, raw_masks)

    # Stage 3: refine RGB/alpha and encode each component as a cropped PNG.
    layers = _extract_object_layers(
        image, image_np, soft_alphas, labels, kernel_size
    )

    # Stage 4: remove all detected components in one final background pass.
    background = _generate_final_background(image, raw_masks, kernel_size)

    return ProcessResult(
        background_base64=_image_to_base64(background),
        original_width=width,
        original_height=height,
        layers=layers,
    )
