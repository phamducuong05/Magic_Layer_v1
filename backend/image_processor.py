"""Orchestrates text-guided component extraction and background inpainting."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

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
from .core.occlusion import (
    ObjectBounds,
    OverlapPair,
    PairDecision,
    assign_pair_roles,
    find_cross_class_overlaps,
)
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


@dataclass
class DetectedObject:
    """One same-class grouped SAM3 object in full-image coordinates."""

    object_id: str
    semantic_class: str
    display_label: str
    modal_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    overlap_partner_ids: set[str] = field(default_factory=set)
    occluder_ids: set[str] = field(default_factory=set)
    amodal_mask: Optional[np.ndarray] = None
    completion_hole_mask: Optional[np.ndarray] = None
    completion_hole_area: Optional[int] = None
    reconstruction_mask: Optional[np.ndarray] = None


def _extract_objects(
    image: Image.Image, keywords: List[str]
) -> List[DetectedObject]:
    """Run SAM3 and describe each same-class grouped detection."""
    processor = model_manager.get_segmentation_model().get_processor()
    objects: List[DetectedObject] = []

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
                bbox = _bbox_from_mask(mask)
                if bbox is None:
                    logger.warning(
                        "[SAM3] Ignoring empty grouped mask for '%s'", keyword
                    )
                    continue

                display_label = (
                    f"{keyword}_{index}"
                    if len(keyword_masks) > 1
                    else keyword
                )
                objects.append(
                    DetectedObject(
                        object_id=f"object-{len(objects)}",
                        semantic_class=keyword,
                        display_label=display_label,
                        modal_mask=mask,
                        bbox=bbox,
                    )
                )
                logger.info("[SAM3] grouped instance '%s'", display_label)

    return objects


def _link_overlap_partners(
    objects: List[DetectedObject],
) -> list[OverlapPair]:
    """Record positive-area, cross-class box overlaps on both objects."""
    for detected in objects:
        detected.overlap_partner_ids.clear()

    pairs = find_cross_class_overlaps(
        [
            ObjectBounds(
                object_id=detected.object_id,
                semantic_class=detected.semantic_class,
                bbox=detected.bbox,
            )
            for detected in objects
        ]
    )
    objects_by_id = {detected.object_id: detected for detected in objects}
    for first_id, second_id in pairs:
        objects_by_id[first_id].overlap_partner_ids.add(second_id)
        objects_by_id[second_id].overlap_partner_ids.add(first_id)

    return pairs


def _complete_overlapping_objects(
    image: Image.Image, objects: List[DetectedObject]
) -> None:
    """Complete each grouped object involved in an overlap as one batch."""
    completion_objects = [
        detected for detected in objects if detected.overlap_partner_ids
    ]
    if not completion_objects:
        return

    amodal_masks = model_manager.get_completion_model().complete(
        image,
        [detected.modal_mask for detected in completion_objects],
        [detected.bbox for detected in completion_objects],
    )
    for detected, amodal_mask in zip(completion_objects, amodal_masks):
        detected.amodal_mask = amodal_mask
        detected.completion_hole_mask = (amodal_mask > 0) & (
            detected.modal_mask == 0
        )
        detected.completion_hole_area = int(
            np.count_nonzero(detected.completion_hole_mask)
        )


def _apply_pair_decisions(
    objects: List[DetectedObject], decisions: List[PairDecision]
) -> None:
    """Record each decisive pair's occluder on its occluded object."""
    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.occluder_ids.clear()

    for decision in decisions:
        if decision.ambiguous:
            continue
        objects_by_id[decision.occluded_id].occluder_ids.add(
            decision.occluder_id
        )


def _build_reconstruction_masks(
    objects: List[DetectedObject], kernel_size: tuple[int, int]
) -> None:
    """Build constrained masks for objects with assigned occluders."""
    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.reconstruction_mask = None
        if (
            not detected.occluder_ids
            or detected.amodal_mask is None
            or detected.completion_hole_mask is None
        ):
            continue

        occluder_union = np.zeros_like(detected.modal_mask, dtype=bool)
        for occluder_id in detected.occluder_ids:
            occluder_union |= objects_by_id[occluder_id].modal_mask > 0

        expanded_support = expand_mask(
            detected.amodal_mask > 0, kernel_size
        ).astype(bool)
        relevant_occluder = (
            occluder_union
            & expanded_support
            & ~(detected.modal_mask > 0)
        )
        detected.reconstruction_mask = (
            detected.completion_hole_mask | relevant_occluder
        )


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
    inpainting_model = model_manager.get_inpainting_model()
    inpaint = inpainting_model.process
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
    soft_alphas: List[np.ndarray],
    kernel_size: tuple[int, int],
) -> Image.Image:
    """Inpaint all components using both hard masks and soft alpha coverage."""
    union_mask = np.logical_or.reduce([mask > 0 for mask in raw_masks])
    for alpha in soft_alphas:
        union_mask |= alpha > _THRESHOLD_ALPHA
    union_mask = expand_mask(union_mask, kernel_size).astype(bool)
    final_mask = Image.fromarray(union_mask.astype(np.uint8) * 255, mode="L")

    inpainting_model = model_manager.get_inpainting_model()
    background = inpainting_model.process(image, final_mask)
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
    objects = _extract_objects(image, keywords)
    if not objects:
        logger.warning("No objects were detected; returning the original background.")
        return ProcessResult(
            background_base64=_image_to_base64(image),
            original_width=width,
            original_height=height,
        )
    overlap_pairs = _link_overlap_partners(objects)
    _complete_overlapping_objects(image, objects)
    pair_decisions = assign_pair_roles(
        overlap_pairs,
        {
            detected.object_id: detected.completion_hole_area
            for detected in objects
            if detected.completion_hole_area is not None
        },
    )
    _apply_pair_decisions(objects, pair_decisions)
    kernel_size = _calc_kernel_size(image_np)
    _build_reconstruction_masks(objects, kernel_size)
    raw_masks = [detected.modal_mask for detected in objects]
    labels = [detected.display_label for detected in objects]

    # Stage 2: turn hard SAM3 masks into edge-aware soft alpha mattes.
    soft_alphas = _refine_masks(image_np, raw_masks)

    # Stage 3: refine RGB/alpha and encode each component as a cropped PNG.
    layers = _extract_object_layers(
        image, image_np, soft_alphas, labels, kernel_size
    )

    # Stage 4: remove all detected components in one final background pass.
    background = _generate_final_background(
        image, raw_masks, soft_alphas, kernel_size
    )

    return ProcessResult(
        background_base64=_image_to_base64(background),
        original_width=width,
        original_height=height,
        layers=layers,
    )
