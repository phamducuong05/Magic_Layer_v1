"""Lazy model resolution and end-to-end image pipeline orchestration."""

import logging
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..config import config
from ..core.helpers import _calc_kernel_size, _image_to_base64
from ..core.occlusion import assign_pair_roles
from .background import generate_final_background
from .completion import (
    complete_objects,
    get_completion_candidates,
    link_overlap_partners,
)
from .layers import extract_object_layers
from .matting import refine_objects
from .reconstruction import (
    apply_pair_decisions,
    build_reconstruction_masks,
    reconstruct_objects,
)
from .segmentation import extract_objects
from .types import ProcessResult

logger = logging.getLogger(__name__)


def _resolve_manager(manager: Any):
    if manager is not None:
        return manager

    from ..models import model_manager

    return model_manager


def _segment(
    image: Image.Image, keywords: Sequence[str], manager: Any
):
    processor = manager.get_segmentation_model().get_processor()
    return extract_objects(image, keywords, processor)


def _complete_candidates(
    image: Image.Image, objects: Sequence, manager: Any
) -> None:
    candidates = get_completion_candidates(objects)
    if candidates:
        completion_config = config.get_pipeline_config("completion")
        complete_objects(
            image,
            candidates,
            manager.get_completion_model(),
            max_area_growth_ratio=float(
                completion_config["max_area_growth_ratio"]
            ),
            max_bbox_growth_ratio=float(
                completion_config["max_bbox_growth_ratio"]
            ),
        )


def process_masks(
    image: Image.Image,
    keywords: Sequence[str],
    manager: Any = None,
) -> list[np.ndarray]:
    """Run segmentation and conditional completion, returning masks only."""
    manager = _resolve_manager(manager)
    image = image.convert("RGB")
    objects = _segment(image, keywords, manager)
    if not objects:
        return []

    link_overlap_partners(objects)
    _complete_candidates(image, objects, manager)
    return [
        detected.amodal_mask
        if detected.amodal_mask is not None
        else detected.modal_mask
        for detected in objects
    ]


def process_image(
    image: Image.Image,
    keywords: Sequence[str],
    manager: Any = None,
) -> ProcessResult:
    """Coordinate segmentation, matting, layer extraction, and inpainting."""
    manager = _resolve_manager(manager)
    image = image.convert("RGB")
    width, height = image.size
    image_np = np.asarray(image, dtype=np.uint8)

    objects = _segment(image, keywords, manager)
    if not objects:
        logger.warning(
            "No objects were detected; returning the original background."
        )
        return ProcessResult(
            background_base64=_image_to_base64(image),
            original_width=width,
            original_height=height,
        )

    overlap_pairs = link_overlap_partners(objects)
    _complete_candidates(image, objects, manager)
    pair_decisions = assign_pair_roles(
        overlap_pairs,
        {
            detected.object_id: detected.completion_hole_area
            for detected in objects
            if detected.completion_hole_area is not None
        },
    )
    apply_pair_decisions(objects, pair_decisions)
    kernel_size = _calc_kernel_size(image_np, 0.0075)
    build_reconstruction_masks(objects, kernel_size)

    inpaint = manager.get_inpainting_model().process
    reconstruction_config = config.get_pipeline_config("reconstruction")
    reconstruct_objects(
        image,
        objects,
        inpaint,
        context_ratio=float(reconstruction_config["context_ratio"]),
    )

    matte = manager.get_matting_model().process
    refine_objects(image_np, objects, matte)

    layers = extract_object_layers(
        image, image_np, objects, kernel_size, inpaint
    )
    background = generate_final_background(
        image, objects, kernel_size, inpaint
    )

    return ProcessResult(
        background_base64=_image_to_base64(background),
        original_width=width,
        original_height=height,
        layers=layers,
    )
