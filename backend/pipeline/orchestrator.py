"""Lazy model resolution and end-to-end image pipeline orchestration."""

import logging
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..config import config
from ..core.helpers import _calc_kernel_size, _image_to_base64
from .background import generate_final_background
from .completion import (
    complete_objects,
    filter_pairs_by_amodal_overlap,
    get_completion_candidates,
    link_overlap_partners,
)
from .layers import extract_object_layers
from .grouping import compose_group_sources, group_reconstructed_objects
from .matting import refine_objects
from .reconstruction import (
    prepare_raw_reconstruction_masks,
    reconstruct_objects,
)
from .segmentation import extract_raw_objects
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
    return extract_raw_objects(image, keywords, processor)


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

    kernel_size = _calc_kernel_size(image_np, 0.0075)
    potential_overlap_pairs = link_overlap_partners(objects)
    _complete_candidates(image, objects, manager)
    overlap_pairs = filter_pairs_by_amodal_overlap(
        objects, potential_overlap_pairs
    )
    if overlap_pairs:
        completion_config = config.get_pipeline_config("completion")
        prepare_raw_reconstruction_masks(
            objects,
            overlap_pairs,
            kernel_size,
            minimum_hole_area_pixels=int(
                completion_config["minimum_hole_area_pixels"]
            ),
            minimum_hole_area_ratio=float(
                completion_config["minimum_hole_area_ratio"]
            ),
            tie_tolerance_ratio=float(
                completion_config["tie_tolerance_ratio"]
            ),
        )

        if any(
            detected.reconstruction_mask is not None
            and np.any(detected.reconstruction_mask)
            for detected in objects
        ) and manager.has_object_reconstruction_model():
            reconstruction_model = (
                manager.get_object_reconstruction_model()
            )
            if reconstruction_model is not None:
                reconstruction_config = config.get_pipeline_config(
                    "object_reconstruction"
                )
                reconstruct_objects(
                    image,
                    objects,
                    reconstruction_model.reconstruct,
                    context_ratio=float(
                        reconstruction_config["context_ratio"]
                    ),
                    blend_allowance_ratio=float(
                        reconstruction_config["blend_allowance_ratio"]
                    ),
                )

    final_groups = group_reconstructed_objects(objects)
    compose_group_sources(image, final_groups)

    matte = manager.get_matting_model().process
    matting_config = config.get_pipeline_config("matting")
    refine_objects(
        image,
        final_groups,
        matte,
        context_ratio=float(matting_config["context_ratio"]),
        support_dilation_pixels=int(
            matting_config["support_dilation_pixels"]
        ),
    )

    background_inpaint = (
        manager.get_background_inpainting_model().process
    )
    layers = extract_object_layers(
        final_groups, kernel_size, background_inpaint
    )
    background = generate_final_background(
        image, final_groups, kernel_size, background_inpaint
    )

    return ProcessResult(
        background_base64=_image_to_base64(background),
        original_width=width,
        original_height=height,
        layers=layers,
    )
