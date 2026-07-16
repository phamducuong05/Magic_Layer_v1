"""Lazy model resolution and end-to-end image pipeline orchestration."""

import logging
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..config import config
from ..core.helpers import _calc_kernel_size, _image_to_base64
from ..core.occlusion import assign_pair_roles, effective_hole_area
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


def _effective_hole_areas(
    objects: Sequence,
    *,
    minimum_pixels: int,
    minimum_modal_ratio: float,
) -> dict[str, int]:
    """Attach noise-filtered areas while retaining raw completion diagnostics."""
    areas: dict[str, int] = {}
    for detected in objects:
        raw_area = detected.completion_hole_area
        if raw_area is None:
            continue

        effective_area = effective_hole_area(
            raw_area,
            int(np.count_nonzero(detected.modal_mask)),
            minimum_pixels=minimum_pixels,
            minimum_modal_ratio=minimum_modal_ratio,
        )
        detected.effective_completion_hole_area = effective_area
        areas[detected.object_id] = effective_area

    return areas


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

    if not manager.has_object_reconstruction_model():
        return [detected.modal_mask for detected in objects]

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
    if manager.has_object_reconstruction_model():
        overlap_pairs = link_overlap_partners(objects)
        _complete_candidates(image, objects, manager)
        completion_config = config.get_pipeline_config("completion")
        effective_hole_areas = _effective_hole_areas(
            objects,
            minimum_pixels=int(
                completion_config["minimum_hole_area_pixels"]
            ),
            minimum_modal_ratio=float(
                completion_config["minimum_hole_area_ratio"]
            ),
        )
        pair_decisions = assign_pair_roles(
            overlap_pairs,
            effective_hole_areas,
            tie_tolerance_ratio=float(
                completion_config["tie_tolerance_ratio"]
            ),
        )
        apply_pair_decisions(objects, pair_decisions)
        build_reconstruction_masks(objects, kernel_size)

        if any(
            detected.reconstruction_mask is not None
            and np.any(detected.reconstruction_mask)
            for detected in objects
        ):
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
                )

    matte = manager.get_matting_model().process
    refine_objects(image_np, objects, matte)

    background_inpaint = (
        manager.get_background_inpainting_model().process
    )
    layers = extract_object_layers(
        image, image_np, objects, kernel_size, background_inpaint
    )
    background = generate_final_background(
        image, objects, kernel_size, background_inpaint
    )

    return ProcessResult(
        background_base64=_image_to_base64(background),
        original_width=width,
        original_height=height,
        layers=layers,
    )
