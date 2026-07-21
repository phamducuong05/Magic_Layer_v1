"""Lazy model resolution and end-to-end image pipeline orchestration."""

import logging
import threading
from functools import wraps
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
from .diagnostics import (
    StageTimings,
    build_pipeline_diagnostics,
    log_pipeline_diagnostics,
    read_peak_gpu_memory,
    start_gpu_memory_tracking,
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
_PIPELINE_LOCK = threading.RLock()


def _serialized_pipeline(function):
    """Prevent concurrent requests from unloading each other's GPU model."""
    @wraps(function)
    def serialized(*args, **kwargs):
        with _PIPELINE_LOCK:
            return function(*args, **kwargs)

    return serialized


def _release_stage_model(manager: Any, category: str) -> None:
    """Release a stage model when supported by the supplied manager."""
    release = getattr(manager, "release_model", None)
    if callable(release):
        release(category)


def _get_diagnostics_config() -> dict[str, bool]:
    """Return backward-compatible observability defaults."""
    defaults = {
        "enabled": True,
        "log_summary": True,
        "track_peak_gpu_memory": False,
    }
    try:
        defaults.update(config.get_pipeline_config("diagnostics"))
    except (KeyError, ValueError):
        pass
    return defaults


def _mark_missing_reconstruction_model(objects: Sequence) -> None:
    """Record why eligible raw objects retained their modal RGB fallback."""
    for detected in objects:
        reconstruction_mask = detected.reconstruction_mask
        if reconstruction_mask is None or not np.any(reconstruction_mask):
            continue
        detected.reconstruction_failure_stage = "model_resolution"
        detected.reconstruction_failure_reason = (
            "no object-reconstruction model is configured"
        )


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


@_serialized_pipeline
def process_masks(
    image: Image.Image,
    keywords: Sequence[str],
    manager: Any = None,
) -> list[np.ndarray]:
    """Run segmentation and conditional completion, returning masks only."""
    manager = _resolve_manager(manager)
    image = image.convert("RGB")
    try:
        objects = _segment(image, keywords, manager)
    finally:
        _release_stage_model(manager, "segmentation")
    if not objects:
        return []

    link_overlap_partners(objects)
    if get_completion_candidates(objects):
        try:
            _complete_candidates(image, objects, manager)
        finally:
            _release_stage_model(manager, "completion")
    return [
        detected.amodal_mask
        if detected.amodal_mask is not None
        else detected.modal_mask
        for detected in objects
    ]


@_serialized_pipeline
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
    diagnostics_config = _get_diagnostics_config()
    diagnostics_enabled = bool(diagnostics_config["enabled"])
    timings = StageTimings()
    gpu_tracking_active = start_gpu_memory_tracking(
        diagnostics_enabled
        and bool(diagnostics_config["track_peak_gpu_memory"])
    )

    try:
        objects = _segment(image, keywords, manager)
    finally:
        _release_stage_model(manager, "segmentation")
    if not objects:
        logger.warning(
            "No objects were detected; returning the original background."
        )
        diagnostics = build_pipeline_diagnostics(
            raw_objects=[],
            final_groups=[],
            potential_pairs=[],
            retained_pairs=[],
            completion_candidate_count=0,
            pair_decisions=[],
            stage_timings_ms=timings.as_milliseconds(),
            peak_gpu_memory_bytes=read_peak_gpu_memory(
                gpu_tracking_active
            ),
        )
        if diagnostics_enabled and diagnostics_config["log_summary"]:
            log_pipeline_diagnostics(diagnostics)
        return ProcessResult(
            background_base64=_image_to_base64(image),
            original_width=width,
            original_height=height,
            diagnostics=diagnostics if diagnostics_enabled else None,
        )

    kernel_size = _calc_kernel_size(image_np, 0.0075)
    potential_overlap_pairs = link_overlap_partners(objects)
    completion_candidate_count = len(get_completion_candidates(objects))
    if completion_candidate_count:
        try:
            with timings.measure("completion"):
                _complete_candidates(image, objects, manager)
        finally:
            _release_stage_model(manager, "completion")
    overlap_pairs = filter_pairs_by_amodal_overlap(
        objects, potential_overlap_pairs
    )
    pair_decisions = []
    if overlap_pairs:
        completion_config = config.get_pipeline_config("completion")
        pair_decisions = prepare_raw_reconstruction_masks(
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

        needs_reconstruction = any(
            detected.reconstruction_mask is not None
            and np.any(detected.reconstruction_mask)
            for detected in objects
        )
        if needs_reconstruction:
            if manager.has_object_reconstruction_model():
                reconstruction_model = None
                try:
                    reconstruction_model = (
                        manager.get_object_reconstruction_model()
                    )
                    if reconstruction_model is None:
                        _mark_missing_reconstruction_model(objects)
                    else:
                        reconstruction_config = config.get_pipeline_config(
                            "object_reconstruction"
                        )
                        with timings.measure("object_reconstruction"):
                            reconstruct_objects(
                                image,
                                objects,
                                reconstruction_model.reconstruct,
                                context_ratio=float(
                                    reconstruction_config["context_ratio"]
                                ),
                                blend_allowance_ratio=float(
                                    reconstruction_config[
                                        "blend_allowance_ratio"
                                    ]
                                ),
                            )
                finally:
                    reconstruction_model = None
                    _release_stage_model(
                        manager, "object_reconstruction"
                    )
            else:
                _mark_missing_reconstruction_model(objects)

    with timings.measure("group_composition"):
        final_groups = group_reconstructed_objects(objects)
        compose_group_sources(image, final_groups)

    matting_config = config.get_pipeline_config("matting")
    matting_model = None
    matte = None
    try:
        matting_model = manager.get_matting_model()
        matte = matting_model.process
        with timings.measure("matting"):
            refine_objects(
                image,
                final_groups,
                matte,
                context_ratio=float(matting_config["context_ratio"]),
                support_dilation_pixels=int(
                    matting_config["support_dilation_pixels"]
                ),
            )
    finally:
        matte = None
        matting_model = None
        _release_stage_model(manager, "matting")

    background_model = None
    background_inpaint = None
    try:
        background_model = manager.get_background_inpainting_model()
        background_inpaint = timings.wrap(
            "background_inpainting", background_model.process
        )
        layers = extract_object_layers(
            final_groups, kernel_size, background_inpaint
        )
        background = generate_final_background(
            image, final_groups, kernel_size, background_inpaint
        )
    finally:
        background_inpaint = None
        background_model = None
        _release_stage_model(manager, "background_inpainting")

    diagnostics = build_pipeline_diagnostics(
        raw_objects=objects,
        final_groups=final_groups,
        potential_pairs=potential_overlap_pairs,
        retained_pairs=overlap_pairs,
        completion_candidate_count=completion_candidate_count,
        pair_decisions=pair_decisions or [],
        stage_timings_ms=timings.as_milliseconds(),
        peak_gpu_memory_bytes=read_peak_gpu_memory(gpu_tracking_active),
    )
    if diagnostics_enabled and diagnostics_config["log_summary"]:
        log_pipeline_diagnostics(diagnostics)

    return ProcessResult(
        background_base64=_image_to_base64(background),
        original_width=width,
        original_height=height,
        layers=layers,
        diagnostics=diagnostics if diagnostics_enabled else None,
    )
