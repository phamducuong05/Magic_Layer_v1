"""Lazy model resolution and end-to-end image pipeline orchestration."""

import threading
from functools import wraps
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..config import config
from ..core.helpers import _calc_kernel_size, _image_to_base64
from ..core.logging import get_logger, log_event, trace_stage
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
from .matting import refine_objects, refine_reconstruction_supports
from .reconstruction import (
    prepare_raw_reconstruction_masks,
    reconstruct_objects,
)
from .relationships import plan_object_relationships
from .segmentation import extract_raw_objects
from .types import ProcessResult

logger = get_logger(__name__)
_PIPELINE_LOCK = threading.RLock()


def _serialized_pipeline(function):
    """Prevent concurrent requests from unloading each other's GPU model."""
    @wraps(function)
    def serialized(*args, **kwargs):
        with _PIPELINE_LOCK:
            return function(*args, **kwargs)

    return serialized


def _trace_pipeline(function):
    """Trace one complete request, including uncaught failures."""
    @wraps(function)
    def traced(image, keywords, *args, **kwargs):
        with trace_stage(
            logger,
            "pipeline",
            image_size=image.size,
            keywords=list(keywords),
        ):
            return function(image, keywords, *args, **kwargs)

    return traced


def _release_stage_model(
    manager: Any, category: str, *, retain_cpu: bool = False
) -> None:
    """Release a stage model, optionally retaining reusable CPU weights."""
    if retain_cpu:
        offload = getattr(manager, "offload_model", None)
        if callable(offload):
            offload(category)
            return
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


def _get_background_inpainting_config() -> dict[str, Any]:
    """Return defaults compatible with configurations predating artifacts."""
    defaults: dict[str, Any] = {
        "mask_dilation_scale": 0.00375,
        "diagnostics_directory": None,
    }
    try:
        defaults.update(config.get_pipeline_config("background_inpainting"))
    except (KeyError, ValueError):
        pass
    return defaults


def _get_segmentation_config() -> dict[str, Any]:
    """Return defaults for segmentation diagnostics."""
    defaults: dict[str, Any] = {
        "diagnostics_directory": None,
    }
    try:
        defaults.update(config.get_pipeline_config("segmentation"))
    except (KeyError, ValueError):
        pass
    return defaults


def _get_cross_class_grouping_config() -> dict[str, Any]:
    """Return tunable relationship defaults for older configurations."""
    defaults: dict[str, Any] = {
        "enabled": True,
        "containment_threshold": 0.70,
        "max_bbox_size_ratio": 0.50,
        "large_mask_ratio": 0.45,
        "large_bbox_ratio": 0.75,
        "dimension_tolerance_ratio": 0.05,
    }
    try:
        defaults.update(config.get_pipeline_config("cross_class_grouping"))
    except (KeyError, ValueError):
        pass
    return defaults


def _plan_relationships(
    objects: Sequence,
    image_size: tuple[int, int],
):
    """Classify raw pairs without invoking any model-dependent stage."""
    settings = _get_cross_class_grouping_config()
    return plan_object_relationships(
        objects,
        image_size,
        enabled=bool(settings["enabled"]),
        containment_threshold=float(settings["containment_threshold"]),
        max_bbox_size_ratio=float(settings["max_bbox_size_ratio"]),
        large_mask_ratio=float(settings["large_mask_ratio"]),
        large_bbox_ratio=float(settings["large_bbox_ratio"]),
        dimension_tolerance_ratio=float(
            settings["dimension_tolerance_ratio"]
        ),
    )


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
    image: Image.Image,
    keywords: Sequence[str],
    manager: Any,
    diagnostics_directory: str | None = None,
):
    processor = manager.get_segmentation_model().get_processor()
    return extract_raw_objects(
        image, keywords, processor, diagnostics_directory=diagnostics_directory
    )


def _complete_candidates(
    image: Image.Image,
    objects: Sequence,
    manager: Any,
    candidates: Sequence | None = None,
) -> None:
    candidates = (
        list(candidates)
        if candidates is not None
        else get_completion_candidates(objects)
    )
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
@_trace_pipeline
def process_masks(
    image: Image.Image,
    keywords: Sequence[str],
    manager: Any = None,
) -> list[np.ndarray]:
    """Run segmentation and conditional completion, returning masks only."""
    manager = _resolve_manager(manager)
    image = image.convert("RGB")
    segmentation_config = _get_segmentation_config()
    segmentation_diagnostics_directory = segmentation_config.get(
        "diagnostics_directory"
    )
    try:
        with trace_stage(logger, "segmentation"):
            objects = _segment(
                image,
                keywords,
                manager,
                diagnostics_directory=segmentation_diagnostics_directory,
            )
            log_event(
                logger,
                "segmentation",
                "result",
                object_count=len(objects),
            )
    finally:
        _release_stage_model(manager, "segmentation")
    if not objects:
        return []

    relationship_plan = _plan_relationships(objects, image.size)
    with trace_stage(logger, "overlap_detection"):
        pairs = link_overlap_partners(
            objects,
            relationship_plan.external_overlap_pairs,
        )
        log_event(
            logger,
            "overlap_detection",
            "result",
            candidate_pair_count=len(pairs),
        )
    candidates = get_completion_candidates(objects)
    if candidates:
        try:
            with trace_stage(
                logger, "completion", candidate_count=len(candidates)
            ):
                _complete_candidates(image, objects, manager, candidates)
        finally:
            _release_stage_model(manager, "completion")
    else:
        log_event(
            logger,
            "completion",
            "skip",
            level="INFO",
            reason="no_cross_class_overlap_candidates",
        )
    return [
        detected.amodal_mask
        if detected.amodal_mask is not None
        else detected.modal_mask
        for detected in objects
    ]


@_serialized_pipeline
@_trace_pipeline
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
    segmentation_config = _get_segmentation_config()
    segmentation_diagnostics_directory = segmentation_config.get(
        "diagnostics_directory"
    )

    try:
        with trace_stage(logger, "segmentation"):
            objects = _segment(
                image,
                keywords,
                manager,
                diagnostics_directory=segmentation_diagnostics_directory,
            )
            log_event(
                logger,
                "segmentation",
                "result",
                object_count=len(objects),
            )
    finally:
        _release_stage_model(manager, "segmentation")
    if not objects:
        logger.warning(
            "No objects were detected; returning the original background."
        )
        log_event(
            logger,
            "pipeline",
            "decision",
            decision="return_original_background",
            reason="no_objects_detected",
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
            relationship_plan=None,
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
    background_config = _get_background_inpainting_config()
    background_kernel_size = _calc_kernel_size(
        image_np,
        float(background_config.get("mask_dilation_scale", 0.00375)),
    )
    background_diagnostics_directory = background_config.get(
        "diagnostics_directory"
    )
    relationship_plan = _plan_relationships(objects, image.size)
    with trace_stage(logger, "overlap_detection"):
        potential_overlap_pairs = link_overlap_partners(
            objects,
            relationship_plan.external_overlap_pairs,
        )
        log_event(
            logger,
            "overlap_detection",
            "result",
            candidate_pair_count=len(potential_overlap_pairs),
        )
    completion_candidates = get_completion_candidates(objects)
    completion_candidate_count = len(completion_candidates)
    if completion_candidate_count:
        try:
            with trace_stage(
                logger,
                "completion",
                candidate_count=completion_candidate_count,
            ):
                with timings.measure("completion"):
                    _complete_candidates(
                        image, objects, manager, completion_candidates
                    )
        finally:
            _release_stage_model(manager, "completion")
    else:
        log_event(
            logger,
            "completion",
            "skip",
            level="INFO",
            reason="no_cross_class_overlap_candidates",
        )

    if potential_overlap_pairs:
        with trace_stage(logger, "amodal_overlap_validation"):
            overlap_pairs = filter_pairs_by_amodal_overlap(
                objects, potential_overlap_pairs
            )
            log_event(
                logger,
                "amodal_overlap_validation",
                "result",
                retained_pair_count=len(overlap_pairs),
            )
    else:
        overlap_pairs = []
        log_event(
            logger,
            "amodal_overlap_validation",
            "skip",
            level="INFO",
            reason="no_potential_pairs",
        )
    pair_decisions = []
    if overlap_pairs:
        completion_config = config.get_pipeline_config("completion")
        reconstruction_config = config.get_pipeline_config(
            "object_reconstruction"
        )
        with trace_stage(logger, "depth_ordering"):
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
                generation_mask_dilation_pixels=int(
                    reconstruction_config.get(
                        "generation_mask_dilation_pixels", 0
                    )
                ),
                generation_mask_closing_pixels=int(
                    reconstruction_config.get(
                        "generation_mask_closing_pixels", 0
                    )
                ),
                foreign_modal_max_hole_area_pixels=int(
                    reconstruction_config.get(
                        "foreign_modal_max_hole_area_pixels", 0
                    )
                ),
                foreign_modal_dilation_pixels=int(
                    reconstruction_config.get(
                        "foreign_modal_dilation_pixels", 0
                    )
                ),
                foreign_modal_closing_pixels=int(
                    reconstruction_config.get(
                        "foreign_modal_closing_pixels", 0
                    )
                ),
                replacement_domain_margin_pixels=int(
                    reconstruction_config.get(
                        "replacement_domain_margin_pixels", 2
                    )
                ),
                accepted_target_bbox_margin_pixels=int(
                    reconstruction_config.get(
                        "accepted_target_bbox_margin_pixels", 4
                    )
                ),
                foreign_protection_dilation_pixels=int(
                    reconstruction_config.get(
                        "foreign_protection_dilation_pixels", 1
                    )
                ),
                support_margin_pixels=int(
                    reconstruction_config.get(
                        "support_margin_pixels", max(kernel_size) // 2
                    )
                ),
                composition_margin_pixels=int(
                    reconstruction_config.get(
                        "composition_margin_pixels",
                        reconstruction_config.get(
                            "support_margin_pixels", max(kernel_size) // 2
                        ),
                    )
                ),
                context_ratio=float(reconstruction_config["context_ratio"]),
            ) or []
            log_event(
                logger,
                "depth_ordering",
                "summary",
                level="INFO",
                pairs=len(pair_decisions),
                one_way=sum(
                    len(decision.reconstruction_directions) == 1
                    for decision in pair_decisions
                ),
                bidirectional=sum(
                    decision.bidirectional for decision in pair_decisions
                ),
                ambiguous=sum(
                    decision.ambiguous for decision in pair_decisions
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
                retain_reconstruction_weights = False
                try:
                    with trace_stage(logger, "object_reconstruction"):
                        reconstruction_model = (
                            manager.get_object_reconstruction_model()
                        )
                        if reconstruction_model is None:
                            _mark_missing_reconstruction_model(objects)
                            log_event(
                                logger,
                                "object_reconstruction",
                                "skip",
                                level="INFO",
                                reason="model_not_resolved",
                            )
                        else:
                            retain_reconstruction_weights = True
                            with timings.measure("object_reconstruction"):
                                reconstruct_objects(
                                    image,
                                    objects,
                                    reconstruction_model.reconstruct,
                                    reconstruct_many=(
                                        reconstruction_model.reconstruct_many
                                    ),
                                    context_ratio=float(
                                        reconstruction_config[
                                            "context_ratio"
                                        ]
                                    ),
                                    blend_allowance_ratio=float(
                                        reconstruction_config[
                                            "blend_allowance_ratio"
                                        ]
                                    ),
                                    prompt_template=str(
                                        reconstruction_config.get(
                                            "prompt_template",
                                            "Reconstruct only the hidden continuation of the "
                                            "{target} behind {occluders}. Do not recreate "
                                            "{occluders}.",
                                        )
                                    ),
                                    style_hint=str(
                                        reconstruction_config.get(
                                            "style_hint", ""
                                        )
                                    ),
                                    diagnostics_directory=(
                                        reconstruction_config.get(
                                            "diagnostics_directory"
                                        )
                                    ),
                                    debug_artifacts_provider=getattr(
                                        reconstruction_model,
                                        "consume_debug_artifacts",
                                        None,
                                    ),
                                )
                finally:
                    reconstruction_model = None
                    _release_stage_model(
                        manager,
                        "object_reconstruction",
                        retain_cpu=retain_reconstruction_weights,
                    )
            else:
                _mark_missing_reconstruction_model(objects)
                log_event(
                    logger,
                    "object_reconstruction",
                    "skip",
                    level="INFO",
                    reason="model_not_configured",
                )
        else:
            log_event(
                logger,
                "object_reconstruction",
                "skip",
                level="INFO",
                reason="no_reconstruction_masks",
            )
    else:
        log_event(
            logger,
            "depth_ordering",
            "skip",
            level="INFO",
            reason="no_retained_amodal_overlap_pairs",
        )
        log_event(
            logger,
            "object_reconstruction",
            "skip",
            level="INFO",
            reason="no_depth_ordered_occluded_objects",
        )

    matting_config = config.get_pipeline_config("matting")
    has_accepted_reconstruction = any(
        detected.reconstruction_canvas is not None for detected in objects
    )
    reconstruction_config = (
        config.get_pipeline_config("object_reconstruction")
        if has_accepted_reconstruction
        else {}
    )
    matting_model = None
    matte = None
    try:
        matting_model = manager.get_matting_model()
        matte = matting_model.process
        if has_accepted_reconstruction and bool(
            reconstruction_config.get("support_refinement_enabled", True)
        ):
            with trace_stage(logger, "reconstruction_support_refinement"):
                with timings.measure("reconstruction_support_refinement"):
                    refine_reconstruction_supports(
                        image,
                        objects,
                        matte,
                        alpha_low_threshold=float(
                            reconstruction_config.get(
                                "support_alpha_low_threshold", 0.2
                            )
                        ),
                        alpha_high_threshold=float(
                            reconstruction_config.get(
                                "support_alpha_high_threshold", 0.7
                            )
                        ),
                        change_threshold=float(
                            reconstruction_config.get(
                                "support_change_threshold", 8.0
                            )
                        ),
                        connection_margin_pixels=int(
                            reconstruction_config.get(
                                "support_connection_margin_pixels", 4
                            )
                        ),
                        max_extension_area_ratio=float(
                            reconstruction_config.get(
                                "support_max_extension_area_ratio", 2.0
                            )
                        ),
                        min_component_area_pixels=int(
                            reconstruction_config.get(
                                "support_min_component_area_pixels", 8
                            )
                        ),
                        generation_evidence_margin_pixels=int(
                            reconstruction_config.get(
                                "support_generation_evidence_margin_pixels",
                                2,
                            )
                        ),
                        require_generation_evidence=bool(
                            reconstruction_config.get(
                                "support_require_generation_evidence", True
                            )
                        ),
                        alpha_write_epsilon=float(
                            reconstruction_config.get(
                                "support_alpha_write_epsilon", 0.01
                            )
                        ),
                        alpha_feather_pixels=int(
                            reconstruction_config.get(
                                "support_alpha_feather_pixels", 0
                            )
                        ),
                        fallback_to_validated_output=bool(
                            reconstruction_config.get(
                                "support_fallback_to_validated_output", True
                            )
                        ),
                        diagnostics_directory=(
                            reconstruction_config.get(
                                "diagnostics_directory"
                            )
                        ),
                    )

        with trace_stage(logger, "grouping", object_count=len(objects)):
            with timings.measure("group_composition"):
                final_groups = group_reconstructed_objects(
                    objects,
                    pair_decisions or [],
                    merge_edges=relationship_plan.merge_edges,
                )
                compose_group_sources(image, final_groups)
            log_event(
                logger,
                "grouping",
                "result",
                group_count=len(final_groups),
            )

        with trace_stage(logger, "matting", group_count=len(final_groups)):
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
        with trace_stage(
            logger, "layer_extraction", group_count=len(final_groups)
        ):
            layers = extract_object_layers(
                final_groups,
                kernel_size,
                background_inpaint,
                final_alpha_threshold=float(
                    matting_config.get("final_alpha_threshold", 0.03)
                ),
                final_min_component_area_pixels=int(
                    matting_config.get(
                        "final_min_component_area_pixels", 4
                    )
                ),
            )
            log_event(
                logger,
                "layer_extraction",
                "result",
                layer_count=len(layers),
            )
        with trace_stage(logger, "background_inpainting"):
            if background_diagnostics_directory is None:
                background = generate_final_background(
                    image,
                    final_groups,
                    background_kernel_size,
                    background_inpaint,
                )
            else:
                background = generate_final_background(
                    image,
                    final_groups,
                    background_kernel_size,
                    background_inpaint,
                    diagnostics_directory=(
                        background_diagnostics_directory
                    ),
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
        relationship_plan=relationship_plan,
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
