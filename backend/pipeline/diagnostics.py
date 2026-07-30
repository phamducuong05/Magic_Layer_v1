"""Structured, pixel-free observability for the image pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ..core.logging import get_logger
from ..core.occlusion import OverlapPair, PairDecision
from .relationships import PairRelation, RelationshipPlan
from .types import DetectedObject, GroupedObject


logger = get_logger(__name__)

REQUIRED_TIMING_STAGES = (
    "completion",
    "object_reconstruction",
    "group_composition",
    "matting",
    "background_inpainting",
)


@dataclass(frozen=True)
class ObjectDiagnostics:
    object_id: str
    semantic_class: str
    modal_area: int
    amodal_area: int
    raw_completion_hole_area: int | None
    effective_completion_hole_area: int | None
    mask_image_ratio: float | None = None
    bbox_image_ratio: float | None = None
    large_mask_veto: bool = False
    large_bbox_veto: bool = False


@dataclass(frozen=True)
class PairDecisionDiagnostics:
    first_id: str
    second_id: str
    occluded_id: str | None
    occluder_id: str | None
    ambiguous_reason: str | None
    reconstruction_directions: tuple[tuple[str, str], ...]
    first_hidden_by_second_area: int
    second_hidden_by_first_area: int
    bidirectional: bool


@dataclass(frozen=True)
class GroupDiagnostics:
    group_id: str
    member_ids: tuple[str, ...]
    reconstruction_conflicts: tuple[tuple[str, str], ...]
    semantic_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationshipDecisionDiagnostics:
    first_id: str
    second_id: str
    relation: str
    reason: str
    containment_ratio: float | None
    bbox_size_ratio: float | None


@dataclass(frozen=True)
class FallbackDiagnostics:
    object_id: str
    stage: str
    reason: str


@dataclass(frozen=True)
class PipelineDiagnostics:
    raw_object_count: int
    final_group_count: int
    potential_overlap_pair_count: int
    retained_amodal_pair_count: int
    completion_candidate_count: int
    objects: tuple[ObjectDiagnostics, ...]
    pair_decisions: tuple[PairDecisionDiagnostics, ...]
    groups: tuple[GroupDiagnostics, ...]
    fallbacks: tuple[FallbackDiagnostics, ...]
    stage_timings_ms: dict[str, float]
    peak_gpu_memory_bytes: int | None
    same_class_merge_count: int = 0
    cross_class_containment_merge_count: int = 0
    regular_overlap_count: int = 0
    external_overlap_count: int = 0
    large_mask_veto_count: int = 0
    large_bbox_veto_count: int = 0
    relationship_decisions: tuple[
        RelationshipDecisionDiagnostics, ...
    ] = ()

    def to_log_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata without image arrays or model features."""
        return asdict(self)


class StageTimings:
    """Accumulate wall-clock durations for named pipeline stages."""

    def __init__(self) -> None:
        self._seconds = {stage: 0.0 for stage in REQUIRED_TIMING_STAGES}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self._seconds[stage] = self._seconds.get(stage, 0.0) + (
                perf_counter() - start
            )

    def wrap(self, stage: str, function: Callable[..., Any]):
        """Return a callable that accumulates only the wrapped model runtime."""

        @wraps(function)
        def timed(*args, **kwargs):
            with self.measure(stage):
                return function(*args, **kwargs)

        return timed

    def as_milliseconds(self) -> dict[str, float]:
        return {
            stage: round(seconds * 1000.0, 3)
            for stage, seconds in self._seconds.items()
        }


def start_gpu_memory_tracking(enabled: bool) -> bool:
    """Reset CUDA peak stats only when explicitly requested and available."""
    if not enabled or not torch.cuda.is_available():
        return False
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:
        logger.warning("Unable to reset CUDA peak memory stats: %s", exc)
        return False
    return True


def read_peak_gpu_memory(tracking_active: bool) -> int | None:
    """Read peak allocated CUDA bytes for an active diagnostic run."""
    if not tracking_active:
        return None
    try:
        return int(torch.cuda.max_memory_allocated())
    except Exception as exc:
        logger.warning("Unable to read CUDA peak memory stats: %s", exc)
        return None


def build_pipeline_diagnostics(
    *,
    raw_objects: Sequence[DetectedObject],
    final_groups: Sequence[GroupedObject],
    potential_pairs: Sequence[OverlapPair],
    retained_pairs: Sequence[OverlapPair],
    completion_candidate_count: int,
    pair_decisions: Sequence[PairDecision],
    stage_timings_ms: Mapping[str, float],
    peak_gpu_memory_bytes: int | None,
    relationship_plan: RelationshipPlan | None = None,
) -> PipelineDiagnostics:
    """Build immutable diagnostic metadata from final pipeline state."""
    relationship_metrics = (
        relationship_plan.object_metrics if relationship_plan is not None else {}
    )
    object_records = tuple(
        ObjectDiagnostics(
            object_id=detected.object_id,
            semantic_class=detected.semantic_class,
            modal_area=int(np.count_nonzero(detected.modal_mask)),
            amodal_area=int(
                np.count_nonzero(
                    detected.amodal_mask
                    if detected.amodal_mask is not None
                    else detected.modal_mask
                )
            ),
            raw_completion_hole_area=detected.completion_hole_area,
            effective_completion_hole_area=(
                detected.effective_completion_hole_area
            ),
            mask_image_ratio=(
                relationship_metrics[detected.object_id].mask_image_ratio
                if detected.object_id in relationship_metrics
                else None
            ),
            bbox_image_ratio=(
                relationship_metrics[detected.object_id].bbox_image_ratio
                if detected.object_id in relationship_metrics
                else None
            ),
            large_mask_veto=(
                relationship_metrics[detected.object_id].large_mask_veto
                if detected.object_id in relationship_metrics
                else False
            ),
            large_bbox_veto=(
                relationship_metrics[detected.object_id].large_bbox_veto
                if detected.object_id in relationship_metrics
                else False
            ),
        )
        for detected in raw_objects
    )
    pair_records = tuple(
        PairDecisionDiagnostics(
            first_id=decision.first_id,
            second_id=decision.second_id,
            occluded_id=decision.occluded_id,
            occluder_id=decision.occluder_id,
            ambiguous_reason=(
                "completion-hole areas are within tie tolerance"
                if decision.ambiguous
                else None
            ),
            reconstruction_directions=decision.reconstruction_directions,
            first_hidden_by_second_area=(
                decision.first_hidden_by_second_area
            ),
            second_hidden_by_first_area=(
                decision.second_hidden_by_first_area
            ),
            bidirectional=decision.bidirectional,
        )
        for decision in pair_decisions
    )
    group_records = tuple(
        GroupDiagnostics(
            group_id=group.group_id,
            member_ids=group.member_ids,
            reconstruction_conflicts=group.reconstruction_conflicts,
            semantic_classes=group.semantic_classes,
        )
        for group in final_groups
    )
    relationship_records = tuple(
        RelationshipDecisionDiagnostics(
            first_id=decision.first_id,
            second_id=decision.second_id,
            relation=decision.relation.value,
            reason=decision.reason,
            containment_ratio=decision.containment_ratio,
            bbox_size_ratio=decision.bbox_size_ratio,
        )
        for decision in (
            relationship_plan.pair_decisions
            if relationship_plan is not None
            else ()
        )
    )
    fallbacks: list[FallbackDiagnostics] = []
    for detected in raw_objects:
        if detected.completion_failure_stage is not None:
            fallbacks.append(
                FallbackDiagnostics(
                    object_id=detected.object_id,
                    stage=f"completion.{detected.completion_failure_stage}",
                    reason=detected.completion_failure_reason or "unknown",
                )
            )
        if detected.reconstruction_failure_stage is not None:
            fallbacks.append(
                FallbackDiagnostics(
                    object_id=detected.object_id,
                    stage=(
                        "reconstruction."
                        f"{detected.reconstruction_failure_stage}"
                    ),
                    reason=detected.reconstruction_failure_reason or "unknown",
                )
            )

    timings = {
        stage: float(stage_timings_ms.get(stage, 0.0))
        for stage in REQUIRED_TIMING_STAGES
    }
    relationship_decisions = (
        relationship_plan.pair_decisions
        if relationship_plan is not None
        else ()
    )
    return PipelineDiagnostics(
        raw_object_count=len(raw_objects),
        final_group_count=len(final_groups),
        potential_overlap_pair_count=len(potential_pairs),
        retained_amodal_pair_count=len(retained_pairs),
        completion_candidate_count=completion_candidate_count,
        objects=object_records,
        pair_decisions=pair_records,
        groups=group_records,
        fallbacks=tuple(fallbacks),
        stage_timings_ms=timings,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        same_class_merge_count=sum(
            decision.relation is PairRelation.SAME_CLASS_MERGE
            for decision in relationship_decisions
        ),
        cross_class_containment_merge_count=sum(
            decision.relation
            is PairRelation.CROSS_CLASS_CONTAINMENT_MERGE
            for decision in relationship_decisions
        ),
        regular_overlap_count=(
            len(relationship_plan.regular_overlap_pairs)
            if relationship_plan is not None
            else 0
        ),
        external_overlap_count=(
            len(relationship_plan.external_overlap_pairs)
            if relationship_plan is not None
            else 0
        ),
        large_mask_veto_count=sum(
            metrics.large_mask_veto
            for metrics in relationship_metrics.values()
        ),
        large_bbox_veto_count=sum(
            metrics.large_bbox_veto
            for metrics in relationship_metrics.values()
        ),
        relationship_decisions=relationship_records,
    )


def log_pipeline_diagnostics(diagnostics: PipelineDiagnostics) -> None:
    """Emit a compact INFO summary and retain full records at DEBUG."""
    logger.info(
        "[PIPELINE] SUMMARY objects=%d groups=%d pairs=%d retained=%d "
        "completion_candidates=%d fallbacks=%d peak_gpu_bytes=%s timings_ms=%s",
        diagnostics.raw_object_count,
        diagnostics.final_group_count,
        diagnostics.potential_overlap_pair_count,
        diagnostics.retained_amodal_pair_count,
        diagnostics.completion_candidate_count,
        len(diagnostics.fallbacks),
        diagnostics.peak_gpu_memory_bytes,
        ",".join(
            f"{stage}:{duration:.1f}"
            for stage, duration in diagnostics.stage_timings_ms.items()
        ),
    )
    if logger.isEnabledFor(10):  # stdlib DEBUG
        logger.debug("[PIPELINE] DETAILS %r", diagnostics.to_log_dict())
