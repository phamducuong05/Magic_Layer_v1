"""Structured, pixel-free observability for the image pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
import json
import logging
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ..core.occlusion import OverlapPair, PairDecision
from .types import DetectedObject, GroupedObject


logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class PairDecisionDiagnostics:
    first_id: str
    second_id: str
    occluded_id: str | None
    occluder_id: str | None
    ambiguous_reason: str | None


@dataclass(frozen=True)
class GroupDiagnostics:
    group_id: str
    member_ids: tuple[str, ...]
    reconstruction_conflicts: tuple[tuple[str, str], ...]


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
) -> PipelineDiagnostics:
    """Build immutable diagnostic metadata from final pipeline state."""
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
        )
        for decision in pair_decisions
    )
    group_records = tuple(
        GroupDiagnostics(
            group_id=group.group_id,
            member_ids=group.member_ids,
            reconstruction_conflicts=group.reconstruction_conflicts,
        )
        for group in final_groups
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
    )


def log_pipeline_diagnostics(diagnostics: PipelineDiagnostics) -> None:
    """Emit one structured summary without pixel or feature payloads."""
    logger.info(
        "Pipeline diagnostics: %s",
        json.dumps(diagnostics.to_log_dict(), sort_keys=True, default=str),
    )
