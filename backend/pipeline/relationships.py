"""Classify raw-object relationships before heavy pipeline stages."""

from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..core.occlusion import (
    OverlapPair,
    bbox_area,
    bbox_containment_metrics,
    bbox_intersection_area,
)
from .types import DetectedObject, MergeEdge


class PairRelation(str, Enum):
    """Exhaustive relationship classes for a raw-object pair."""

    SAME_CLASS_MERGE = "same_class_merge"
    CROSS_CLASS_CONTAINMENT_MERGE = "cross_class_containment_merge"
    REGULAR_CROSS_CLASS_OVERLAP = "regular_cross_class_overlap"
    DISJOINT = "disjoint"


@dataclass(frozen=True)
class ObjectRelationshipMetrics:
    """Image-coverage metrics computed once per raw object."""

    bbox_area: int
    mask_area: int
    bbox_image_ratio: float
    mask_image_ratio: float
    large_mask_veto: bool
    large_bbox_veto: bool

    @property
    def cross_class_merge_veto(self) -> bool:
        """Return whether this object is too large for cross-class merging."""
        return self.large_mask_veto or self.large_bbox_veto


@dataclass(frozen=True)
class PairRelationshipDecision:
    """Auditable classification result for one raw-object pair."""

    first_id: str
    second_id: str
    relation: PairRelation
    reason: str
    containment_ratio: float | None = None
    bbox_size_ratio: float | None = None


@dataclass(frozen=True)
class RelationshipPlan:
    """Immutable pair plan consumed by completion and final grouping."""

    merge_edges: tuple[MergeEdge, ...]
    regular_overlap_pairs: tuple[OverlapPair, ...]
    external_overlap_pairs: tuple[OverlapPair, ...]
    component_by_object_id: Mapping[str, int]
    object_metrics: Mapping[str, ObjectRelationshipMetrics]
    pair_decisions: tuple[PairRelationshipDecision, ...]


def _validate_ratio(name: str, value: float, *, allow_zero: bool) -> None:
    lower_bound_valid = value >= 0.0 if allow_zero else value > 0.0
    if not lower_bound_valid or value > 1.0:
        comparator = "0.0 <= value <= 1.0" if allow_zero else (
            "0.0 < value <= 1.0"
        )
        raise ValueError(f"{name} must satisfy {comparator}")


def _validate_inputs(
    objects: Sequence[DetectedObject],
    image_size: tuple[int, int],
    *,
    containment_threshold: float,
    max_bbox_size_ratio: float,
    large_mask_ratio: float,
    large_bbox_ratio: float,
    dimension_tolerance_ratio: float,
    mutual_containment_threshold: float | None = None,
) -> None:
    _validate_ratio(
        "containment_threshold",
        containment_threshold,
        allow_zero=True,
    )
    _validate_ratio(
        "max_bbox_size_ratio",
        max_bbox_size_ratio,
        allow_zero=False,
    )
    _validate_ratio("large_mask_ratio", large_mask_ratio, allow_zero=True)
    _validate_ratio("large_bbox_ratio", large_bbox_ratio, allow_zero=True)
    if mutual_containment_threshold is not None:
        _validate_ratio(
            "mutual_containment_threshold",
            mutual_containment_threshold,
            allow_zero=True,
        )
    if not 0.0 <= dimension_tolerance_ratio < 1.0:
        raise ValueError(
            "dimension_tolerance_ratio must satisfy 0.0 <= value < 1.0"
        )

    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_size dimensions must be positive")

    object_ids = [detected.object_id for detected in objects]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("duplicate object_id values are not allowed")

    expected_shape = (image_height, image_width)
    for detected in objects:
        if detected.modal_mask.shape != expected_shape:
            raise ValueError(
                f"modal mask shape for {detected.object_id!r} must be "
                f"{expected_shape}, got {detected.modal_mask.shape}"
            )


def _build_object_metrics(
    objects: Sequence[DetectedObject],
    image_size: tuple[int, int],
    *,
    large_mask_ratio: float,
    large_bbox_ratio: float,
) -> dict[str, ObjectRelationshipMetrics]:
    image_width, image_height = image_size
    image_area = image_width * image_height
    metrics: dict[str, ObjectRelationshipMetrics] = {}
    for detected in objects:
        object_bbox_area = bbox_area(detected.original_modal_bbox)
        mask_area = int(np.count_nonzero(detected.modal_mask))
        bbox_image_ratio = object_bbox_area / image_area
        mask_image_ratio = mask_area / image_area
        metrics[detected.object_id] = ObjectRelationshipMetrics(
            bbox_area=object_bbox_area,
            mask_area=mask_area,
            bbox_image_ratio=bbox_image_ratio,
            mask_image_ratio=mask_image_ratio,
            large_mask_veto=mask_image_ratio >= large_mask_ratio,
            large_bbox_veto=bbox_image_ratio >= large_bbox_ratio,
        )
    return metrics


def plan_object_relationships(
    objects: Sequence[DetectedObject],
    image_size: tuple[int, int],
    *,
    enabled: bool,
    containment_threshold: float,
    max_bbox_size_ratio: float,
    large_mask_ratio: float,
    large_bbox_ratio: float,
    dimension_tolerance_ratio: float,
    mutual_containment_threshold: float | None = None,
    enable_bidirectional_merge: bool = True,
) -> RelationshipPlan:
    """Plan merge intent and external overlaps from immutable raw geometry."""
    _validate_inputs(
        objects,
        image_size,
        containment_threshold=containment_threshold,
        max_bbox_size_ratio=max_bbox_size_ratio,
        large_mask_ratio=large_mask_ratio,
        large_bbox_ratio=large_bbox_ratio,
        dimension_tolerance_ratio=dimension_tolerance_ratio,
        mutual_containment_threshold=mutual_containment_threshold,
    )
    object_metrics = _build_object_metrics(
        objects,
        image_size,
        large_mask_ratio=large_mask_ratio,
        large_bbox_ratio=large_bbox_ratio,
    )

    merge_edges: list[MergeEdge] = []
    regular_pairs: list[OverlapPair] = []
    decisions: list[PairRelationshipDecision] = []

    for first, second in combinations(objects, 2):
        pair = (first.object_id, second.object_id)
        first_bbox = first.original_modal_bbox
        second_bbox = second.original_modal_bbox
        intersection_area = bbox_intersection_area(first_bbox, second_bbox)

        if first.semantic_class == second.semantic_class:
            if intersection_area > 0:
                merge_edges.append(
                    MergeEdge(
                        first.object_id,
                        second.object_id,
                        "same_class_bbox_overlap",
                    )
                )
                relation = PairRelation.SAME_CLASS_MERGE
                reason = "same_class_bbox_overlap"
            else:
                relation = PairRelation.DISJOINT
                reason = "original_modal_bboxes_do_not_overlap"
            decisions.append(
                PairRelationshipDecision(
                    first.object_id,
                    second.object_id,
                    relation,
                    reason,
                )
            )
            continue

        if intersection_area <= 0:
            decisions.append(
                PairRelationshipDecision(
                    first.object_id,
                    second.object_id,
                    PairRelation.DISJOINT,
                    "no_positive_bbox_overlap",
                )
            )
            continue

        containment = bbox_containment_metrics(first_bbox, second_bbox)
        containment_ratio = (
            containment.containment_ratio if containment is not None else None
        )
        bbox_size_ratio = (
            containment.bbox_size_ratio if containment is not None else None
        )

        merge_reason: str | None = None
        if not enabled:
            failure_reason = "cross_class_grouping_disabled"
        elif containment is None:
            failure_reason = "invalid_bbox"
        elif (
            object_metrics[first.object_id].large_mask_veto
            or object_metrics[second.object_id].large_mask_veto
        ):
            # Mask coverage is the primary signal when both vetoes apply.
            failure_reason = "large_mask_ratio_veto"
        elif (
            object_metrics[first.object_id].large_bbox_veto
            or object_metrics[second.object_id].large_bbox_veto
        ):
            failure_reason = "large_bbox_ratio_veto"
        else:
            # Check Strategy 1: Bidirectional Occlusion (Pixel-level)
            first_hole = (
                first.completion_hole_mask > 0
                if first.completion_hole_mask is not None
                else (first.amodal_mask > 0) & ~(first.modal_mask > 0)
                if first.amodal_mask is not None
                else None
            )
            second_hole = (
                second.completion_hole_mask > 0
                if second.completion_hole_mask is not None
                else (second.amodal_mask > 0) & ~(second.modal_mask > 0)
                if second.amodal_mask is not None
                else None
            )
            is_bidirectional = False
            if (
                enable_bidirectional_merge
                and first_hole is not None
                and second_hole is not None
            ):
                first_on_second = np.count_nonzero(
                    first_hole & (second.modal_mask > 0)
                )
                second_on_first = np.count_nonzero(
                    second_hole & (first.modal_mask > 0)
                )
                is_bidirectional = first_on_second > 0 and second_on_first > 0

            # Check Strategy 2: Mutual BBox Containment (BBox-level)
            first_area = bbox_area(first_bbox)
            second_area = bbox_area(second_bbox)
            containment_first_in_second = (
                intersection_area / first_area if first_area > 0 else 0.0
            )
            containment_second_in_first = (
                intersection_area / second_area if second_area > 0 else 0.0
            )
            is_mutual_containment = (
                mutual_containment_threshold is not None
                and containment_first_in_second >= mutual_containment_threshold
                and containment_second_in_first >= mutual_containment_threshold
            )

            if is_bidirectional:
                merge_reason = "cross_class_bidirectional_intertwined"
                failure_reason = ""
            elif is_mutual_containment:
                merge_reason = "cross_class_mutual_bbox_containment"
                failure_reason = ""
            elif containment.containment_ratio < containment_threshold:
                failure_reason = "containment_ratio_below_threshold"
            elif containment.bbox_size_ratio > max_bbox_size_ratio:
                failure_reason = "bbox_size_ratio_not_dominant"
            else:
                boxes = (first_bbox, second_bbox)
                small_bbox = boxes[containment.smaller_index]
                large_bbox = boxes[containment.larger_index]
                _, _, small_width, small_height = small_bbox
                _, _, large_width, large_height = large_bbox
                minimum_scale = 1.0 - dimension_tolerance_ratio
                dimensions_dominant = (
                    large_width >= small_width * minimum_scale
                    and large_height >= small_height * minimum_scale
                )
                if not dimensions_dominant:
                    failure_reason = "bbox_dimensions_not_dominant"
                else:
                    merge_reason = "cross_class_bbox_containment"
                    failure_reason = ""

        if merge_reason is not None:
            merge_edges.append(
                MergeEdge(
                    first.object_id,
                    second.object_id,
                    merge_reason,
                    containment_ratio=containment_ratio,
                    bbox_size_ratio=bbox_size_ratio,
                )
            )
            relation = PairRelation.CROSS_CLASS_CONTAINMENT_MERGE
            reason = merge_reason
        else:
            regular_pairs.append(pair)
            relation = PairRelation.REGULAR_CROSS_CLASS_OVERLAP
            reason = failure_reason

        decisions.append(
            PairRelationshipDecision(
                first.object_id,
                second.object_id,
                relation,
                reason,
                containment_ratio=containment_ratio,
                bbox_size_ratio=bbox_size_ratio,
            )
        )

    parents = list(range(len(objects)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root == second_root:
            return
        lower_root, higher_root = sorted((first_root, second_root))
        parents[higher_root] = lower_root

    index_by_id = {
        detected.object_id: index for index, detected in enumerate(objects)
    }
    for edge in merge_edges:
        union(index_by_id[edge.first_id], index_by_id[edge.second_id])

    component_by_object_id = {
        detected.object_id: find(index)
        for index, detected in enumerate(objects)
    }
    external_pairs = tuple(
        pair
        for pair in regular_pairs
        if component_by_object_id[pair[0]] != component_by_object_id[pair[1]]
    )
    external_pair_keys = {frozenset(pair) for pair in external_pairs}
    decisions = [
        replace(decision, reason="internal_pair_suppressed")
        if (
            decision.relation is PairRelation.REGULAR_CROSS_CLASS_OVERLAP
            and frozenset((decision.first_id, decision.second_id))
            not in external_pair_keys
        )
        else decision
        for decision in decisions
    ]
    return RelationshipPlan(
        merge_edges=tuple(merge_edges),
        regular_overlap_pairs=tuple(regular_pairs),
        external_overlap_pairs=external_pairs,
        component_by_object_id=MappingProxyType(component_by_object_id),
        object_metrics=MappingProxyType(object_metrics),
        pair_decisions=tuple(decisions),
    )
