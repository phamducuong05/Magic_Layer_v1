"""Tests for raw-object relationship planning."""

import numpy as np
import pytest

from backend.pipeline.relationships import (
    PairRelation,
    plan_object_relationships,
)
from backend.pipeline.types import DetectedObject
from backend.pipeline.types import MergeEdge


def _object(
    object_id: str,
    semantic_class: str,
    bbox: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, int] = (100, 100),
    mask_bbox: tuple[int, int, int, int] | None = None,
) -> DetectedObject:
    mask = np.zeros(image_shape, dtype=np.uint8)
    x, y, width, height = mask_bbox or bbox
    mask[y : y + height, x : x + width] = 255
    return DetectedObject(
        object_id=object_id,
        semantic_class=semantic_class,
        display_label=semantic_class,
        modal_mask=mask,
        bbox=bbox,
    )


SETTINGS = {
    "enabled": True,
    "containment_threshold": 0.70,
    "max_bbox_size_ratio": 0.50,
    "large_mask_ratio": 0.45,
    "large_bbox_ratio": 0.75,
    "dimension_tolerance_ratio": 0.05,
}


def test_merge_edge_canonicalizes_member_identity():
    edge = MergeEdge(
        first_id="b",
        second_id="a",
        reason="cross_class_bbox_containment",
        containment_ratio=0.8,
        bbox_size_ratio=0.3,
    )

    assert edge.member_ids == ("a", "b")


def test_cross_class_containment_creates_merge_edge_not_overlap():
    large = _object("large", "table", (0, 0, 60, 60))
    small = _object("small", "product", (10, 10, 20, 20))

    plan = plan_object_relationships(
        [large, small],
        image_size=(100, 100),
        **SETTINGS,
    )

    assert [edge.member_ids for edge in plan.merge_edges] == [
        ("large", "small")
    ]
    assert plan.regular_overlap_pairs == ()
    assert plan.external_overlap_pairs == ()
    assert plan.pair_decisions[0].relation is (
        PairRelation.CROSS_CLASS_CONTAINMENT_MERGE
    )


def test_similar_area_overlap_remains_regular_overlap():
    first = _object("first", "person", (0, 0, 50, 50))
    second = _object("second", "car", (5, 5, 50, 50))

    plan = plan_object_relationships(
        [first, second],
        image_size=(100, 100),
        **SETTINGS,
    )

    assert plan.merge_edges == ()
    assert plan.regular_overlap_pairs == (("first", "second"),)
    assert plan.pair_decisions[0].reason == "bbox_size_ratio_not_dominant"


def test_axis_incompatible_boxes_do_not_merge():
    horizontal = _object("wide", "shelf", (0, 0, 80, 20))
    vertical = _object("tall", "product", (10, 0, 20, 25))

    plan = plan_object_relationships(
        [horizontal, vertical],
        image_size=(100, 100),
        **SETTINGS,
    )

    assert plan.merge_edges == ()
    assert plan.external_overlap_pairs == (("wide", "tall"),)
    assert plan.pair_decisions[0].reason == (
        "bbox_dimensions_not_dominant"
    )


def test_mask_ratio_veto_blocks_only_cross_class_containment():
    large_mask = _object("large", "table", (0, 0, 70, 70))
    contained = _object("small", "product", (10, 10, 20, 20))
    same_class = _object("same", "table", (5, 5, 20, 20))

    plan = plan_object_relationships(
        [large_mask, contained, same_class],
        image_size=(100, 100),
        **SETTINGS,
    )

    reasons = {edge.reason for edge in plan.merge_edges}
    assert "same_class_bbox_overlap" in reasons
    assert not any(
        edge.member_ids == ("large", "small")
        for edge in plan.merge_edges
    )
    assert ("large", "small") in plan.regular_overlap_pairs
    large_small = next(
        decision
        for decision in plan.pair_decisions
        if {decision.first_id, decision.second_id} == {"large", "small"}
    )
    assert large_small.reason == "large_mask_ratio_veto"


def test_bbox_ratio_veto_catches_sparse_large_extent():
    sparse = _object(
        "sparse",
        "frame",
        (0, 0, 90, 90),
        mask_bbox=(0, 0, 5, 5),
    )
    small = _object("small", "product", (10, 10, 20, 20))

    plan = plan_object_relationships(
        [sparse, small],
        image_size=(100, 100),
        **SETTINGS,
    )

    metrics = plan.object_metrics["sparse"]
    assert metrics.mask_image_ratio == 0.0025
    assert metrics.bbox_image_ratio == 0.81
    assert metrics.large_bbox_veto is True
    assert plan.merge_edges == ()
    assert plan.external_overlap_pairs == (("sparse", "small"),)
    assert plan.pair_decisions[0].reason == "large_bbox_ratio_veto"


def test_mask_veto_takes_precedence_when_both_large_vetoes_apply():
    large = _object("large", "frame", (0, 0, 90, 90))
    small = _object("small", "product", (10, 10, 20, 20))

    plan = plan_object_relationships(
        [large, small],
        image_size=(100, 100),
        **SETTINGS,
    )

    assert plan.object_metrics["large"].large_mask_veto is True
    assert plan.object_metrics["large"].large_bbox_veto is True
    assert plan.pair_decisions[0].reason == "large_mask_ratio_veto"


def test_transitive_merge_suppresses_regular_internal_edge():
    first = _object("a", "class-a", (0, 0, 40, 40))
    middle = _object("b", "class-b", (26, 5, 20, 20))
    third = _object("c", "class-c", (39, 8, 10, 10))

    plan = plan_object_relationships(
        [first, middle, third],
        image_size=(100, 100),
        **SETTINGS,
    )

    assert {
        edge.member_ids for edge in plan.merge_edges
    } == {("a", "b"), ("b", "c")}
    assert len(set(plan.component_by_object_id.values())) == 1
    assert plan.external_overlap_pairs == ()
    suppressed = next(
        decision
        for decision in plan.pair_decisions
        if {decision.first_id, decision.second_id} == {"a", "c"}
    )
    assert suppressed.reason == "internal_pair_suppressed"


def test_disabled_feature_reproduces_existing_cross_class_overlap():
    large = _object("large", "table", (0, 0, 60, 60))
    small = _object("small", "product", (10, 10, 20, 20))

    plan = plan_object_relationships(
        [large, small],
        image_size=(100, 100),
        **{**SETTINGS, "enabled": False},
    )

    assert plan.merge_edges == ()
    assert plan.external_overlap_pairs == (("large", "small"),)


def test_relationship_plan_rejects_duplicate_ids():
    first = _object("duplicate", "person", (0, 0, 20, 20))
    second = _object("duplicate", "chair", (5, 5, 5, 5))

    with pytest.raises(ValueError, match="duplicate object_id"):
        plan_object_relationships(
            [first, second],
            image_size=(100, 100),
            **SETTINGS,
        )


def test_relationship_plan_rejects_mask_shape_mismatch():
    detected = _object(
        "object",
        "person",
        (0, 0, 20, 20),
        image_shape=(50, 50),
    )

    with pytest.raises(ValueError, match="modal mask shape"):
        plan_object_relationships(
            [detected],
            image_size=(100, 100),
            **SETTINGS,
        )


def test_relationship_plan_validates_threshold_ranges():
    with pytest.raises(ValueError, match="containment_threshold"):
        plan_object_relationships(
            [],
            image_size=(100, 100),
            **{**SETTINGS, "containment_threshold": 1.1},
        )


def test_cross_class_bidirectional_intertwined_merge():
    man = _object("man", "man", (10, 10, 50, 80))
    woman = _object("woman", "woman", (20, 10, 50, 80))
    man.amodal_mask = man.modal_mask.copy()
    woman.amodal_mask = woman.modal_mask.copy()
    man.completion_hole_mask = np.zeros((100, 100), dtype=bool)
    man.completion_hole_mask[20:30, 25:35] = True
    woman.completion_hole_mask = np.zeros((100, 100), dtype=bool)
    woman.completion_hole_mask[40:50, 20:30] = True

    plan = plan_object_relationships(
        [man, woman],
        image_size=(100, 100),
        **SETTINGS,
    )

    assert len(plan.merge_edges) == 1
    assert plan.merge_edges[0].reason == "cross_class_bidirectional_intertwined"
    assert plan.pair_decisions[0].relation is (
        PairRelation.CROSS_CLASS_CONTAINMENT_MERGE
    )


def test_cross_class_mutual_bbox_containment_merge():
    man = _object("man", "man", (10, 10, 50, 80))
    woman = _object("woman", "woman", (20, 10, 50, 80))

    plan = plan_object_relationships(
        [man, woman],
        image_size=(100, 100),
        **{**SETTINGS, "mutual_containment_threshold": 0.40},
    )

    assert len(plan.merge_edges) == 1
    assert plan.merge_edges[0].reason == "cross_class_mutual_bbox_containment"
    assert plan.pair_decisions[0].relation is (
        PairRelation.CROSS_CLASS_CONTAINMENT_MERGE
    )
