"""Tests for cross-class overlap detection and pairwise role assignment."""

import pytest

from backend.core import occlusion
from backend.core.occlusion import (
    ObjectBounds,
    bbox_area,
    bbox_containment_metrics,
    bbox_intersection_area,
    effective_hole_area,
    find_cross_class_overlaps,
)


def _object(object_id, semantic_class, bbox):
    return ObjectBounds(
        object_id=object_id,
        semantic_class=semantic_class,
        bbox=bbox,
    )


def test_bbox_containment_metrics_use_smaller_box_denominator():
    metrics = bbox_containment_metrics(
        (0, 0, 100, 100),
        (20, 20, 20, 30),
    )

    assert metrics is not None
    assert metrics.smaller_index == 1
    assert metrics.larger_index == 0
    assert metrics.intersection_area == 600
    assert metrics.smaller_area == 600
    assert metrics.larger_area == 10_000
    assert metrics.containment_ratio == pytest.approx(1.0)
    assert metrics.bbox_size_ratio == pytest.approx(0.06)


def test_bbox_intersection_requires_positive_area():
    assert bbox_intersection_area((0, 0, 10, 10), (10, 0, 5, 5)) == 0
    assert bbox_intersection_area((0, 0, 10, 10), (10, 10, 5, 5)) == 0


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 0, 10),
        (0, 0, 10, 0),
        (0, 0, -1, 10),
        (0, 0, 10, -1),
    ],
)
def test_invalid_bbox_has_zero_area_and_no_containment_metrics(bbox):
    assert bbox_area(bbox) == 0
    assert bbox_containment_metrics(bbox, (0, 0, 20, 20)) is None


def test_equal_area_boxes_have_stable_input_order():
    metrics = bbox_containment_metrics((0, 0, 10, 10), (1, 1, 10, 10))

    assert metrics is not None
    assert metrics.smaller_index == 0
    assert metrics.larger_index == 1
    assert metrics.containment_ratio == pytest.approx(0.81)
    assert metrics.bbox_size_ratio == pytest.approx(1.0)


def test_returns_overlapping_pair_from_different_classes():
    objects = [
        _object("person-1", "person", (0, 0, 10, 10)),
        _object("chair-1", "chair", (5, 5, 10, 10)),
    ]

    assert find_cross_class_overlaps(objects) == [("person-1", "chair-1")]


def test_ignores_non_overlapping_boxes():
    objects = [
        _object("person-1", "person", (0, 0, 4, 4)),
        _object("chair-1", "chair", (5, 5, 4, 4)),
    ]

    assert find_cross_class_overlaps(objects) == []


def test_ignores_overlapping_boxes_from_same_class():
    objects = [
        _object("person-1", "person", (0, 0, 10, 10)),
        _object("person-2", "person", (5, 5, 10, 10)),
    ]

    assert find_cross_class_overlaps(objects) == []


def test_ignores_edge_and_corner_contact():
    objects = [
        _object("base", "person", (0, 0, 4, 4)),
        _object("edge", "chair", (4, 1, 3, 2)),
        _object("corner", "table", (4, 4, 3, 3)),
    ]

    assert find_cross_class_overlaps(objects) == []


def test_returns_each_pair_once_in_input_order():
    objects = [
        _object("person-1", "person", (0, 0, 10, 10)),
        _object("chair-1", "chair", (1, 1, 3, 3)),
        _object("table-1", "table", (2, 2, 3, 3)),
    ]

    assert find_cross_class_overlaps(objects) == [
        ("person-1", "chair-1"),
        ("person-1", "table-1"),
        ("chair-1", "table-1"),
    ]


def test_empty_input_returns_no_pairs():
    assert find_cross_class_overlaps([]) == []


def test_hole_below_absolute_noise_floor_has_zero_effective_area():
    assert effective_hole_area(
        raw_area=15,
        modal_area=1_000,
        minimum_pixels=16,
        minimum_modal_ratio=0.01,
    ) == 0


def test_hole_below_modal_relative_noise_floor_has_zero_effective_area():
    assert effective_hole_area(
        raw_area=30,
        modal_area=4_000,
        minimum_pixels=16,
        minimum_modal_ratio=0.01,
    ) == 0


def test_meaningful_hole_preserves_raw_area_as_effective_area():
    assert effective_hole_area(
        raw_area=40,
        modal_area=4_000,
        minimum_pixels=16,
        minimum_modal_ratio=0.01,
    ) == 40


def test_larger_first_hole_assigns_first_object_as_occluded():
    decisions = occlusion.assign_pair_roles(
        [("person-1", "chair-1")],
        {"person-1": 12, "chair-1": 4},
    )

    decision = decisions[0]
    assert decision.occluded_id == "person-1"
    assert decision.occluder_id == "chair-1"
    assert decision.ambiguous is False


def test_larger_second_hole_assigns_second_object_as_occluded():
    decisions = occlusion.assign_pair_roles(
        [("person-1", "chair-1")],
        {"person-1": 3, "chair-1": 9},
    )

    decision = decisions[0]
    assert decision.occluded_id == "chair-1"
    assert decision.occluder_id == "person-1"
    assert decision.ambiguous is False


def test_equal_holes_create_ambiguous_decision_without_roles():
    decisions = occlusion.assign_pair_roles(
        [("person-1", "chair-1")],
        {"person-1": 5, "chair-1": 5},
    )

    decision = decisions[0]
    assert decision.occluded_id is None
    assert decision.occluder_id is None
    assert decision.ambiguous is True


def test_near_equal_holes_within_relative_tolerance_are_ambiguous():
    decisions = occlusion.assign_pair_roles(
        [("person-1", "chair-1")],
        {"person-1": 100, "chair-1": 91},
        tie_tolerance_ratio=0.1,
    )

    decision = decisions[0]
    assert decision.occluded_id is None
    assert decision.occluder_id is None
    assert decision.ambiguous is True


def test_hole_difference_above_relative_tolerance_assigns_roles():
    decisions = occlusion.assign_pair_roles(
        [("person-1", "chair-1")],
        {"person-1": 100, "chair-1": 89},
        tie_tolerance_ratio=0.1,
    )

    decision = decisions[0]
    assert decision.occluded_id == "person-1"
    assert decision.occluder_id == "chair-1"


def test_multi_pair_roles_are_independent_and_keep_pair_order():
    decisions = occlusion.assign_pair_roles(
        [("person-1", "chair-1"), ("person-1", "table-1")],
        {"person-1": 10, "chair-1": 4, "table-1": 20},
    )

    assert [
        (decision.occluded_id, decision.occluder_id)
        for decision in decisions
    ] == [
        ("person-1", "chair-1"),
        ("table-1", "person-1"),
    ]


def test_directional_roles_allow_both_objects_to_be_reconstructed():
    decisions = occlusion.assign_directional_pair_roles(
        [("person", "book")],
        {
            ("person", "book"): 12,
            ("book", "person"): 5,
        },
        tie_tolerance_ratio=0.1,
    )

    decision = decisions[0]
    assert decision.reconstruction_directions == (
        ("person", "book"),
        ("book", "person"),
    )
    assert decision.bidirectional is True
    assert decision.occluded_id == "person"
    assert decision.occluder_id == "book"


def test_directional_roles_skip_pair_without_directional_overlap():
    decision = occlusion.assign_directional_pair_roles(
        [("person", "book")],
        {("person", "book"): 0, ("book", "person"): 0},
    )[0]

    assert decision.reconstruction_directions == ()
    assert decision.ambiguous is True
