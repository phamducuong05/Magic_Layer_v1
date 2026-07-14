"""Tests for cross-class bounding-box overlap detection."""

from backend.core.occlusion import ObjectBounds, find_cross_class_overlaps


def _object(object_id, semantic_class, bbox):
    return ObjectBounds(
        object_id=object_id,
        semantic_class=semantic_class,
        bbox=bbox,
    )


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
