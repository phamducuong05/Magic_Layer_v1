# Cross-Class Containment Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable cross-class containment grouping that suppresses only internal occlusion relationships while preserving member-level completion, reconstruction, existing same-class grouping, and all downstream contracts.

**Architecture:** Compute immutable raw-object geometry and classify every pair immediately after segmentation. Store validated same-class and cross-class merge edges as grouping intent, filter regular overlaps by connected-component membership, and continue running the existing heavy workflow on raw external pairs. Materialize final `GroupedObject` instances only after member-level completion, reconstruction, and support refinement.

**Tech Stack:** Python 3.10+, NumPy, Pillow, PyYAML, pytest, existing dataclasses and union-find patterns.

## Global Constraints

- Bounding boxes use `(x, y, width, height)`.
- `containment_ratio = intersection_area / smaller_bbox_area`.
- `bbox_size_ratio = smaller_bbox_area / larger_bbox_area`.
- Initial thresholds are `containment_threshold=0.70`, `max_bbox_size_ratio=0.50`, `large_mask_ratio=0.45`, `large_bbox_ratio=0.75`, and `dimension_tolerance_ratio=0.05`.
- Large-object veto uses `mask_image_ratio >= large_mask_ratio OR bbox_image_ratio >= large_bbox_ratio`.
- Large-object veto applies only to cross-class containment merges.
- Existing same-class grouping behavior and stable primary-member selection remain unchanged.
- Merge transitivity uses validated raw-object edges only; never generate an edge from a union bbox.
- Regular overlap edges inside one transitive merge component are suppressed.
- Completion and reconstruction continue to operate on raw members and raw external pairs.
- Merge intent does not suppress an independently valid external pair; a member is excluded only when it has no raw external pair of its own.
- Reconstruction remains member-specific: when C occludes B, only B receives B–C reconstruction state.
- `process_masks()` continues returning one mask per raw object in raw-object order.
- Do not change completion validation, depth ordering, reconstruction-mask generation, reconstruction inference, support refinement, composition priority, matting, layer extraction, background generation, or `ProcessResult`.
- Keep all new geometry and relationship planning deterministic and free of model-manager dependencies.

---

## File Map

### New file

- `backend/pipeline/relationships.py` — classify raw-object pairs, compute coverage vetoes, build merge-intent components, and expose external overlap pairs.
- `tests/test_relationships.py` — focused tests for pair classification, veto behavior, transitivity, and feature disablement.

### Modified files

- `backend/core/occlusion.py` — pure bounding-box area, intersection, and containment metric utilities.
- `backend/pipeline/types.py` — shared immutable `MergeEdge` provenance and multi-class group metadata.
- `backend/pipeline/completion/completion.py` — link overlap partners from an explicit prefiltered pair list while retaining the current fallback API.
- `backend/pipeline/grouping.py` — consume explicit merge edges and materialize cross-class groups without changing legacy same-class behavior.
- `backend/pipeline/orchestrator.py` — invoke relationship planning before completion in both public workflows and pass merge edges into final grouping.
- `backend/pipeline/diagnostics.py` — expose relationship metrics and counts without pixel payloads.
- `backend/config.yaml` — add the approved cross-class grouping settings.
- `docs/image_to_components_pipeline.md` — document the new relationship-planning stage and member-level external occlusion rule.
- `tests/test_occlusion.py` — geometry utility tests.
- `tests/test_pipeline_architecture.py` — explicit-pair completion linkage and `process_masks()` non-regression.
- `tests/test_grouping.py` — explicit merge-edge materialization, primary identity, and A–B–C orchestrator behavior.
- `tests/test_pipeline_diagnostics.py` — relationship diagnostic serialization.

---

### Task 1: Bounding-Box Containment Geometry

**Files:**
- Modify: `backend/core/occlusion.py:8-92`
- Modify: `tests/test_occlusion.py:1-73`

**Interfaces:**
- Consumes: `BoundingBox = tuple[int, int, int, int]`
- Produces:
  - `BBoxContainmentMetrics`
  - `bbox_area(bbox: BoundingBox) -> int`
  - `bbox_intersection_area(first: BoundingBox, second: BoundingBox) -> int`
  - `bbox_containment_metrics(first: BoundingBox, second: BoundingBox) -> BBoxContainmentMetrics | None`

- [ ] **Step 1: Write failing tests for containment math**

Add to `tests/test_occlusion.py`:

```python
import pytest

from backend.core.occlusion import (
    bbox_area,
    bbox_containment_metrics,
    bbox_intersection_area,
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
```

- [ ] **Step 2: Run the geometry tests and verify RED**

Run:

```powershell
pytest tests/test_occlusion.py -k "bbox_containment or bbox_intersection or invalid_bbox or equal_area" -v
```

Expected: collection fails because the three new functions and `BBoxContainmentMetrics` do not exist.

- [ ] **Step 3: Implement the pure geometry utilities**

Add above `find_cross_class_overlaps()` in `backend/core/occlusion.py`:

```python
@dataclass(frozen=True)
class BBoxContainmentMetrics:
    """Symmetric containment metrics with stable smaller/larger identities."""

    intersection_area: int
    smaller_area: int
    larger_area: int
    smaller_index: int
    larger_index: int
    containment_ratio: float
    bbox_size_ratio: float


def bbox_area(bbox: BoundingBox) -> int:
    """Return positive bbox area, or zero for invalid dimensions."""
    _, _, width, height = bbox
    if width <= 0 or height <= 0:
        return 0
    return int(width) * int(height)


def bbox_intersection_area(
    first: BoundingBox,
    second: BoundingBox,
) -> int:
    """Return positive intersection area for two xywh boxes."""
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    if bbox_area(first) == 0 or bbox_area(second) == 0:
        return 0
    width = min(
        first_x + first_width,
        second_x + second_width,
    ) - max(first_x, second_x)
    height = min(
        first_y + first_height,
        second_y + second_height,
    ) - max(first_y, second_y)
    if width <= 0 or height <= 0:
        return 0
    return int(width) * int(height)


def bbox_containment_metrics(
    first: BoundingBox,
    second: BoundingBox,
) -> BBoxContainmentMetrics | None:
    """Return containment metrics, or None when either box is invalid."""
    areas = (bbox_area(first), bbox_area(second))
    if min(areas) <= 0:
        return None
    smaller_index, larger_index = (
        (0, 1) if areas[0] <= areas[1] else (1, 0)
    )
    smaller_area = areas[smaller_index]
    larger_area = areas[larger_index]
    intersection_area = bbox_intersection_area(first, second)
    return BBoxContainmentMetrics(
        intersection_area=intersection_area,
        smaller_area=smaller_area,
        larger_area=larger_area,
        smaller_index=smaller_index,
        larger_index=larger_index,
        containment_ratio=intersection_area / smaller_area,
        bbox_size_ratio=smaller_area / larger_area,
    )
```

Refactor `find_cross_class_overlaps()` to call `bbox_intersection_area()` instead of maintaining duplicate intersection math. Preserve its input ordering and same-class skip exactly.

- [ ] **Step 4: Run focused and existing occlusion tests**

Run:

```powershell
pytest tests/test_occlusion.py -v
```

Expected: all tests pass, including the pre-existing cross-class overlap and role-assignment tests.

- [ ] **Step 5: Commit the geometry unit**

```powershell
git add backend/core/occlusion.py tests/test_occlusion.py
git commit -m "feat: add bbox containment metrics"
```

---

### Task 2: Relationship Planner and Large-Object Veto

**Files:**
- Create: `backend/pipeline/relationships.py`
- Create: `tests/test_relationships.py`
- Modify: `backend/pipeline/types.py:1-140`
- Modify: `backend/config.yaml:17-31`

**Interfaces:**
- Consumes:
  - `bbox_containment_metrics(...)`
  - `DetectedObject`
  - image size `(width, height)`
- Produces:
  - `MergeEdge`
  - `PairRelation`
  - `ObjectRelationshipMetrics`
  - `PairRelationshipDecision`
  - `RelationshipPlan`
  - `plan_object_relationships(...) -> RelationshipPlan`

- [ ] **Step 1: Add the shared immutable merge-edge type**

Write a failing import test at the top of `tests/test_relationships.py`:

```python
from backend.pipeline.types import MergeEdge


def test_merge_edge_canonicalizes_member_identity():
    edge = MergeEdge(
        first_id="b",
        second_id="a",
        reason="cross_class_bbox_containment",
        containment_ratio=0.8,
        bbox_size_ratio=0.3,
    )

    assert edge.member_ids == ("a", "b")
```

Run:

```powershell
pytest tests/test_relationships.py::test_merge_edge_canonicalizes_member_identity -v
```

Expected: FAIL because `MergeEdge` is not defined.

Add before `DetectedObject` in `backend/pipeline/types.py`:

```python
@dataclass(frozen=True)
class MergeEdge:
    first_id: str
    second_id: str
    reason: str
    containment_ratio: Optional[float] = None
    bbox_size_ratio: Optional[float] = None

    @property
    def member_ids(self) -> tuple[str, str]:
        return tuple(sorted((self.first_id, self.second_id)))
```

Run the focused test again and expect PASS.

- [ ] **Step 2: Write failing planner tests**

Add helpers and tests to `tests/test_relationships.py`:

```python
import numpy as np
import pytest

from backend.pipeline.relationships import (
    PairRelation,
    plan_object_relationships,
)
from backend.pipeline.types import DetectedObject


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
```

- [ ] **Step 3: Run planner tests and verify RED**

Run:

```powershell
pytest tests/test_relationships.py -v
```

Expected: the merge-edge test passes and planner tests fail because `relationships.py` and its interfaces do not exist.

- [ ] **Step 4: Implement planner types and validation**

Create `backend/pipeline/relationships.py` with these public interfaces:

```python
"""Raw-object relationship planning before completion and reconstruction."""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
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
    SAME_CLASS_MERGE = "same_class_merge"
    CROSS_CLASS_CONTAINMENT_MERGE = "cross_class_containment_merge"
    REGULAR_CROSS_CLASS_OVERLAP = "regular_cross_class_overlap"
    DISJOINT = "disjoint"


@dataclass(frozen=True)
class ObjectRelationshipMetrics:
    bbox_area: int
    mask_area: int
    bbox_image_ratio: float
    mask_image_ratio: float
    large_mask_veto: bool
    large_bbox_veto: bool

    @property
    def cross_class_merge_veto(self) -> bool:
        return self.large_mask_veto or self.large_bbox_veto


@dataclass(frozen=True)
class PairRelationshipDecision:
    first_id: str
    second_id: str
    relation: PairRelation
    reason: str
    containment_ratio: float | None = None
    bbox_size_ratio: float | None = None


@dataclass(frozen=True)
class RelationshipPlan:
    merge_edges: tuple[MergeEdge, ...]
    regular_overlap_pairs: tuple[OverlapPair, ...]
    external_overlap_pairs: tuple[OverlapPair, ...]
    component_by_object_id: Mapping[str, int]
    object_metrics: Mapping[str, ObjectRelationshipMetrics]
    pair_decisions: tuple[PairRelationshipDecision, ...]
```

Implement:

```python
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
) -> RelationshipPlan:
```

The implementation must:

1. Validate thresholds with the exact ranges in the design spec.
2. Reject duplicate `object_id` values with `ValueError`.
3. Validate every modal mask shape as `(image_height, image_width)`.
4. Precompute object metrics once.
5. Preserve input pair order from `itertools.combinations`.
6. Always create existing same-class merge edges for positive bbox overlap.
7. When disabled, classify every positive cross-class overlap as regular.
8. When enabled, evaluate containment, area dominance, veto, then dimension dominance in that order.
9. Use:

```python
large_width >= small_width * (1.0 - dimension_tolerance_ratio)
large_height >= small_height * (1.0 - dimension_tolerance_ratio)
```

10. Build components from all merge edges.
11. Suppress every regular pair whose endpoints have the same component ID.
12. Return tuples and immutable dataclasses at the public boundary.

Add the approved config block under `pipeline:` before `completion:` in `backend/config.yaml`.

- [ ] **Step 5: Run relationship and occlusion tests**

Run:

```powershell
pytest tests/test_relationships.py tests/test_occlusion.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the relationship-planning unit**

```powershell
git add backend/pipeline/relationships.py backend/pipeline/types.py backend/config.yaml tests/test_relationships.py
git commit -m "feat: plan cross-class containment relationships"
```

---

### Task 3: Explicit External-Pair Completion Linkage

**Files:**
- Modify: `backend/pipeline/completion/completion.py:26-58`
- Modify: `tests/test_pipeline_architecture.py:49-95`

**Interfaces:**
- Consumes: `Sequence[OverlapPair]` from `RelationshipPlan.external_overlap_pairs`
- Produces:
  - `link_overlap_partners(objects, pairs=None) -> list[OverlapPair]`
- Preserves:
  - Current no-argument behavior for existing callers and tests.
  - Current `overlap_partner_ids` semantics.
  - Current completion candidate ordering.

- [ ] **Step 1: Write a failing explicit-pair test**

Add to `tests/test_pipeline_architecture.py`:

```python
def test_completion_linking_uses_explicit_external_pairs_only():
    from backend.pipeline.completion import (
        get_completion_candidates,
        link_overlap_partners,
    )
    from backend.pipeline.types import DetectedObject

    mask = np.ones((4, 5), dtype=np.uint8)
    first = DetectedObject(
        "a", "class-a", "class-a", mask.copy(), (0, 0, 5, 4)
    )
    second = DetectedObject(
        "b", "class-b", "class-b", mask.copy(), (0, 0, 5, 4)
    )
    third = DetectedObject(
        "c", "class-c", "class-c", mask.copy(), (0, 0, 5, 4)
    )

    pairs = link_overlap_partners(
        [first, second, third],
        pairs=[("b", "c")],
    )

    assert pairs == [("b", "c")]
    assert first.overlap_partner_ids == set()
    assert second.overlap_partner_ids == {"c"}
    assert third.overlap_partner_ids == {"b"}
    assert [item.object_id for item in get_completion_candidates(
        [first, second, third]
    )] == ["b", "c"]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
pytest tests/test_pipeline_architecture.py::test_completion_linking_uses_explicit_external_pairs_only -v
```

Expected: FAIL because `link_overlap_partners()` does not accept `pairs`.

- [ ] **Step 3: Implement explicit-pair linkage**

Change the signature in `backend/pipeline/completion/completion.py`:

```python
def link_overlap_partners(
    objects: Sequence[DetectedObject],
    pairs: Sequence[OverlapPair] | None = None,
) -> list[OverlapPair]:
    """Link prefiltered external pairs, or discover legacy overlaps."""
    for detected in objects:
        detected.overlap_partner_ids.clear()

    linked_pairs = list(pairs) if pairs is not None else (
        find_cross_class_overlaps(
            [
                ObjectBounds(
                    object_id=detected.object_id,
                    semantic_class=detected.semantic_class,
                    bbox=detected.original_modal_bbox,
                )
                for detected in objects
            ]
        )
    )
    objects_by_id = {detected.object_id: detected for detected in objects}
    for first_id, second_id in linked_pairs:
        if first_id not in objects_by_id or second_id not in objects_by_id:
            raise ValueError(
                f"overlap pair references unknown objects: "
                f"{first_id!r}, {second_id!r}"
            )
        objects_by_id[first_id].overlap_partner_ids.add(second_id)
        objects_by_id[second_id].overlap_partner_ids.add(first_id)
        log_event(
            logger,
            "overlap_detection",
            "pair_decision",
            first_id=first_id,
            second_id=second_id,
            decision="completion_candidate",
            reason="external_cross_class_bbox_overlap",
        )
    return linked_pairs
```

Do not modify `get_completion_candidates()`, `complete_objects()`, or `filter_pairs_by_amodal_overlap()`.

- [ ] **Step 4: Run completion and architecture tests**

Run:

```powershell
pytest tests/test_pipeline_architecture.py tests/test_completion_architecture.py tests/test_completion_features.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit explicit-pair linkage**

```powershell
git add backend/pipeline/completion/completion.py tests/test_pipeline_architecture.py
git commit -m "refactor: link explicit external overlap pairs"
```

---

### Task 4: Final Group Materialization from Merge Edges

**Files:**
- Modify: `backend/pipeline/types.py:95-140`
- Modify: `backend/pipeline/grouping.py:33-190`
- Modify: `tests/test_grouping.py:34-138`

**Interfaces:**
- Consumes:
  - `Sequence[MergeEdge] | None`
  - Existing `Sequence[PairDecision]`
- Produces:
  - `group_reconstructed_objects(objects, pair_decisions=(), *, merge_edges=None)`
  - `GroupedObject.semantic_classes`
  - `GroupedObject.merge_edges`
- Preserves:
  - Existing positional `pair_decisions` argument.
  - Existing same-class fallback when `merge_edges is None`.
  - Existing depth ordering and composition behavior.

- [ ] **Step 1: Write failing explicit-merge tests**

Add to `tests/test_grouping.py`:

```python
def test_explicit_cross_class_merge_edge_materializes_one_group():
    from backend.pipeline.grouping import group_reconstructed_objects
    from backend.pipeline.types import MergeEdge

    container = _detected("table", "table", (0, 0, 8, 8))
    product = _detected("product", "product", (2, 2, 3, 3))
    edge = MergeEdge(
        first_id="table",
        second_id="product",
        reason="cross_class_bbox_containment",
        containment_ratio=1.0,
        bbox_size_ratio=9 / 64,
    )

    groups = group_reconstructed_objects(
        [container, product],
        merge_edges=[edge],
    )

    assert len(groups) == 1
    assert groups[0].member_ids == ("table", "product")
    assert groups[0].semantic_classes == ("table", "product")
    assert groups[0].merge_edges == (edge,)


def test_cross_class_group_primary_uses_largest_modal_area():
    from backend.pipeline.grouping import group_reconstructed_objects
    from backend.pipeline.types import MergeEdge

    small_first = _detected("small", "hat", (2, 2, 2, 2))
    small_first.segmentation_index = 0
    large_second = _detected("large", "person", (0, 0, 8, 8))
    large_second.segmentation_index = 1
    edge = MergeEdge("small", "large", "cross_class_bbox_containment")

    group = group_reconstructed_objects(
        [small_first, large_second],
        merge_edges=[edge],
    )[0]

    assert group.group_id == "group-large"
    assert group.semantic_class == "person"
    assert group.display_label == large_second.display_label
    assert group.segmentation_index == 1


def test_large_same_class_group_keeps_existing_primary_order():
    from backend.pipeline.grouping import group_reconstructed_objects

    small_first = _detected("first", "person", (1, 1, 3, 3))
    small_first.segmentation_index = 0
    large_second = _detected("second", "person", (0, 0, 8, 8))
    large_second.segmentation_index = 1

    group = group_reconstructed_objects([small_first, large_second])[0]

    assert group.group_id == "group-first"
    assert group.segmentation_index == 0
```

- [ ] **Step 2: Run focused grouping tests and verify RED**

Run:

```powershell
pytest tests/test_grouping.py -k "explicit_cross_class or cross_class_group_primary or large_same_class" -v
```

Expected: FAIL because the new keyword argument and provenance fields do not exist.

- [ ] **Step 3: Extend `GroupedObject` metadata**

Add defaults after `segmentation_index` in `backend/pipeline/types.py`:

```python
semantic_classes: tuple[str, ...] = ()
merge_edges: tuple[MergeEdge, ...] = ()
```

No existing constructor needs modification because both fields have defaults.

- [ ] **Step 4: Make grouping consume explicit edges**

Change the signature:

```python
def group_reconstructed_objects(
    objects: Sequence[DetectedObject],
    pair_decisions: Sequence[PairDecision] = (),
    *,
    merge_edges: Sequence[MergeEdge] | None = None,
) -> list[GroupedObject]:
```

Behavior:

```python
if merge_edges is None:
    effective_edges = _legacy_same_class_merge_edges(ordered)
else:
    effective_edges = tuple(merge_edges)
```

Implement `_legacy_same_class_merge_edges()` by moving the current same-class positive-bbox-overlap criterion into a helper that returns `MergeEdge` instances. Preserve existing pair-decision log reasons.

Validate explicit edges before union:

```python
index_by_id = {
    detected.object_id: index
    for index, detected in enumerate(ordered)
}
for edge in effective_edges:
    if edge.first_id not in index_by_id or edge.second_id not in index_by_id:
        raise ValueError(
            f"merge edge references unknown objects: "
            f"{edge.first_id!r}, {edge.second_id!r}"
        )
    union(index_by_id[edge.first_id], index_by_id[edge.second_id])
```

For each group:

```python
member_classes = tuple(dict.fromkeys(
    member.semantic_class for member in members
))
is_multiclass = len(member_classes) > 1
primary = (
    max(
        members,
        key=lambda member: (
            int(np.count_nonzero(member.modal_mask)),
            -member.segmentation_index,
        ),
    )
    if is_multiclass
    else members[0]
)
group_edges = tuple(
    edge
    for edge in effective_edges
    if edge.first_id in member_ids and edge.second_id in member_ids
)
```

Use `primary` for `group_id`, `semantic_class`, `display_label`, and `segmentation_index`. Preserve the current stable member tuple ordering.

- [ ] **Step 5: Run all grouping tests**

Run:

```powershell
pytest tests/test_grouping.py tests/test_group_layers.py tests/test_group_matting.py tests/test_group_background.py -v
```

Expected: all tests pass, including existing depth ordering and reconstruction conflict tests.

- [ ] **Step 6: Commit final-group support**

```powershell
git add backend/pipeline/types.py backend/pipeline/grouping.py tests/test_grouping.py
git commit -m "feat: materialize groups from merge intent"
```

---

### Task 5: Orchestrator Integration Without Heavy-Workflow Changes

**Files:**
- Modify: `backend/pipeline/orchestrator.py:80-238`
- Modify: `backend/pipeline/orchestrator.py:316-676`
- Modify: `tests/test_pipeline_architecture.py:157-230`
- Modify: `tests/test_grouping.py:299-449`

**Interfaces:**
- Consumes:
  - `plan_object_relationships(...)`
  - `RelationshipPlan.external_overlap_pairs`
  - `RelationshipPlan.merge_edges`
  - explicit-pair `link_overlap_partners(...)`
  - explicit-edge `group_reconstructed_objects(...)`
- Produces:
  - Relationship planning in both `process_masks()` and `process_image()`.
- Preserves:
  - Existing completion, depth ordering, reconstruction, support refinement, and downstream call signatures.

- [ ] **Step 1: Write a failing `process_masks()` containment test**

Add to `tests/test_pipeline_architecture.py`:

```python
def test_process_masks_skips_internal_containment_completion(
    monkeypatch,
):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    large_mask = np.zeros((20, 20), dtype=np.uint8)
    large_mask[0:12, 0:12] = 255
    small_mask = np.zeros((20, 20), dtype=np.uint8)
    small_mask[3:7, 3:7] = 255
    large = DetectedObject(
        "large", "table", "table", large_mask, (0, 0, 12, 12)
    )
    small = DetectedObject(
        "small", "product", "product", small_mask, (3, 3, 4, 4)
    )
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[large, small]),
    )
    complete = Mock()
    monkeypatch.setattr(orchestrator, "_complete_candidates", complete)

    masks = orchestrator.process_masks(
        Image.new("RGB", (20, 20)),
        ["table", "product"],
        manager,
    )

    assert len(masks) == 2
    complete.assert_not_called()
```

- [ ] **Step 2: Write a failing A–B merge intent plus B–C external test**

Add an orchestrator-level test to `tests/test_grouping.py` using the existing process-image monkeypatch fixture pattern:

```python
def test_merge_intent_does_not_expand_external_completion_target(
    monkeypatch,
):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    def detected(
        object_id: str,
        semantic_class: str,
        bbox: tuple[int, int, int, int],
    ) -> DetectedObject:
        mask = np.zeros((40, 40), dtype=np.uint8)
        x, y, width, height = bbox
        mask[y : y + height, x : x + width] = 255
        return DetectedObject(
            object_id,
            semantic_class,
            semantic_class,
            mask,
            bbox,
        )

    # A contains exactly 70% of B. C overlaps only B's exposed strip.
    a = detected("a", "table", (0, 0, 20, 20))
    b = detected("b", "product", (13, 5, 10, 10))
    c = detected("c", "hand", (21, 5, 6, 10))
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[a, b, c]),
    )
    captured_completion_ids: list[str] = []

    def capture_completion(
        _image,
        _objects,
        _manager,
        candidates,
    ):
        captured_completion_ids.extend(
            item.object_id for item in candidates
        )

    monkeypatch.setattr(
        orchestrator,
        "_complete_candidates",
        Mock(side_effect=capture_completion),
    )

    masks = orchestrator.process_masks(
        Image.new("RGB", (40, 40)),
        ["table", "product", "hand"],
        manager=manager,
    )

    assert len(masks) == 3
    assert captured_completion_ids == ["b", "c"]
    assert "a" not in captured_completion_ids
```

This geometry is intentional: B extends beyond A while still meeting the
`0.70` containment threshold, so C can overlap B without independently
overlapping A. The test therefore proves that merge intent itself does not
expand completion candidates.

- [ ] **Step 3: Run the two tests and verify RED**

Run:

```powershell
pytest tests/test_pipeline_architecture.py::test_process_masks_skips_internal_containment_completion tests/test_grouping.py::test_merge_intent_does_not_expand_external_completion_target -v
```

Expected: FAIL because neither workflow invokes relationship planning.

- [ ] **Step 4: Add backward-compatible configuration resolution**

In `backend/pipeline/orchestrator.py`:

```python
def _get_cross_class_grouping_config() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": True,
        "containment_threshold": 0.70,
        "max_bbox_size_ratio": 0.50,
        "large_mask_ratio": 0.45,
        "large_bbox_ratio": 0.75,
        "dimension_tolerance_ratio": 0.05,
    }
    try:
        defaults.update(
            config.get_pipeline_config("cross_class_grouping")
        )
    except (KeyError, ValueError):
        pass
    return defaults
```

Add:

```python
def _plan_relationships(
    objects: Sequence,
    image_size: tuple[int, int],
):
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
```

- [ ] **Step 5: Integrate `process_masks()`**

Immediately after the empty-object return:

```python
relationship_plan = _plan_relationships(objects, image.size)
with trace_stage(logger, "overlap_detection"):
    pairs = link_overlap_partners(
        objects,
        relationship_plan.external_overlap_pairs,
    )
```

Keep `get_completion_candidates()`, `_complete_candidates()`, model release, and returned mask list unchanged.

- [ ] **Step 6: Integrate `process_image()`**

After kernel/config preparation and before completion:

```python
relationship_plan = _plan_relationships(objects, image.size)
with trace_stage(logger, "overlap_detection"):
    potential_overlap_pairs = link_overlap_partners(
        objects,
        relationship_plan.external_overlap_pairs,
    )
```

Leave all code from `get_completion_candidates()` through reconstruction support refinement algorithmically unchanged.

Change only the final grouping call:

```python
final_groups = group_reconstructed_objects(
    objects,
    pair_decisions or [],
    merge_edges=relationship_plan.merge_edges,
)
```

Do not pass a materialized component into completion or reconstruction.

- [ ] **Step 7: Assert final grouping receives the prevalidated edge**

Update the existing
`test_orchestrator_groups_after_reconstruction_before_downstream` with:

```python
from backend.pipeline.types import MergeEdge

expected_edge = MergeEdge(
    "first",
    "second",
    "cross_class_bbox_containment",
    containment_ratio=0.8,
    bbox_size_ratio=0.4,
)
relationship_plan = Mock(
    external_overlap_pairs=(("first", "second"),),
    merge_edges=(expected_edge,),
)
monkeypatch.setattr(
    orchestrator,
    "_plan_relationships",
    Mock(return_value=relationship_plan),
)
```

Replace its overlap stub with:

```python
def link(supplied, pairs):
    assert supplied == objects
    assert tuple(pairs) == (("first", "second"),)
    first.overlap_partner_ids.add("second")
    second.overlap_partner_ids.add("first")
    return list(pairs)

monkeypatch.setattr(
    orchestrator,
    "link_overlap_partners",
    Mock(side_effect=link),
)
```

Change its group side effect to:

```python
def group(supplied, decisions, *, merge_edges):
    assert supplied == objects
    assert decisions == []
    assert tuple(merge_edges) == (expected_edge,)
    events.append("group")
    return final_groups
```

- [ ] **Step 8: Add member-specific reconstruction regression**

Extend the orchestrator A–B–C fixture with a deterministic pair-decision stub:

```python
decision = PairDecision(
    first_id="b",
    second_id="c",
    occluded_id="b",
    occluder_id="c",
    reconstruction_directions=(("b", "c"),),
)

def prepare(supplied, pairs, *_args, **_kwargs):
    assert supplied == [a, b, c]
    assert list(pairs) == [("b", "c")]
    b.reconstruction_mask = np.zeros_like(b.modal_mask, dtype=bool)
    b.reconstruction_mask[5, 21] = True
    b.occluder_ids.add("c")
    b.occluder_classes.add(c.semantic_class)
    return [decision]

monkeypatch.setattr(
    orchestrator,
    "prepare_raw_reconstruction_masks",
    Mock(side_effect=prepare),
)
reconstructed_ids: list[str] = []

def reconstruct(_image, supplied, *_args, **_kwargs):
    reconstructed_ids.extend(
        item.object_id
        for item in supplied
        if item.reconstruction_mask is not None
        and np.any(item.reconstruction_mask)
    )
    b.reconstruction_canvas = Image.new("RGB", (2, 2), "red")

monkeypatch.setattr(
    orchestrator,
    "reconstruct_objects",
    Mock(side_effect=reconstruct),
)

orchestrator.process_image(
    Image.new("RGB", (40, 40)),
    ["table", "product", "hand"],
    manager=manager,
)

objects_by_id = {item.object_id: item for item in (a, b, c)}
assert reconstructed_ids == ["b"]
assert objects_by_id["b"].reconstruction_mask is not None
assert objects_by_id["a"].reconstruction_mask is None
assert objects_by_id["a"].occluder_ids == set()
assert objects_by_id["b"].occluder_ids == {"c"}
```

- [ ] **Step 9: Run orchestrator and architecture regression tests**

Run:

```powershell
pytest tests/test_pipeline_architecture.py tests/test_grouping.py tests/test_completion_mask_inputs.py tests/test_reconstruction_support_refinement.py -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit orchestrator integration**

```powershell
git add backend/pipeline/orchestrator.py tests/test_pipeline_architecture.py tests/test_grouping.py
git commit -m "feat: route external pairs through existing pipeline"
```

---

### Task 6: Relationship Diagnostics

**Files:**
- Modify: `backend/pipeline/diagnostics.py:32-82`
- Modify: `backend/pipeline/diagnostics.py:143-243`
- Modify: `backend/pipeline/orchestrator.py:295-310`
- Modify: `backend/pipeline/orchestrator.py:744-755`
- Modify: `tests/test_pipeline_diagnostics.py:30-146`

**Interfaces:**
- Consumes: `RelationshipPlan | None`
- Produces backward-compatible additions:
  - `ObjectDiagnostics.mask_image_ratio`
  - `ObjectDiagnostics.bbox_image_ratio`
  - `ObjectDiagnostics.large_mask_veto`
  - `ObjectDiagnostics.large_bbox_veto`
  - `GroupDiagnostics.semantic_classes`
  - relationship counts and pair decisions in `PipelineDiagnostics`

- [ ] **Step 1: Write failing diagnostic tests**

Add to `tests/test_pipeline_diagnostics.py`:

```python
def test_diagnostics_record_relationship_plan_without_pixel_data():
    from backend.pipeline.diagnostics import build_pipeline_diagnostics
    from backend.pipeline.relationships import plan_object_relationships
    from backend.pipeline.types import DetectedObject

    large_mask = np.zeros((20, 20), dtype=np.uint8)
    large_mask[0:10, 0:10] = 255
    small_mask = np.zeros((20, 20), dtype=np.uint8)
    small_mask[2:5, 2:5] = 255
    large = DetectedObject(
        "large", "table", "table", large_mask, (0, 0, 10, 10)
    )
    small = DetectedObject(
        "small", "product", "product", small_mask, (2, 2, 3, 3)
    )
    plan = plan_object_relationships(
        [large, small],
        image_size=(20, 20),
        enabled=True,
        containment_threshold=0.70,
        max_bbox_size_ratio=0.50,
        large_mask_ratio=0.45,
        large_bbox_ratio=0.75,
        dimension_tolerance_ratio=0.05,
    )

    diagnostics = build_pipeline_diagnostics(
        raw_objects=[large, small],
        final_groups=[],
        potential_pairs=[],
        retained_pairs=[],
        completion_candidate_count=0,
        pair_decisions=[],
        stage_timings_ms={},
        peak_gpu_memory_bytes=None,
        relationship_plan=plan,
    )

    assert diagnostics.cross_class_containment_merge_count == 1
    assert diagnostics.external_overlap_count == 0
    assert diagnostics.relationship_decisions[0].containment_ratio == 1.0
    assert not hasattr(
        diagnostics.objects[0],
        "modal_mask",
    )
```

Add a separate group-provenance test:

```python
def test_group_diagnostics_record_multiclass_provenance():
    from backend.pipeline.diagnostics import build_pipeline_diagnostics
    from backend.pipeline.grouping import group_reconstructed_objects
    from backend.pipeline.types import MergeEdge

    table = _object("table", "table", 1)
    product = _object("product", "product", 2)
    edge = MergeEdge(
        "table",
        "product",
        "cross_class_bbox_containment",
        containment_ratio=1.0,
        bbox_size_ratio=0.25,
    )
    group = group_reconstructed_objects(
        [table, product],
        merge_edges=[edge],
    )[0]

    diagnostics = build_pipeline_diagnostics(
        raw_objects=[table, product],
        final_groups=[group],
        potential_pairs=[],
        retained_pairs=[],
        completion_candidate_count=0,
        pair_decisions=[],
        stage_timings_ms={},
        peak_gpu_memory_bytes=None,
    )

    assert diagnostics.groups[0].semantic_classes == (
        "table",
        "product",
    )
```

- [ ] **Step 2: Run diagnostic tests and verify RED**

Run:

```powershell
pytest tests/test_pipeline_diagnostics.py -v
```

Expected: FAIL because diagnostics do not accept or serialize a relationship plan.

- [ ] **Step 3: Add backward-compatible diagnostic dataclasses**

Add:

```python
@dataclass(frozen=True)
class RelationshipDecisionDiagnostics:
    first_id: str
    second_id: str
    relation: str
    reason: str
    containment_ratio: float | None
    bbox_size_ratio: float | None
```

Append defaulted fields to `PipelineDiagnostics`:

```python
same_class_merge_count: int = 0
cross_class_containment_merge_count: int = 0
regular_overlap_count: int = 0
external_overlap_count: int = 0
large_mask_veto_count: int = 0
large_bbox_veto_count: int = 0
relationship_decisions: tuple[
    RelationshipDecisionDiagnostics, ...
] = ()
```

Append optional/defaulted coverage fields to `ObjectDiagnostics` and `semantic_classes` to `GroupDiagnostics`. Keep every existing field name intact.

Change the builder signature:

```python
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
```

When `relationship_plan is None`, emit existing values and zero/empty defaults.

- [ ] **Step 4: Pass relationship plans from the orchestrator**

For the no-object early return, pass `relationship_plan=None`.

For the normal return, pass:

```python
relationship_plan=relationship_plan
```

Do not change timing, GPU memory, fallback, or pair-decision diagnostics.

- [ ] **Step 5: Run diagnostic and trace tests**

Run:

```powershell
pytest tests/test_pipeline_diagnostics.py tests/test_pipeline_trace_logging.py tests/test_pipeline_decision_logging.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit diagnostics**

```powershell
git add backend/pipeline/diagnostics.py backend/pipeline/orchestrator.py tests/test_pipeline_diagnostics.py
git commit -m "feat: report containment grouping diagnostics"
```

---

### Task 7: Documentation and Full Regression Verification

**Files:**
- Modify: `docs/image_to_components_pipeline.md`
- Verify: all files changed in Tasks 1-6

**Interfaces:**
- Consumes: final implemented pipeline and configuration.
- Produces: user-facing architecture documentation matching runtime behavior.

- [ ] **Step 1: Update the pipeline document**

Insert a new stage between segmentation and amodal completion:

```markdown
## Relationship Planning — Cross-Class Containment

Each raw object keeps its own mask and reconstruction state. The planner emits:

- same-class merge edges using the existing overlap rule;
- cross-class containment merge edges using containment, bbox-size dominance,
  per-axis dominance, and large-object veto;
- external regular-overlap pairs for the existing heavy workflow.

Large-object veto is:

`mask_image_ratio >= 0.45 OR bbox_image_ratio >= 0.75`

and applies only to cross-class containment merging.

Merge intent is materialized only after member-level reconstruction. If A and B
will be grouped but C occludes only B, completion uses external pair B–C and
only B becomes the reconstruction target.
```

Update stage numbering and the final grouping section to state that explicit merge edges are used. Do not rewrite unrelated model or background-inpainting documentation.

- [ ] **Step 2: Run formatting and focused static checks**

Run:

```powershell
python -m compileall backend/core/occlusion.py backend/pipeline
git diff --check
```

Expected: compile succeeds and `git diff --check` prints no errors.

- [ ] **Step 3: Run the complete focused feature suite**

Run:

```powershell
pytest tests/test_occlusion.py tests/test_relationships.py tests/test_pipeline_architecture.py tests/test_grouping.py tests/test_pipeline_diagnostics.py tests/test_completion_mask_inputs.py tests/test_reconstruction_validation.py tests/test_reconstruction_support_refinement.py tests/test_group_layers.py tests/test_group_matting.py tests/test_group_background.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run the full test suite**

Run:

```powershell
pytest -q
```

Expected: zero failures. Record the exact passed/skipped counts in the implementation handoff.

- [ ] **Step 5: Inspect the final diff for scope**

Run:

```powershell
git status --short
git diff --stat
git diff -- backend/core/occlusion.py backend/pipeline/relationships.py backend/pipeline/types.py backend/pipeline/completion/completion.py backend/pipeline/grouping.py backend/pipeline/orchestrator.py backend/pipeline/diagnostics.py backend/config.yaml docs/image_to_components_pipeline.md
```

Verify:

- No model adapter or inference implementation changed.
- No completion/reconstruction algorithm changed.
- Same-class grouping tests remain intact.
- No unrelated user files are staged.
- `.superpowers/` preview files are not committed.

- [ ] **Step 6: Commit documentation and verification state**

```powershell
git add docs/image_to_components_pipeline.md
git commit -m "docs: explain containment grouping pipeline"
```

- [ ] **Step 7: Request final code review**

Use `superpowers:requesting-code-review` and provide:

- The approved design spec.
- This implementation plan.
- The full commit range from Task 1 through Task 7.
- The focused and full pytest results.
- An explicit review request for same-class non-regression and A–B–C member-specific reconstruction.
