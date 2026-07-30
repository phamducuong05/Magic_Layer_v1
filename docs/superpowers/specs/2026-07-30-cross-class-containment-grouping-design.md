# Cross-Class Containment Grouping Design

**Date:** 2026-07-30
**Status:** Approved design
**Scope:** Add cross-class containment grouping without changing the behavior of the existing mask, amodal completion, depth-ordering, reconstruction, same-class grouping, matting, layer extraction, or background workflows.

## 1. Objective

The pipeline currently groups overlapping objects of the same semantic class. It also sends overlapping objects of different classes through amodal completion, occlusion analysis, and optional RGB reconstruction.

This feature adds a second grouping rule for objects of different classes:

- A smaller bounding box is substantially contained by a larger bounding box.
- The two bounding boxes differ substantially in area.
- Neither object is vetoed as a large cross-class container candidate.

Such objects must belong to one final draggable component. Their internal relationship must not be treated as occlusion. They must nevertheless remain independent raw objects during amodal completion and reconstruction so that an external object can occlude and reconstruct only the affected member.

## 2. Design Principles

1. **Record grouping intent early; materialize groups late.**
   Cross-class containment is classified before completion, but raw objects are not physically merged until final grouping.

2. **Suppress relationships, not objects.**
   An internal merge relationship is excluded from occlusion processing. Its members remain eligible for completion and reconstruction because of external relationships.

3. **Preserve existing heavy workflows.**
   Existing mask creation, completion, completion validation, amodal-overlap filtering, depth ordering, reconstruction-mask preparation, reconstruction inference, support refinement, composition, matting, layer extraction, and background generation retain their current algorithms and contracts.

4. **Preserve same-class grouping.**
   Large-object veto applies only to cross-class containment merges. Existing same-class overlap grouping remains unchanged.

5. **Use original raw-object geometry.**
   Every merge edge is validated using the original raw bounding boxes. A union bounding box must never generate a new merge edge.

6. **Keep member-level external occlusion.**
   If objects A and B have merge intent and external object C occludes only B, the heavy workflow processes B and C. A is not included merely because A and B will later form one group.

## 3. Terminology

- **Raw object:** One `DetectedObject` produced by segmentation.
- **Merge edge:** A validated relationship requiring two raw objects to belong to the same final component.
- **Merge-intent component:** A connected component in the merge-edge graph. It predicts final membership without merging raw processing state.
- **Internal pair:** Two raw objects in the same merge-intent component.
- **External pair:** Two raw objects in different merge-intent components.
- **Materialized group:** The final `GroupedObject` created after completion, reconstruction, and member-level support refinement.
- **Large-object veto:** A rule that prevents an object from participating in a cross-class containment merge. It does not prevent same-class grouping or ordinary occlusion processing.

## 4. Geometry Metrics

Bounding boxes use the existing `(x, y, width, height)` convention.

For two valid boxes A and B:

```text
intersection_area = area(A ∩ B)
smaller_bbox_area = min(area(A), area(B))
larger_bbox_area  = max(area(A), area(B))

containment_ratio = intersection_area / smaller_bbox_area
bbox_size_ratio   = smaller_bbox_area / larger_bbox_area
```

The containment ratio deliberately differs from IoU. A small box fully inside a large box has a containment ratio of `1.0`, even when its IoU is low.

The geometry utility must:

- Reject or safely return a non-match for boxes with non-positive width or height.
- Treat edge-touching boxes as zero-area intersections.
- Avoid division by zero.
- Identify the smaller and larger boxes explicitly.
- Use the original modal bounding boxes captured during segmentation.
- Handle threshold equality inclusively.

## 5. Object-to-Image Coverage

For each raw object:

```text
image_area       = image_width × image_height
mask_area        = count_nonzero(modal_mask)
bbox_area        = width × height

mask_image_ratio = mask_area / image_area
bbox_image_ratio = bbox_area / image_area
```

The modal mask is used because relationship classification occurs before amodal completion.

An object is vetoed from cross-class containment grouping when either coverage threshold is reached:

```text
large_object_veto =
    mask_image_ratio >= large_mask_ratio
    OR
    bbox_image_ratio >= large_bbox_ratio
```

The logical `OR` is intentional:

- `mask_image_ratio` detects an object that truly occupies much of the image.
- `bbox_image_ratio` detects an irregular or sparse object whose bounding box spans enough of the image to contain many unrelated detections.

The veto has no effect on:

- Existing same-class grouping.
- Regular cross-class overlap detection.
- Amodal completion caused by an external overlap.
- Depth ordering or reconstruction.

## 6. Configuration

Initial configurable defaults:

```yaml
pipeline:
  cross_class_grouping:
    enabled: true
    containment_threshold: 0.70
    max_bbox_size_ratio: 0.50
    large_mask_ratio: 0.45
    large_bbox_ratio: 0.75
    dimension_tolerance_ratio: 0.05
```

Meanings:

- `containment_threshold`: Minimum intersection coverage of the smaller bounding box.
- `max_bbox_size_ratio`: Maximum allowed smaller-to-larger bounding-box area ratio. `0.50` requires the larger box to have at least twice the area.
- `large_mask_ratio`: Modal-mask image coverage that vetoes cross-class containment merging.
- `large_bbox_ratio`: Bounding-box image coverage that vetoes cross-class containment merging.
- `dimension_tolerance_ratio`: Small tolerance for segmentation jitter when checking that the larger-area box is not materially smaller along either axis.

All thresholds must be validated at configuration boundaries:

```text
0.0 <= containment_threshold <= 1.0
0.0 < max_bbox_size_ratio <= 1.0
0.0 <= large_mask_ratio <= 1.0
0.0 <= large_bbox_ratio <= 1.0
0.0 <= dimension_tolerance_ratio < 1.0
```

The defaults are starting values and must remain configurable for tuning against representative images.

## 7. Pair Classification

Each raw-object pair receives one relationship classification:

```python
class PairRelation(Enum):
    SAME_CLASS_MERGE = auto()
    CROSS_CLASS_CONTAINMENT_MERGE = auto()
    REGULAR_CROSS_CLASS_OVERLAP = auto()
    DISJOINT = auto()
```

### 7.1 Same-class pair

The existing behavior remains authoritative:

```text
same semantic class AND positive original-bbox overlap
    → SAME_CLASS_MERGE
```

Large-object veto is not evaluated for this relationship.

### 7.2 Cross-class containment pair

A cross-class pair becomes a merge edge only when all conditions hold:

```text
positive-area intersection
AND containment_ratio >= containment_threshold
AND bbox_size_ratio <= max_bbox_size_ratio
AND neither endpoint has large_object_veto
AND larger_width  >= smaller_width  × (1 - dimension_tolerance_ratio)
AND larger_height >= smaller_height × (1 - dimension_tolerance_ratio)
```

The dimension check prevents boxes with incompatible horizontal and vertical shapes from being treated as containment merely because their area metrics pass.

### 7.3 Regular cross-class overlap

If a cross-class pair has positive-area overlap but fails any containment-merge condition, it remains a regular overlap:

```text
cross-class positive overlap
AND not CROSS_CLASS_CONTAINMENT_MERGE
    → REGULAR_CROSS_CLASS_OVERLAP
```

This includes pairs rejected by large-object veto. They continue through the existing completion and occlusion workflow.

### 7.4 Disjoint pair

Pairs without positive-area overlap are unrelated.

## 8. Merge Edges and Transitivity

Relationship planning produces explicit merge edges with diagnostics:

```python
@dataclass(frozen=True)
class MergeEdge:
    first_id: str
    second_id: str
    reason: str
    containment_ratio: float | None = None
    bbox_size_ratio: float | None = None
```

Reasons include:

- `same_class_bbox_overlap`
- `cross_class_bbox_containment`

Union-find runs over all valid merge edges. Transitivity is intentional:

```text
A contains B
B contains C
→ final component {A, B, C}
```

The B–C edge is validated using B and C, not the union bounding box of A and B.

If A–C is a regular occlusion edge but A, B, and C are connected through other valid merge edges, the final component is still `{A, B, C}`. The A–C relationship becomes internal and is suppressed. Regular internal edges do not block transitive grouping.

## 9. External-Pair Projection

After union-find assigns a merge-intent component ID to every raw object, regular cross-class overlap pairs are filtered:

```python
external_overlap_pairs = [
    pair
    for pair in regular_overlap_pairs
    if component_id(pair.first_id) != component_id(pair.second_id)
]
```

Only `external_overlap_pairs` populate `overlap_partner_ids` and enter the existing heavy pipeline.

This filtering suppresses:

- Direct containment pairs.
- Regular pairs that become internal through transitive merge edges.

It preserves the original raw member IDs for every external relationship. Component union bounding boxes are not used to invent external overlaps.

## 10. Revised `process_image` Data Flow

```text
1. Segmentation
   → raw DetectedObject instances

2. Relationship planning
   → object coverage metrics
   → same-class merge edges
   → cross-class containment merge edges
   → regular cross-class overlap pairs

3. Build merge-intent components
   → union-find over merge edges

4. Filter external overlap pairs
   → remove every pair whose endpoints share a component

5. Existing amodal completion workflow
   → candidates are endpoints of external overlap pairs
   → algorithms and validation remain unchanged

6. Existing amodal-overlap validation
   → external pairs only

7. Existing depth ordering and reconstruction-mask preparation
   → raw objects and raw external pair IDs

8. Existing object reconstruction
   → only raw objects with non-empty reconstruction masks

9. Existing reconstruction-support refinement
   → each reconstructed raw object independently

10. Final materialized grouping
    → union raw members using the prevalidated merge edges
    → compose member-level visible and reconstructed RGB

11. Existing group matting, layer extraction, and background generation
```

## 11. A–B–C Behavior

Given:

```text
A contains B
B regularly overlaps C
C is the occluder of B
```

The expected behavior is:

```text
merge_edges             = [(A, B)]
external_overlap_pairs  = [(B, C)]
completion_candidates   = [B, C]
reconstruction_target   = B
reconstruction_occluder = C
final_groups            = [{A, B}, {C}]
```

Important invariants:

- A–B never enters occlusion analysis.
- A is not completed or reconstructed merely because A will be grouped with B.
- The existing completion workflow may process both B and C because depth direction is determined after amodal evidence is available.
- Only B receives reconstruction state when C is determined to occlude B.
- B's accepted reconstructed RGB support is retained when A and B are materialized as one final group.

The same rule applies symmetrically when C occludes A instead of B.

## 12. Final Group Materialization

Final grouping receives raw objects, explicit merge edges, and existing pair decisions.

For each connected component:

```text
group.modal_mask =
    union(member.modal_mask)

group.amodal_mask =
    union(member.amodal_mask if present else member.modal_mask)

group.effective_support_mask =
    union(member reconstruction support when accepted,
          otherwise member modal support)
```

`compose_group_sources()` continues to:

- Initialize from the original image.
- Protect visible modal pixels.
- Apply accepted reconstruction writes per raw member.
- Resolve reconstruction-write conflicts deterministically.
- Record conflicts for diagnostics.

A member without reconstruction data requires no fabricated reconstruction fields. Existing modal fallback behavior remains valid.

## 13. Multi-Class Group Identity

A cross-class group must retain member provenance:

```text
member_ids
members
semantic_classes
merge_edges
```

The primary member is selected by largest modal-mask area, with segmentation order as the deterministic tie-breaker. Its class and display label remain the backward-compatible primary `semantic_class` and `display_label` exposed downstream.

Same-class groups retain their existing effective label because all members share the same class.

## 14. `process_masks` Contract

`process_masks()` keeps its current raw-object output contract:

- One returned mask per segmented raw object.
- Original object ordering remains unchanged.
- Internal containment relationships do not create completion candidates.
- External relationships continue to use the existing completion behavior.
- Masks are not collapsed into one mask per final draggable group.

This avoids a breaking API change while keeping relationship semantics consistent with `process_image()`.

## 15. Diagnostics

Existing diagnostic fields remain backward compatible. New diagnostics should record:

- Same-class merge-edge count.
- Cross-class containment merge-edge count.
- Large-mask veto count.
- Large-bbox veto count.
- Regular cross-class overlap count before component filtering.
- External overlap count after component filtering.
- Per-pair containment ratio and bounding-box size ratio.
- Per-object mask-image and bbox-image ratios.
- Final group member IDs and semantic classes.

Decision reasons must distinguish:

- `same_class_bbox_overlap`
- `cross_class_bbox_containment`
- `large_mask_ratio_veto`
- `large_bbox_ratio_veto`
- `bbox_size_ratio_not_dominant`
- `bbox_dimensions_not_dominant`
- `regular_cross_class_overlap`
- `internal_pair_suppressed`

## 16. Compatibility Boundaries

The implementation must not change the internal algorithms or public behavior of:

- Segmentation and modal-mask creation.
- Completion model invocation and output validation.
- Amodal-mask overlap validation.
- Depth-order decision logic.
- Reconstruction-mask generation.
- Object reconstruction model invocation and validation.
- Reconstruction support refinement.
- Existing same-class grouping criteria.
- Source-composition conflict priority.
- Final matting.
- Layer extraction.
- Background inpainting.
- `ProcessResult` fields.

Permitted integration changes are limited to:

- New geometry and coverage utilities.
- New relationship-planning data structures.
- Passing explicit external overlap pairs into the existing workflow.
- Passing explicit merge edges into final grouping.
- Adding backward-compatible diagnostics and configuration.

## 17. Error Handling

- Invalid boxes cannot produce containment merge edges.
- Invalid or mismatched mask shapes fail at the existing object-validation boundary.
- Missing reconstruction data uses the existing modal fallback.
- Unknown IDs in relationship plans raise a clear validation error before heavy inference.
- Duplicate edges are canonicalized and deduplicated.
- Empty input preserves current empty-result behavior.
- Disabling `cross_class_grouping.enabled` restores current cross-class behavior exactly.

## 18. Performance

Pair classification remains `O(n²)`, matching the current pairwise overlap search. This is acceptable for the expected number of detections per image.

The implementation should calculate per-object areas and coverage ratios once, then reuse them for all pairs. Spatial indexing is out of scope unless profiling shows that pair classification is a bottleneck.

## 19. Test Strategy

### Geometry unit tests

- Fully contained boxes return containment ratio `1.0`.
- Partial containment returns the expected ratio.
- Edge-touching boxes return `0.0`.
- Invalid and zero-area boxes cannot merge.
- Equal or near-equal boxes fail the size-dominance rule.
- Threshold equality passes.
- Axis-incompatible boxes fail dimension dominance.

### Large-object veto tests

- Mask ratio alone triggers veto.
- Bbox ratio alone triggers veto.
- A vetoed cross-class pair cannot create a containment merge edge.
- A vetoed cross-class overlap remains a regular overlap.
- A large same-class pair still groups under existing rules.

### Relationship-planning tests

- Cross-class containment produces one merge edge and no overlap pair.
- Non-containment cross-class overlap produces one regular overlap pair.
- A–B and B–C valid merge edges create `{A, B, C}`.
- No merge edge is created from a component union bounding box.
- A regular A–C edge becomes suppressed when A and C are transitively connected.
- Edge order and object input order do not change component membership.

### Orchestrator tests

- A–B containment alone invokes neither completion nor reconstruction.
- A–B merge intent plus B–C external overlap gives B and C as completion candidates, not A.
- When C occludes B, only B receives reconstruction state.
- When C occludes A, only A receives reconstruction state.
- Existing completion and reconstruction call contracts remain unchanged.
- Disabling the feature reproduces current cross-class behavior.

### Final grouping and downstream tests

- A and B become one final draggable component.
- B's accepted reconstruction support survives grouping with A.
- Missing reconstruction fields use modal fallback without crashes.
- Multi-class provenance and primary-label selection are deterministic.
- Existing same-class grouping tests remain unchanged and pass.
- `process_masks()` retains raw-object count and ordering.
- Existing pipeline architecture and diagnostics tests remain green.

## 20. Acceptance Criteria

The design is complete when all of the following hold:

1. Cross-class pairs merge only when containment, size dominance, dimension dominance, and veto rules pass.
2. Large-object veto uses both modal-mask and bbox image coverage with `OR`.
3. Large-object veto affects only cross-class containment merging.
4. Same-class grouping behavior is unchanged.
5. Internal merge-intent pairs never enter occlusion or reconstruction.
6. External occlusion remains member-specific.
7. C occluding A or B processes only the affected raw member for reconstruction.
8. Transitive grouping uses only validated raw-object merge edges.
9. Regular internal edges do not block an otherwise valid transitive group.
10. Final groups retain member-level accepted reconstruction support.
11. Existing heavy-processing algorithms and downstream contracts remain unchanged.
12. Feature disablement restores existing cross-class behavior.
