# SAM3 Duplicate Mask Suppression Design

## Goal

Prevent one physical object from becoming multiple raw pipeline objects when
two SAM3 keywords return effectively the same segmentation mask.

The change runs immediately after each SAM3 mask is normalized and before
completion, occlusion analysis, reconstruction, grouping, diagnostics export,
and downstream layer creation.

## Duplicate Metric

For two non-empty binary masks `A` and `B`, calculate:

```text
intersection = area(A AND B)
duplicate_overlap = intersection / max(area(A), area(B))
```

The masks are duplicates when:

```text
duplicate_overlap >= duplicate_mask_overlap_threshold
```

The default threshold is `0.90`. This metric is equivalent to requiring both:

```text
intersection / area(A) >= 0.90
intersection / area(B) >= 0.90
```

It therefore requires near-equality in both directions. A small mask that is
contained by a much larger mask is not a duplicate even when
`intersection / min(area(A), area(B))` is high.

## Stable Target Priority

SAM3 processes keywords in their supplied order. The resolved keyword list
places user targets before detected occluders.

For every new normalized non-empty mask:

1. Compare it with retained masks in their stable creation order.
2. If it duplicates a retained mask, keep the retained object and discard the
   new candidate.
3. Otherwise retain the new object.

This preserves the earlier target label. For example, if `man` appears before
`person` and both return the same mask, the `man` object remains.

The retained mask and label are not unioned, enlarged, renamed, or otherwise
modified.

## Non-Transitive Suppression

Suppression is greedy and compares a candidate only with masks that remain
retained.

If `A` duplicates `B`, `B` is discarded. A later `C` is compared with retained
`A`, not with discarded `B`. Duplicate relationships are not transitively
closed because a chain of near-overlaps can bridge two real instances.

## Pipeline Integration

Add the threshold to segmentation configuration:

```yaml
pipeline:
  segmentation:
    duplicate_mask_overlap_threshold: 0.90
```

`_get_segmentation_config()` provides the same default for older
configurations. `_segment()` passes the value to `extract_raw_objects()`.

`extract_raw_objects()` performs suppression after mask normalization and
empty-mask rejection, but before creating and appending a `DetectedObject`.
Because duplicate candidates are skipped before object creation:

- `object_id` remains contiguous;
- `segmentation_index` remains contiguous;
- diagnostics save only retained objects;
- all existing downstream behavior receives a deduplicated raw-object list.

The implementation checks bounding-box intersection before doing full mask
intersection, avoiding unnecessary full-image operations for disjoint masks.

## Validation

`duplicate_mask_overlap_threshold` must be finite and within `[0, 1]`.
Invalid configuration raises `ValueError` at segmentation time instead of
silently changing deduplication behavior.

Empty masks continue to use the existing rejection path and never enter
duplicate comparison.

## Diagnostics

When a duplicate is suppressed, emit a segmentation decision containing:

- kept object ID and keyword;
- discarded keyword and its SAM3 mask index;
- measured duplicate overlap;
- configured threshold;
- reason `duplicate_mask_overlap`.

Normal retained objects continue using the existing `object_detected` event.

## Tests

Focused segmentation tests cover:

1. Two different keywords returning the same mask retain the earlier target.
2. Masks with duplicate overlap exactly `0.90` suppress the later candidate.
3. A small mask contained within a much larger mask remains separate.
4. Overlapping instances below `0.90` remain separate.
5. Greedy suppression does not apply transitive closure through a discarded
   bridge mask.
6. Invalid threshold values are rejected.

No completion, reconstruction, grouping, matting, or inpainting behavior is
changed.
