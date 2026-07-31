# SAM3 Duplicate Mask Suppression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suppress duplicate SAM3 masks returned by different keywords while preserving the first target-priority object.

**Architecture:** Add a validated overlap helper in `segmentation.py`, apply stable greedy suppression immediately after mask normalization and empty-mask rejection, and pass a configurable `0.90` threshold from `backend/config.yaml` through the orchestrator. Suppressed candidates never become `DetectedObject` instances, so downstream stages remain unchanged.

**Tech Stack:** Python, NumPy, PyTorch, Pillow, pytest.

## Global Constraints

- Duplicate metric is `intersection / max(area(A), area(B))`.
- Duplicate threshold defaults to `0.90` and is inclusive.
- Keep the earlier keyword/object; target order already precedes occluder order.
- Suppression is greedy and non-transitive.
- Do not union masks, rename labels, or alter downstream completion/reconstruction/grouping.
- Use the Conda `layer` Python interpreter for focused tests.

---

### Task 1: Add stable duplicate suppression to raw SAM3 extraction

**Files:**
- Modify: `backend/pipeline/segmentation.py`
- Modify: `backend/pipeline/orchestrator.py`
- Modify: `backend/config.yaml`
- Test: `tests/test_image_processor.py`
- Test: `tests/test_pipeline_architecture.py`

**Interfaces:**
- Add `extract_raw_objects(..., duplicate_mask_overlap_threshold: float = 0.90) -> list[DetectedObject]`.
- Add `_duplicate_mask_overlap(first: np.ndarray, second: np.ndarray) -> float`.
- `_segment()` consumes the segmentation config and passes the threshold to `extract_raw_objects()`.

- [ ] **Step 1: Write failing segmentation tests**

Add tests for:

```python
# identical masks from later keyword: first target wins
objects = extract_raw_objects(image, ["man", "person"], processor)
assert len(objects) == 1
assert objects[0].semantic_class == "man"

# exactly 90% overlap suppresses the later mask
assert len(extract_raw_objects(image, ["first", "second"], processor)) == 1

# nested but much smaller object is not a duplicate
assert len(extract_raw_objects(image, ["large", "small"], processor)) == 2

# overlap below threshold stays separate
assert len(extract_raw_objects(image, ["a", "b"], processor)) == 2

# A-B duplicate does not cause C to be compared transitively through B
assert retained_labels == ["a", "c"]
```

Add a threshold validation test that passes `-0.1` and `1.1` and expects
`ValueError`. Extend the architecture assertion to require duplicate
suppression source in `extract_raw_objects()` and no use of the old
`_merge_overlapping_masks` helper.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest -q tests/test_image_processor.py -k extract_raw_objects tests/test_pipeline_architecture.py -k segmentation
```

Expected: duplicate cases currently return two objects and invalid thresholds
are not accepted as a new argument.

- [ ] **Step 3: Implement the overlap helper**

In `segmentation.py`, normalize both masks to boolean and return `0.0` for an
empty mask. Otherwise implement:

```python
intersection = np.count_nonzero(first_bool & second_bool)
denominator = max(
    np.count_nonzero(first_bool),
    np.count_nonzero(second_bool),
)
return intersection / denominator
```

Validate the threshold with `math.isfinite()` and `[0.0, 1.0]` bounds.

- [ ] **Step 4: Implement greedy suppression before object creation**

After `_normalise_mask()` and `_bbox_from_mask()` reject empty masks, compare
the candidate against retained `(mask, keyword, object_id)` records. Use a
bounding-box overlap precheck, then `_duplicate_mask_overlap()`.

```python
if duplicate_overlap >= duplicate_mask_overlap_threshold:
    log_event(
        logger,
        "segmentation",
        "mask_decision",
        keyword=keyword,
        mask_index=index,
        decision="reject",
        reason="duplicate_mask_overlap",
        kept_object_id=kept.object_id,
        overlap_ratio=duplicate_overlap,
        threshold=duplicate_mask_overlap_threshold,
    )
    continue
```

Only retained candidates receive `DetectedObject`, `object_id`, and
`segmentation_index`. Keep the current display-label behavior for retained
SAM3 instances.

- [ ] **Step 5: Wire configuration through orchestration**

Add:

```yaml
pipeline:
  segmentation:
    duplicate_mask_overlap_threshold: 0.90
```

Add the same default in `_get_segmentation_config()`, extend `_segment()` with
the threshold argument, and pass it from both segmentation call sites in
`process_masks()` and `process_image()`.

- [ ] **Step 6: Run focused tests and compile check**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest -q tests/test_image_processor.py -k extract_raw_objects tests/test_pipeline_architecture.py -k segmentation
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m compileall -q backend tests
git diff --check
```

Expected: all selected tests pass, compilation succeeds, and no diff errors
are reported.

- [ ] **Step 7: Commit the scoped implementation**

```powershell
git add -- backend/pipeline/segmentation.py backend/pipeline/orchestrator.py backend/config.yaml tests/test_image_processor.py tests/test_pipeline_architecture.py docs/superpowers/plans/2026-07-31-sam3-duplicate-mask-suppression.md
git commit -m "feat: suppress duplicate SAM3 masks"
```
