# BiRefNet Alpha Coverage Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every SAM3-confirmed group member when BiRefNet or final alpha refinement omits its pixels.

**Architecture:** Add one focused alpha-recovery helper to `matting.py`. It builds a safe per-member fallback domain from the visible modal mask and accepted reconstruction replacement domain, measures coverage, and unions hard alpha only for failed members. Call it once after BiRefNet and once after final color refinement.

**Tech Stack:** Python, NumPy, OpenCV, PyTorch, Pillow, pytest.

## Global Constraints

- Preserve existing SAM3, grouping, completion, reconstruction, and background-inpainting behavior.
- A non-reconstructed member falls back to `modal_mask`.
- A reconstructed member falls back to `modal_mask OR (reconstruction_replacement_domain_mask AND reconstruction_accepted_rgb_mask)`.
- Validate and recover each group member independently.
- Use `alpha_presence_threshold: 0.05` and `min_member_alpha_coverage_ratio: 0.95` as defaults.
- Run only focused matting and layer tests with the Conda `layer` interpreter.

---

### Task 1: Per-member alpha coverage recovery

**Files:**
- Modify: `backend/pipeline/matting.py`
- Test: `tests/test_group_matting.py`

**Interfaces:**
- Consumes: `GroupedObject`, `SquareROI`, member modal and reconstruction masks.
- Produces: `recover_missing_member_alpha(alpha, group, roi, presence_threshold, min_coverage_ratio, stage) -> np.ndarray`.

- [ ] **Step 1: Write failing tests for empty alpha and safe replacement-domain fallback**

Add tests that construct:

```python
member.reconstruction_canvas = Image.new("RGB", (4, 4), "red")
member.reconstruction_replacement_domain_mask = replacement_domain
member.reconstruction_accepted_rgb_mask = accepted_rgb
```

Return an all-zero matte and assert:

```python
assert group.soft_alpha[modal_mask].min() == 1.0
assert group.soft_alpha[safe_hidden_pixel] == 1.0
assert group.soft_alpha[protected_foreign_pixel] == 0.0
```

Add a two-member group where BiRefNet covers the first member but omits the
second. Assert the first member retains its original soft alpha and only the
second member receives hard fallback.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest -q tests/test_group_matting.py
```

Expected: new assertions fail because current `refine_objects()` accepts the
empty/incomplete BiRefNet alpha.

- [ ] **Step 3: Implement the fallback-domain and recovery helper**

Add private domain construction:

```python
def _member_alpha_fallback_domain(member: DetectedObject) -> np.ndarray:
    modal = member.modal_mask > 0
    replacement = member.reconstruction_replacement_domain_mask
    accepted = member.reconstruction_accepted_rgb_mask
    if (
        member.reconstruction_canvas is None
        or replacement is None
        or accepted is None
        or replacement.shape != modal.shape
        or accepted.shape != modal.shape
    ):
        return modal
    return modal | (replacement.astype(bool) & accepted.astype(bool))
```

Add a recovery function that:

```python
validated = np.nan_to_num(
    np.clip(alpha.astype(np.float64), 0.0, 1.0),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
for member in group.members:
    domain = crop_array(_member_alpha_fallback_domain(member), roi)
    domain_area = np.count_nonzero(domain)
    coverage = (
        np.count_nonzero(domain & (validated > presence_threshold))
        / domain_area
        if domain_area
        else 1.0
    )
    if coverage < min_coverage_ratio:
        validated[domain] = np.maximum(validated[domain], 1.0)
```

Validate both configuration values are finite and in `[0, 1]`, validate alpha
shape against `roi.size`, and log each member fallback with group ID, member
ID, stage, coverage ratio, domain pixels, and reason.

- [ ] **Step 4: Apply recovery after BiRefNet**

Extend `refine_objects()` with:

```python
alpha_presence_threshold: float = 0.05
min_member_alpha_coverage_ratio: float = 0.95
```

After support clipping, call the helper with `stage="birefnet_output"` before
restoring the crop into `group.soft_alpha`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest -q tests/test_group_matting.py
```

Expected: all tests pass.

---

### Task 2: Final layer alpha invariant and configuration wiring

**Files:**
- Modify: `backend/pipeline/layers.py`
- Modify: `backend/pipeline/orchestrator.py`
- Modify: `backend/config.yaml`
- Test: `tests/test_group_layers.py`
- Test: `tests/test_pipeline_architecture.py`

**Interfaces:**
- Consumes: Task 1 `recover_missing_member_alpha(...)`.
- Produces: final RGBA alpha that covers every member fallback domain.

- [ ] **Step 1: Write a failing final-guard test**

Mock `refine_alpha_with_colors()` to return an all-zero alpha for a valid
group, then call `extract_object_layers()`. Assert:

```python
assert len(result) == 1
assert np.any(np.asarray(encoded_layer)[..., 3] > 0)
```

For a reconstructed member, also assert a protected replacement-domain pixel
remains transparent.

- [ ] **Step 2: Run the layer test and verify failure**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest -q tests/test_group_layers.py
```

Expected: current code skips the group as `empty_refined_alpha`.

- [ ] **Step 3: Apply the final invariant**

Extend `extract_object_layers()` with:

```python
alpha_presence_threshold: float = 0.05
min_member_alpha_coverage_ratio: float = 0.95
```

After `_clean_final_alpha()` and before real-pixel clipping/bbox extraction,
call:

```python
refined_alpha = recover_missing_member_alpha(
    refined_alpha,
    group,
    roi,
    presence_threshold=alpha_presence_threshold,
    min_coverage_ratio=min_member_alpha_coverage_ratio,
    stage="final_alpha",
)
```

- [ ] **Step 4: Wire configuration through the orchestrator**

Add to `backend/config.yaml`:

```yaml
matting:
  alpha_presence_threshold: 0.05
  min_member_alpha_coverage_ratio: 0.95
```

Pass the two settings to both `refine_objects()` and
`extract_object_layers()` using `.get()` with the same defaults.

- [ ] **Step 5: Run focused verification**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest -q tests/test_group_matting.py tests/test_group_layers.py tests/test_pipeline_architecture.py
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m compileall -q backend tests
git diff --check
```

Expected: focused tests pass, compile succeeds, and `git diff --check` reports
no errors.

- [ ] **Step 6: Commit implementation**

Stage only the scoped implementation, tests, configuration, and plan:

```powershell
git add -- backend/pipeline/matting.py backend/pipeline/layers.py backend/pipeline/orchestrator.py backend/config.yaml tests/test_group_matting.py tests/test_group_layers.py tests/test_pipeline_architecture.py docs/superpowers/plans/2026-07-30-birefnet-alpha-coverage-fallback.md
git commit -m "fix: preserve objects when BiRefNet alpha is incomplete"
```
