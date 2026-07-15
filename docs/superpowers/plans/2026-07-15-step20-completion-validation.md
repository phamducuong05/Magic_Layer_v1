# Step 20 Completion Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent malformed or failed SDAmodal completion outputs from reaching role assignment and reconstruction.

**Architecture:** `pipeline/completion.py` owns canonical binary-mask validation and modal fallback because it already maps model outputs to `DetectedObject` records. The orchestrator reads pipeline-owned limits and passes them explicitly, preserving lazy model resolution and allowing valid objects to survive another object's invalid output.

**Tech Stack:** Python, NumPy, Pillow, YAML, pytest.

## Global Constraints

- Implement Step 20 only; noise floors, tie tolerance, reconstruction fallback, and structured diagnostics remain later steps.
- Use `max_area_growth_ratio: 4.0` where ratio means `amodal_area / modal_area`.
- Use `max_bbox_growth_ratio: 9.0` where ratio means `amodal_bbox_area / modal_bbox_area`.
- A fallback object receives its binary modal mask, an empty boolean completion-hole mask, and completion-hole area `0`.
- Batch inference exceptions and output-count mismatches fall back every candidate.
- One invalid output must not discard other valid outputs from the same correctly sized batch.
- Preserve the intentionally configured SDAmodal checkpoint path.

---

### Task 1: Per-object output validation

**Files:**
- Modify: `backend/pipeline/completion.py`
- Modify: `tests/test_image_processor.py`
- Modify: `tests/test_pipeline_architecture.py`

**Interfaces:**
- `complete_objects(image, candidates, completion_model, *, max_area_growth_ratio, max_bbox_growth_ratio) -> None`.
- Valid outputs are stored as full-image boolean masks.
- Invalid outputs use modal fallback without raising.

- [ ] **Step 1: Write failing validation tests**

Add public-behavior tests for wrong shape, non-numeric dtype, non-finite values, empty support, missing modal pixels, excessive total area growth, excessive bounding-box growth, and a mixed batch where one valid object remains completed while another falls back.

- [ ] **Step 2: Run tests and verify RED**

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q tests\test_image_processor.py -k "completion_rejects or completion_preserves_valid"
```

Expected: failures because current code stores malformed outputs directly or raises while computing holes.

- [ ] **Step 3: Implement validation and fallback helpers**

Add focused private helpers that canonicalize real numeric/bool arrays, compute tight bounding-box area, compare configured ratios, store valid holes, and store modal fallback. Do not union missing modal pixels into a prediction because preservation is an adapter contract that must be verified.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and expect every selected test to pass.

### Task 2: Batch contract and shared-inference fallback

**Files:**
- Modify: `backend/pipeline/completion.py`
- Modify: `tests/test_image_processor.py`

**Interfaces:**
- A completion exception or non-sequence/count-mismatched output falls back all candidates and returns normally.
- A correctly sized batch continues through independent per-object validation.

- [ ] **Step 1: Write failing batch-fallback tests**

Add tests where `completion_model.complete` raises and where it returns too few masks. Assert every candidate receives modal fallback fields and no exception escapes.

- [ ] **Step 2: Run tests and verify RED**

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q tests\test_image_processor.py -k "completion_batch_failure or completion_output_count"
```

Expected: the exception escapes or the final candidate retains unset completion fields.

- [ ] **Step 3: Implement batch fallback**

Wrap the shared model call, require a sequence with exactly one item per candidate, log an ordinary warning, apply modal fallback to all candidates, and return without resolving later stages differently.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and expect both tests to pass.

### Task 3: Configuration and orchestrator wiring

**Files:**
- Modify: `backend/config.yaml`
- Modify: `backend/pipeline/orchestrator.py`
- Modify: `tests/test_image_processor.py`
- Modify: `tests/test_pipeline_architecture.py`

**Interfaces:**
- `config.get_pipeline_config("completion")` returns both approved limits.
- `_complete_candidates` passes both limits explicitly without initializing completion when there are no overlap candidates.

- [ ] **Step 1: Write failing configuration and wiring assertions**

Assert the exact YAML values, explicit keyword arguments, and unchanged no-overlap lazy bypass.

- [ ] **Step 2: Run tests and verify RED**

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q tests\test_image_processor.py tests\test_pipeline_architecture.py -k "completion_validation_config or completion_candidates"
```

Expected: configuration and explicit validation arguments are absent.

- [ ] **Step 3: Add settings and wire them into completion**

Add the `pipeline.completion` mapping and pass `float(...)` values to `complete_objects` only after non-empty candidates are selected.

- [ ] **Step 4: Run focused and full verification**

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q tests\test_image_processor.py tests\test_pipeline_architecture.py
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q
git diff --check
```

Expected: Step 20-focused tests pass. The full suite may retain only the documented stale checkpoint-path expectation failure.
