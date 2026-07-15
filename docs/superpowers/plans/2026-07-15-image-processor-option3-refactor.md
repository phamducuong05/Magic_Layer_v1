# Image Processor Option 3 Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `backend/image_processor.py` into function-based pipeline modules with explicit model dependencies while preserving runtime output and compatibility imports.

**Architecture:** Stage modules receive processors, adapters, or bound inference callables and never access `model_manager`. A new orchestrator owns lazy model resolution. `backend/image_processor.py` remains a facade with public entry points, dataclass exports, and old private-signature adapters.

**Tech Stack:** Python, dataclasses, NumPy, PIL, OpenCV, PyTorch, pytest.

## Global Constraints

- Do not implement hidden RGB reconstruction or amodal matting.
- Do not modify model configuration or the server checkpoint path.
- Preserve no-detection, no-overlap, exact-tie, modal-matting, and final-background behavior.
- Preserve direct imports of existing dataclasses and private compatibility functions.
- Do not stage or overwrite user changes in `backend/core/helpers.py` or `backend/core/occlusion.py`.
- Leave all changes uncommitted because commit permission was declined.

---

### Task 1: Define extracted contracts and dependency boundaries

**Files:**
- Create: `backend/pipeline/__init__.py`
- Create: `backend/pipeline/types.py`
- Create: `tests/test_pipeline_architecture.py`

**Interfaces:**
- Produces: `ObjectLayer`, `ProcessResult`, and `DetectedObject` with existing fields plus `soft_alpha: Optional[np.ndarray]`.

- [ ] Write a failing test importing all three types from
  `backend.pipeline.types`, constructing a `DetectedObject`, and asserting
  `soft_alpha is None`.
- [ ] Run `pytest tests/test_pipeline_architecture.py -q` and verify an import
  failure because `backend.pipeline` does not exist.
- [ ] Create the package and move-equivalent dataclass definitions into
  `types.py` without modifying `image_processor.py` yet.
- [ ] Re-run the focused test and require it to pass.

### Task 2: Extract segmentation, completion, and reconstruction

**Files:**
- Create: `backend/pipeline/segmentation.py`
- Create: `backend/pipeline/completion.py`
- Create: `backend/pipeline/reconstruction.py`
- Modify: `tests/test_pipeline_architecture.py`

**Interfaces:**
- `extract_objects(image, keywords, processor) -> list[DetectedObject]`
- `link_overlap_partners(objects) -> list[OverlapPair]`
- `get_completion_candidates(objects) -> list[DetectedObject]`
- `complete_objects(image, candidates, completion_model) -> None`
- `apply_pair_decisions(objects, decisions) -> None`
- `build_reconstruction_masks(objects, kernel_size) -> None`

- [ ] Add failing tests proving segmentation uses the supplied fake processor,
  candidate selection preserves order, completion uses the supplied fake
  model, and no stage source contains `model_manager`.
- [ ] Run the focused tests and verify missing-module failures.
- [ ] Move the current algorithms into the three modules, replacing model
  lookup with explicit parameters and preserving mutations/calculations.
- [ ] Run focused tests and `tests/test_image_processor.py` to detect behavioral
  drift before continuing.

### Task 3: Extract matting and rendering with explicit callables

**Files:**
- Create: `backend/pipeline/matting.py`
- Create: `backend/pipeline/layers.py`
- Create: `backend/pipeline/background.py`
- Modify: `tests/test_pipeline_architecture.py`

**Interfaces:**
- `refine_masks(image_np, raw_masks, matte) -> list[np.ndarray]`
- `refine_objects(image_np, objects, matte) -> None`
- `extract_layers(image, image_np, soft_alphas, labels, kernel_size, inpaint) -> list[ObjectLayer]`
- `extract_object_layers(image, image_np, objects, kernel_size, inpaint) -> list[ObjectLayer]`
- `generate_background_from_masks(image, raw_masks, soft_alphas, kernel_size, inpaint) -> Image.Image`
- `generate_final_background(image, objects, kernel_size, inpaint) -> Image.Image`

- [ ] Add failing tests with fake `matte` and `inpaint` callables, asserting
  alpha is attached to its object and rendering functions do not access a
  global manager.
- [ ] Verify missing-module or missing-function failures.
- [ ] Extract the current algorithms. Object-based functions adapt object state
  to the list-based primitives so compatibility behavior has one implementation.
- [ ] Run focused architecture and existing image-processor tests.

### Task 4: Add explicit-dependency orchestration and compatibility facade

**Files:**
- Create: `backend/pipeline/orchestrator.py`
- Rewrite: `backend/image_processor.py`
- Modify: `tests/test_image_processor.py`
- Modify: `tests/test_pipeline_architecture.py`

**Interfaces:**
- `process_masks(image, keywords, manager=model_manager) -> list[np.ndarray]`
- `process_image(image, keywords, manager=model_manager) -> ProcessResult`
- Existing facade function names and signatures remain importable.

- [ ] Add failing orchestrator tests using a fake manager. Verify completion is
  not requested without candidates and inpainting is resolved once then reused
  for layers and background.
- [ ] Implement the orchestrator in the exact current runtime order.
- [ ] Replace `image_processor.py` with facade exports and adapters. Compatibility
  adapters resolve models and call the new explicit-dependency functions.
- [ ] Replace only implementation-coupled orchestration tests with injected
  manager tests; keep their output and ordering assertions.
- [ ] Run `tests/test_pipeline_architecture.py`,
  `tests/test_image_processor.py`, and notebook structural tests.

### Task 5: Repository verification

**Files:**
- Verify all new pipeline modules, facade, and tests.

- [ ] Run `python -m compileall -q backend/pipeline backend/image_processor.py`.
- [ ] Run the full deterministic suite excluding only the acknowledged stale
  checkpoint-path assertion.
- [ ] Run the unfiltered suite and confirm its only failure is still
  `test_completion_configuration_contains_initial_adapter_settings`.
- [ ] Run `git diff --check` on scoped files.
- [ ] Confirm `git status --short` still shows the user's existing helper and
  occlusion changes and that this refactor did not modify their diffs.

### Task 6: Remove transitional compatibility wrappers

**Files:**
- Modify: `examples/partial_pipeline_v1.ipynb`
- Modify: `backend/image_processor.py`
- Modify: `tests/test_image_processor.py`
- Modify: `tests/test_partial_pipeline_notebook.py`

**Interfaces:**
- Notebook imports explicit functions from `backend.pipeline` stage modules.
- `backend.image_processor` exports only `DetectedObject`, `ObjectLayer`,
  `ProcessResult`, `process_image`, and `process_masks`.

- [ ] Update structural tests to require explicit stage imports and reject
  private imports from `backend.image_processor`; run them RED against the
  transitional facade/notebook.
- [ ] Migrate direct private-wrapper tests to call segmentation, completion,
  reconstruction, matting, layer, and background functions with explicit fake
  dependencies.
- [ ] Update the notebook stage calls and lazy model resolution, storing alpha
  on each object and reusing one inpainting callable.
- [ ] Replace `image_processor.py` with public orchestrator/type re-exports.
- [ ] Run notebook, image-pipeline, architecture, compile, and full deterministic
  verification while preserving the acknowledged checkpoint-path mismatch.
