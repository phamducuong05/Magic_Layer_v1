# Background Inpainting Diagnostics and Morphology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save the final-background image after LaMa, after composition blending, and after palette refinement while reducing background dilation and ensuring morphology kernels are centered odd dimensions.

**Architecture:** Background adapters report in-memory stage images through an optional callback; the pipeline owns artifact paths and numbering. Background mask dilation is configured separately from the kernel used by foreground-layer extraction. Shared NumPy/OpenCV morphology normalizes dynamic kernels to positive odd dimensions without enlarging even requests.

**Tech Stack:** Python, NumPy, Pillow, OpenCV, pytest, YAML.

## Global Constraints

- Keep `refine_background` enabled and behaviorally unchanged.
- Save `01_after_lama.png`, `02_after_composition_blend.png`, and `03_after_palette_refine.png`.
- Do not commit or stage changes.
- Preserve backward compatibility when diagnostics are disabled.

---

### Task 1: Center morphology kernels

**Files:**
- Modify: `backend/core/layerd_refine.py`
- Modify: `backend/core/helpers.py`
- Test: `tests/test_image_processor.py`

**Interfaces:**
- Produces: `normalize_morphology_kernel(kernel_size: tuple[int, int] | int) -> tuple[int, int]`
- Consumes: integer or `(height, width)` morphology dimensions.

- [ ] **Step 1: Write failing tests**

Add tests proving that a dynamic `4 x 6` kernel becomes `3 x 5`, a `1 x 1` kernel is unchanged, and `_calc_kernel_size` returns odd dimensions.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_image_processor.py -k "morphology_kernel or calc_kernel_size" -q`; expect failure because normalization is absent.

- [ ] **Step 3: Implement minimal normalization**

Normalize each dimension with `max(1, value)` followed by subtraction of one when even. Apply it inside `expand_mask`, `shrink_mask`, and `_calc_kernel_size`.

- [ ] **Step 4: Verify GREEN**

Run the same focused pytest command and expect all selected tests to pass.

### Task 2: Save three background stages

**Files:**
- Modify: `backend/models/base.py`
- Modify: `backend/models/background_inpainting/original_lama/adapter.py`
- Modify: `backend/models/background_inpainting/simple_lama/adapter.py`
- Modify: `backend/models/background_inpainting/sdxl/adapter.py`
- Modify: `backend/pipeline/background.py`
- Modify: `backend/pipeline/orchestrator.py`
- Test: `tests/test_original_lama_adapter.py`
- Test: `tests/test_image_processor.py`

**Interfaces:**
- Produces: optional `artifact_callback(stage: str, image: Image.Image) -> None` on background adapters.
- Produces: `diagnostics_directory` argument on final-background generation.

- [ ] **Step 1: Write failing adapter test**

Use a fake Original LaMa runtime and assert callbacks receive `after_lama` before `after_composition_blend`, with the first image raw and the second source-preserving.

- [ ] **Step 2: Verify adapter RED**

Run `pytest tests/test_original_lama_adapter.py -k artifact -q`; expect failure because `artifact_callback` is unsupported.

- [ ] **Step 3: Implement callback support**

Extend the base and all three adapters with the optional callback. Emit after model inference and after `preserve_unmasked_pixels`.

- [ ] **Step 4: Write failing pipeline artifact test**

Run `generate_background_from_masks` against a temporary directory and assert exact files `01_after_lama.png`, `02_after_composition_blend.png`, and `03_after_palette_refine.png` contain the expected stage colors.

- [ ] **Step 5: Verify pipeline RED**

Run the selected pipeline test and expect failure because no diagnostics directory is accepted or written.

- [ ] **Step 6: Implement pipeline-owned saving**

Create the directory only when configured, map callback stages to numbered filenames, save the palette-refined output last, and pass the configured directory from the orchestrator.

- [ ] **Step 7: Verify GREEN**

Run both focused test files and expect all tests to pass.

### Task 3: Reduce background dilation through configuration

**Files:**
- Modify: `backend/config.yaml`
- Modify: `backend/pipeline/orchestrator.py`
- Test: `tests/test_image_processor.py`

**Interfaces:**
- Consumes: `pipeline.background_inpainting.mask_dilation_scale`.
- Consumes: `pipeline.background_inpainting.diagnostics_directory`.

- [ ] **Step 1: Write failing orchestration test**

Assert final-background generation receives a separately calculated kernel while foreground layer extraction retains its existing kernel.

- [ ] **Step 2: Verify RED**

Run the focused orchestration test and expect failure because one shared kernel is still passed to both calls.

- [ ] **Step 3: Implement configuration**

Add `mask_dilation_scale: 0.00375` and `diagnostics_directory: outputs/background_debug`. Change active Original LaMa expansions from `21/15` to `11/7`; calculate only the final-background kernel from the new scale.

- [ ] **Step 4: Verify GREEN and regressions**

Run focused tests, then the project test suite. Confirm only previously known baseline failures remain.

