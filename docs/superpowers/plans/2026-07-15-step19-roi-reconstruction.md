# Step 19 ROI Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct hidden RGB from crop-sized, padded square ROIs instead of full-image inpainting inputs.

**Architecture:** A reusable `SquareROI` value object owns unclamped full-image geometry, padding, aligned crop operations, and restoration. Reconstruction derives the ROI from each selected object's amodal support, inpaints the aligned RGB/mask crops once, and stores the reconstructed crop plus ROI metadata on the object.

**Tech Stack:** Python, dataclasses, NumPy, Pillow, pytest, YAML.

## Global Constraints

- Implement Step 19 only; do not add completion fallback, reconstruction validation, or cropped matting behavior from later steps.
- Use `pipeline.reconstruction.context_ratio: 0.25`.
- `reconstruction_canvas` stores a square RGB crop, never a full-image copy.
- Inpainting masks must be binary `0/255` PIL `L` images derived from `reconstruction_mask`.
- Reconstruction remains before BiRefNet.
- Preserve the intentionally configured SDAmodal checkpoint path.

---

### Task 1: Reusable square ROI geometry

**Files:**
- Create: `backend/pipeline/roi.py`
- Test: `tests/test_image_processor.py`

**Interfaces:**
- Produces: `SquareROI(x, y, size, image_width, image_height)`.
- Produces: `square_roi_from_support(support, context_ratio) -> SquareROI`.
- Produces: `crop_image(image, roi)`, `crop_array(array, roi)`, and `restore_array(crop, roi)`.

- [ ] **Step 1: Write failing ROI tests**

Add tests that assert a centered support expands by 25%, a border ROI reports padding, image and mask crops stay aligned, and `restore_array` removes padding and restores full-image coordinates.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q tests\test_image_processor.py -k "square_roi or roi_crop"
```

Expected: collection/import failure because `backend.pipeline.roi` does not exist.

- [ ] **Step 3: Implement the minimal ROI module**

Implement an immutable dataclass with `box`, `clipped_box`, `padding`, and `inner_box` properties. Compute the square side as `ceil(max(tight_width, tight_height) * (1 + 2 * context_ratio))`, center it on the support bbox, allow negative/out-of-image coordinates, and use constant-zero padding for aligned crops.

- [ ] **Step 4: Run ROI tests and verify GREEN**

Run the command from Step 2 and expect all selected tests to pass.

### Task 2: Crop-sized hidden-RGB reconstruction

**Files:**
- Modify: `backend/pipeline/types.py`
- Modify: `backend/pipeline/reconstruction.py`
- Modify: `tests/test_image_processor.py`

**Interfaces:**
- `DetectedObject.reconstruction_canvas: Optional[Image.Image]` is the padded square RGB crop.
- `DetectedObject.reconstruction_roi: Optional[SquareROI]` maps that crop to the source image.
- `reconstruct_objects(image, objects, inpaint, *, context_ratio)` performs selective reconstruction.

- [ ] **Step 1: Replace the existing full-image reconstruction test**

Assert that only a non-empty reconstruction mask triggers inpainting, the ROI is derived from `amodal_mask`, the RGB and hard-mask inputs have identical square dimensions, the mask contains only `0/255`, and the normalized result and ROI are stored.

- [ ] **Step 2: Run the reconstruction test and verify RED**

Run:

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q tests\test_image_processor.py -k reconstruct_objects_inpaints
```

Expected: failure because the current implementation passes a full-image input and stores no ROI.

- [ ] **Step 3: Implement crop-sized reconstruction**

Clear stale reconstruction state, skip `None` or empty masks, derive a `SquareROI` from `amodal_mask`, crop the original RGB image and reconstruction mask with the same ROI, convert the mask to PIL `L`, invoke inpainting once with semantic context, resize the result to the crop size if needed, convert it to RGB, and store it with the ROI.

- [ ] **Step 4: Run the reconstruction test and verify GREEN**

Run the command from Step 2 and expect it to pass.

### Task 3: Configuration and orchestration

**Files:**
- Modify: `backend/config.yaml`
- Modify: `backend/config.py`
- Modify: `backend/pipeline/orchestrator.py`
- Modify: `tests/test_image_processor.py`

**Interfaces:**
- `ConfigManager.get_pipeline_config(stage) -> Dict[str, Any]` returns non-model pipeline settings.
- `process_image` supplies the configured ratio to `reconstruct_objects` before resolving/running matting.

- [ ] **Step 1: Write failing configuration and wiring assertions**

Assert the YAML ratio is `0.25` and the orchestrator passes it to reconstruction before matting.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q tests\test_image_processor.py -k "reconstruction_context or process_image_coordinates"
```

Expected: failure because the pipeline setting and keyword argument are absent.

- [ ] **Step 3: Add configuration and wire it into orchestration**

Add the YAML block, add a safe dictionary-returning configuration accessor, and pass `context_ratio=float(config.get_pipeline_config("reconstruction")["context_ratio"])` into `reconstruct_objects`.

- [ ] **Step 4: Run focused and full verification**

Run:

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q tests\test_image_processor.py tests\test_pipeline_architecture.py
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q
git diff --check
```

Expected: focused tests pass. The full suite may retain only the documented stale checkpoint-path expectation failure.
