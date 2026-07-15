# Amodal Occlusion Integration: Coding Progress

## 1. Final Goal

Extend the existing image-layer pipeline so that it can reconstruct an object
that is partially hidden by another detected object.

The required runtime order is:

1. Segment objects with SAM3 and group masks by semantic class.
2. Calculate one modal bounding box for every grouped object.
3. Find positive-area bounding-box overlaps between different classes.
4. Skip amodal completion completely when no cross-class overlap exists.
5. Complete every unique object involved in an overlap once.
6. Calculate each object's newly completed area.
7. Compare completed areas independently for every overlap pair.
8. Treat the larger-area object as occluded and the other as its occluder.
9. Build a binary reconstruction mask from the completion hole and the
   spatially relevant part of every assigned occluder.
10. Reconstruct the occluded object's hidden RGB content with ROI-based
    inpainting driven by that hard mask.
11. Run BiRefNet on an expanded square crop. Ordinary and fallback objects use
    the original RGB image with modal support; successfully reconstructed
    objects use reconstructed RGB with amodal support.
12. Recover clean foreground colors, export completed RGBA object layers, and
    preserve the existing original-image final-background behavior.

The workflow must not use an explicit occlusion signal, class heuristic, or
global depth ordering.

## 2. Current Verified State

- Implementation currently reaches reconstruction-mask construction.
- Hidden RGB reconstruction has not been implemented yet.
- BiRefNet still processes the original image and modal masks for every object.
- Matting and per-layer background estimation still run on full-image inputs;
  expanded square per-object ROIs have not been implemented yet.
- The image-processing implementation has been split into explicit
  `backend/pipeline/` stages. `backend/image_processor.py` is now only the
  small public compatibility facade.
- The latest verification after that refactor produced **80 passing tests and
  one known checkpoint-path expectation failure**. The configured server path
  is intentionally preserved; the stale local test expectation must not drive
  a production-path change.
- Real SDAmodal/DIFT checkpoint quality has not yet been evaluated in this
  integration. Checkpoints remain assumed available and correctly loadable.

## 3. Completed Changes

### 3.1 Grouped object records and bounding boxes

`backend/pipeline/segmentation.py` converts grouped SAM3 results into
`DetectedObject` records defined in `backend/pipeline/types.py`. Each record
preserves:

- Stable object ID.
- Semantic class.
- Display label.
- Full-image modal mask.
- Existing tight modal bounding box in `(x, y, width, height)` format.
- Cross-class overlap partners.
- Assigned occluders.
- Amodal mask and completion-hole information.
- Reconstruction mask.
- Per-object soft alpha once matting has completed.

Bounding boxes are calculated after same-class mask grouping. Overlap checks
therefore operate on grouped semantic objects, not on pre-grouping components.

### 3.2 Cross-class overlap graph

`backend/core/occlusion.py` now provides pure overlap geometry:

- Every unordered object pair is considered once.
- Same-class pairs are ignored.
- Edge or corner contact is ignored.
- Only positive-area bounding-box intersection creates an overlap edge.

`backend/pipeline/completion.py::link_overlap_partners()` records every edge in
both participating
`DetectedObject.overlap_partner_ids` sets.

### 3.3 Completion model architecture

The model system now includes a `completion` category:

- `BaseCompletionModel` defines the batch completion contract.
- `ModelRegistry` supports completion adapters.
- `ModelManager` lazily creates and caches the configured completion model.
- Completion is excluded from normal startup warm-up.
- `backend/config.yaml` selects the `sdamodal` adapter and identifies its
  configuration and checkpoint paths.

The application-facing completion contract is:

```text
complete(image, modal_masks, bboxes) -> amodal_masks
```

Mask and bounding-box order must remain aligned from input to output.

### 3.4 SDAmodal package and device isolation

The SDAmodal inference path was made package-safe and device-aware:

- Production imports use the application package instead of ambiguous
  top-level research-module names.
- Package initializers were added where required.
- Hard-coded CUDA placement was removed from the production completion path.
- Model loading uses the selected application device.
- Checkpoint loading is device portable.
- DIFT and SDAmodal receive the same selected device.

### 3.5 In-memory DIFT extraction

`DIFTFeatureExtractor` now accepts a PIL image and returns the four required
feature levels in memory.

- Request-time feature files are not written or read.
- DIFT features are extracted once for a non-empty completion batch.
- The same feature pyramid is shared by all completion candidates.
- Empty completion batches skip DIFT entirely.

### 3.6 Modal-mask and bounding-box preparation

`backend/models/completion/mask_inputs.py` prepares SDAmodal inputs while
preserving the grouped bounding boxes already produced by the segmentation
stage.

- It does not recalculate tight boxes from the masks.
- It validates mask dimensions and mask/box count alignment.
- It expands the supplied boxes using SDAmodal's configured `enlarge_box`
  value.
- It preserves full-image mask order.

### 3.7 SDAmodal loading and batch completion

The completion package now contains:

- A reusable SDAmodal model loader.
- A batch-completion function that consumes an existing DIFT pyramid.
- Full-image restoration of predicted patches.
- Union with the original modal mask so visible pixels cannot disappear.
- A registered `SDAmodalCompletionModel` adapter that connects model loading,
  DIFT extraction, and batch completion.

### 3.8 Conditional completion in the image pipeline

`backend/pipeline/completion.py::get_completion_candidates()` and
`complete_objects()` now:

- Selects only objects with at least one cross-class overlap partner.
- Includes each object once even when it has multiple partners.
- Skips completion-model initialization when no overlap exists.
- Calls SDAmodal once with the whole candidate batch.
- Maps returned amodal masks back to the corresponding object records.

### 3.9 Completion holes and areas

For every completed object, the pipeline stores:

```text
completion_hole_mask = (amodal_mask > 0) AND (modal_mask == 0)
completion_hole_area = count(completion_hole_mask)
```

The calculation works with boolean, `0/1`, and `0/255` masks. Objects that do
not run through completion currently retain `None` completion fields.

### 3.10 Pairwise role decisions

`assign_pair_roles()` compares hole areas independently for every overlap
edge:

- Larger first area: first object is occluded.
- Larger second area: second object is occluded.
- Equal areas: the pair is ambiguous and receives no roles.

No global depth order is constructed. An object may be occluded in one pair
and act as an occluder in another pair.

`backend/pipeline/reconstruction.py::apply_pair_decisions()` consumes these
decisions and stores unique assigned occluders in each occluded object's
`occluder_ids`. Ambiguous decisions do not add an occluder.

### 3.11 Reconstruction-mask construction

`backend/pipeline/reconstruction.py::build_reconstruction_masks()` creates one
full-image boolean mask for every object with assigned occluders:

1. Union all assigned occluders' modal masks.
2. Expand the occluded object's amodal support with the existing image-derived
   kernel.
3. Keep only occluder pixels within that expanded support.
4. Exclude the occluded object's known modal pixels.
5. Union the result with the completion-hole mask.

This supports multiple occluders in one future inpainting call, protects known
visible pixels, and excludes distant portions of large occluders.

The exact construction is:

```text
completion_hole = amodal_mask AND NOT modal_mask

relevant_occluder =
    union(assigned_occluder_modal_masks)
    AND expanded_amodal_support
    AND NOT modal_mask

reconstruction_mask = completion_hole OR relevant_occluder
```

`reconstruction_mask` is a hard boolean mask. It is not a soft alpha matte and
must remain separate from the later matting result.

### 3.12 Pipeline-module refactor

The former monolithic `backend/image_processor.py` implementation is now split
into explicit stages:

- `pipeline/segmentation.py`: grouped object extraction.
- `pipeline/completion.py`: overlap links and amodal completion.
- `pipeline/reconstruction.py`: pair-role application and reconstruction-mask
  construction.
- `pipeline/matting.py`: per-object alpha generation.
- `pipeline/layers.py`: foreground recovery and RGBA layer export.
- `pipeline/background.py`: final original-image background generation.
- `pipeline/orchestrator.py`: dependency injection and runtime ordering.

`backend/image_processor.py` exports only the supported public API and data
types. New implementation work should target the owning pipeline module, not
restore private compatibility wrappers to the facade.

## 4. Important Gaps in the Current Implementation

The following requirements from `second_plan.md` are not complete and must not
be treated as finished:

- Completion output count, shape, dtype, and finite-value validation.
- Per-object fallback when a completion result is invalid.
- Request-level fallback when DIFT or batch completion fails.
- Configurable completion-growth validation.
- Configurable minimum meaningful hole area.
- Configurable near-tie tolerance; current logic handles exact ties only.
- Reconstruction-specific inpainting configuration and prompt policy.
- Hidden RGB reconstruction and reconstruction-result validation.
- Modal fallback when reconstruction fails.
- Shared expanded-square ROI geometry, boundary padding, and coordinate
  restoration.
- Per-object BiRefNet source/support selection and cropped inference.
- Modal or amodal inference ROIs, distinct from final refined-alpha export
  bounds.
- Layer extraction from the same reconstructed canvas used for matting.
- ROI-based temporary background inpainting for foreground-color recovery.
- Structured diagnostics, timings, and fallback reasons.
- Real-model acceptance testing and GPU-memory measurement.

Also note that the `input_size` and `enlarge_box` values under
`backend/config.yaml` must be reconciled with the values read from
`config_SDAmodal.yaml`; there should be one authoritative runtime source.

## 5. Remaining Implementation Steps

Each step below should remain a separate review checkpoint: present the small
plan, obtain approval, implement with tests, and wait for verification before
continuing.

### Step 19: Reconstruct hidden RGB

- Add reusable ROI helpers that calculate a support bbox, expand it by a
  configurable context ratio, convert it to a square, handle image-boundary
  padding, crop aligned images/masks, and restore crop coordinates.
- Add `reconstruction_crop` and `reconstruction_roi` to `DetectedObject` (or
  use an equivalent single typed reconstruction-result record). Avoid storing
  one unnecessary full-image copy per object.
- Select only objects whose `reconstruction_mask` exists and is non-empty.
  The presence of `amodal_mask` alone is not a reconstruction trigger because
  both members of an overlap pair run through completion.
- Calculate the reconstruction ROI from `amodal_mask`, then crop the original
  image and `reconstruction_mask` with exactly the same ROI.
- Convert the cropped reconstruction mask to a binary `0/255` PIL `L` image.
  Never pass `soft_alpha` directly to an inpainting model.
- Call the reconstruction-capable inpainting backend once per selected object
  with the original RGB crop, hard reconstruction mask, and semantic-class
  prompt context where supported.
- Store the normalized reconstructed RGB crop and its full-image ROI.
- Do not run reconstruction for ordinary, ambiguous, or empty-mask objects.

### Step 20: Harden completion output validation and fallback

- Require one output per requested completion object.
- Validate full-image shape, finite values, non-empty masks, and modal-pixel
  preservation.
- Enforce configured area-growth and bounding-box-growth limits.
- Fall back invalid objects to their modal masks with zero effective hole area.
- Fall back all completion candidates safely if shared DIFT/batch inference
  fails.
- Continue processing unaffected objects whenever possible.

This hardening must be completed before reconstructed objects are allowed to
enter the user-visible matting and layer-output path.

### Step 21: Add noise-floor and tie-tolerance configuration

- Add an absolute and/or modal-area-relative minimum meaningful hole area.
- Add a configurable tie tolerance.
- Compare effective areas rather than raw boundary-growth noise.
- Mark equal or near-equal pairs ambiguous.
- Keep raw areas available for diagnostics.

### Step 22: Separate reconstruction inpainting configuration

- Add reconstruction-specific prompt and mask/blending settings.
- Keep these settings separate from final-background removal settings.
- Let LaMa ignore textual prompts while SDXL receives object-aware context.
- Verify reconstruction calls and final-background calls use their intended
  policies.
- Keep reconstruction inpainting distinct from the later temporary background
  inpainting used to recover clean foreground colors for RGBA export.

### Step 23: Validate reconstruction and provide modal fallback

- Validate returned image type, ROI dimensions, and RGB conversion.
- Ensure pixels outside the permitted hard-mask/blend region remain unchanged
  inside the crop.
- Confirm the completion-hole region contains usable RGB data.
- On failure, clear the reconstruction crop/result and mark the object for its
  ordinary original-image/modal-mask path.
- Ensure one object's failure does not discard other successful objects.

### Step 24: Refactor BiRefNet to per-object inputs

- Replace the shared full-image/raw-mask-list matting interface with
  object-based cropped matting inputs.
- Ordinary, ambiguous, or failed objects use an expanded square crop from the
  original image and `modal_mask` support.
- Successfully reconstructed objects use their reconstructed RGB crop and
  aligned `amodal_mask` support.
- Use the selected support only to choose the ROI, optionally guide the input,
  and constrain output. The current BiRefNet adapter receives an RGB image, not
  a second mask argument.
- Preserve a small amount of real surrounding RGB context instead of zeroing
  every pixel outside the hard support before inference.
- Resize the returned alpha to the unpadded ROI, constrain it to a narrow
  dilation of the selected modal/amodal support, and paste it into a zero-valued
  full-image alpha canvas.
- Store the result as `DetectedObject.soft_alpha` and retain the exact source
  crop/ROI identity required by layer extraction.
- Prove reconstructed objects enter BiRefNet only after reconstruction is
  complete. Ordinary objects still run BiRefNet directly on their original RGB
  crops.

### Step 25: Refactor object-layer extraction

- Consume `DetectedObject` records instead of synchronized masks and labels.
- Use the same original or reconstructed RGB crop that was used for each
  object's matting pass.
- Build the temporary background-estimation inpainting mask as a hard binary
  mask from the selected modal/amodal support, optionally unioned with
  thresholded alpha coverage. Do not pass fractional `soft_alpha` directly to
  inpainting.
- Run this temporary background inpainting on the per-object ROI instead of
  repeatedly processing the full source image. This pass removes the whole
  object to estimate its background; it is separate from hidden-object
  reconstruction, which removes only `reconstruction_mask`.
- Run foreground-color recovery with the selected source crop, its inpainted
  background estimate, and the soft alpha.
- Calculate the tight export bbox from the final `refined_alpha`, after alpha
  and color refinement. Do not reuse the square inference ROI as the exported
  layer bounds.
- Map the tight crop back to full-image `(x, y)` coordinates and preserve
  display labels.
- Prevent foreground-color refinement from reintroducing pixels from the
  original occluder-filled image.

### Step 26: Preserve final-background semantics

- Keep final background inpainting based on the original source image.
- Base the removal union on original modal object coverage and only soft-alpha
  coverage associated with the originally visible modal support.
- Do not blindly add amodal completion holes to the global removal mask.
- In particular, do not union a reconstructed object's full amodal
  `soft_alpha` into the original-image background-removal mask. Constrain that
  coverage to a narrow dilation of its original `modal_mask` or retain a
  separate visible-alpha coverage mask.
- Ensure per-object reconstruction crops/results never become the
  final-background input.
- Keep a single final background-inpainting pass.

### Step 27: Add diagnostics and observability

- Record object count, overlap-edge count, and unique completion-candidate
  count.
- Record modal, amodal, raw-hole, and effective-hole areas.
- Record pair decisions and ambiguity reasons.
- Record completion, reconstruction, and matting timings.
- Record fallback stage and reason without logging image data or features.
- Optionally record peak GPU memory in diagnostic mode.

### Step 28: Add end-to-end deterministic tests

Cover at minimum:

- No-overlap bypass.
- Either member of a two-object pair becoming occluded.
- Exact and near ties.
- Same-class overlap bypass.
- One object with multiple occluders.
- Three-object overlap chains.
- Invalid completion output.
- Completion-batch failure.
- Reconstruction failure.
- Ordinary and reconstructed matting paths.
- Expanded square ROI creation, border padding, and full-image coordinate
  restoration.
- Amodal versus modal layer bounds.
- Hard-mask enforcement at both reconstruction and background-estimation
  inpainting boundaries.
- Final refined-alpha export bounds without clipping newly recovered edges.
- Final background isolation from per-object reconstruction results.

### Step 29: Run real-model acceptance and performance evaluation

- Use a fixed set of representative occlusion scenes.
- Inspect modal masks, amodal masks, holes, role decisions, reconstruction
  masks, reconstructed canvases, alpha mattes, layers, and final backgrounds.
- Tune only documented thresholds and expansion/blending settings.
- Measure completion latency, reconstruction latency, total request latency,
  and peak GPU memory.
- Consider mixed precision or offloading only after output-equivalence checks.

## 6. Final Acceptance Criteria

The integration is complete only when all of the following are true:

- Cross-class grouped-box overlap is the only completion trigger.
- Both members of every overlap pair are completed once per request.
- No-overlap requests do not initialize or run DIFT/SDAmodal.
- Occluded/occluder roles depend only on validated completion-hole areas.
- Ambiguous pairs do not trigger destructive reconstruction.
- Multiple assigned occluders produce one constrained reconstruction pass per
  occluded object.
- Successful hidden-object reconstruction happens before BiRefNet.
- Every inpainting adapter receives a binary hard mask; `soft_alpha` is reserved
  for alpha refinement, foreground recovery, and RGBA compositing.
- Matting uses expanded square per-object ROIs and restores alpha to full-image
  coordinates.
- Ordinary and fallback objects use original RGB with modal support.
- Successfully completed layers use reconstructed RGB with amodal support.
- Layer export uses tight bounds from final refined alpha, not the square
  inference ROI or a pre-refinement mask.
- The final background still comes from the original image and visible-object
  removal coverage.
- Failure paths return safe modal results instead of malformed completed
  layers.
- Deterministic tests pass and real-checkpoint acceptance scenes have been
  manually reviewed.

## 7. Local Verification Command

Run the full deterministic suite with:

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q
```

Most recent verified result before this documentation update:

```text
80 passed, 1 known checkpoint-path expectation failure
```

The remaining failure reflects a stale local expected checkpoint path. The
runtime path is intentionally configured to match the server and should remain
unchanged.
