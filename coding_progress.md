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
9. Reconstruct the occluded object's hidden RGB content with inpainting.
10. Run BiRefNet only after reconstruction for successfully reconstructed
    objects.
11. Return completed RGBA object layers while preserving the existing final
    background-removal behavior.

The workflow must not use an explicit occlusion signal, class heuristic, or
global depth ordering.

## 2. Current Verified State

- Implementation currently reaches reconstruction-mask construction.
- Hidden RGB reconstruction has not been implemented yet.
- BiRefNet still processes the original image and modal masks for every object.
- The latest deterministic test run passes: **67 tests passed**.
- Real SDAmodal/DIFT checkpoint quality has not yet been evaluated in this
  integration. Checkpoints remain assumed available and correctly loadable.

## 3. Completed Changes

### 3.1 Grouped object records and bounding boxes

`backend/image_processor.py` now converts grouped SAM3 results into
`DetectedObject` records. Each record preserves:

- Stable object ID.
- Semantic class.
- Display label.
- Full-image modal mask.
- Existing tight modal bounding box in `(x, y, width, height)` format.
- Cross-class overlap partners.
- Assigned occluders.
- Amodal mask and completion-hole information.
- Reconstruction mask.

Bounding boxes are calculated after same-class mask grouping. Overlap checks
therefore operate on grouped semantic objects, not on pre-grouping components.

### 3.2 Cross-class overlap graph

`backend/core/occlusion.py` now provides pure overlap geometry:

- Every unordered object pair is considered once.
- Same-class pairs are ignored.
- Edge or corner contact is ignored.
- Only positive-area bounding-box intersection creates an overlap edge.

`_link_overlap_partners()` records every edge in both participating
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
preserving the grouped bounding boxes already produced by
`image_processor.py`.

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

`_complete_overlapping_objects()` now:

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

`_apply_pair_decisions()` consumes these decisions and stores unique assigned
occluders in each occluded object's `occluder_ids`. Ambiguous decisions do not
add an occluder.

### 3.11 Reconstruction-mask construction

`_build_reconstruction_masks()` creates one full-image boolean mask for every
object with assigned occluders:

1. Union all assigned occluders' modal masks.
2. Expand the occluded object's amodal support with the existing image-derived
   kernel.
3. Keep only occluder pixels within that expanded support.
4. Exclude the occluded object's known modal pixels.
5. Union the result with the completion-hole mask.

This supports multiple occluders in one future inpainting call, protects known
visible pixels, and excludes distant portions of large occluders.

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
- Per-object BiRefNet canvas/support selection.
- Amodal bounding boxes for reconstructed layer cropping.
- Layer extraction from the same reconstructed canvas used for matting.
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

- Add `reconstruction_canvas` to `DetectedObject`.
- Select only objects with non-empty reconstruction masks.
- Call the existing inpainting model once per selected object using the
  original image, its reconstruction mask, and semantic-class prompt context.
- Normalize the result to full-size RGB and store it on that object.
- Do not run reconstruction for ordinary or ambiguous objects.

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

### Step 23: Validate reconstruction and provide modal fallback

- Validate returned image type, dimensions, and RGB conversion.
- Ensure pixels outside the permitted blend region remain unchanged.
- Confirm the completion-hole region contains usable RGB data.
- On failure, clear the reconstruction canvas and mark the object for its
  ordinary original-image/modal-mask path.
- Ensure one object's failure does not discard other successful objects.

### Step 24: Refactor BiRefNet to per-object inputs

- Replace the shared-image/raw-mask-list matting interface with object-based
  matting inputs.
- Ordinary, ambiguous, or failed objects use the original image and modal
  support.
- Successfully reconstructed objects use their reconstruction canvas and
  amodal support.
- Constrain alpha to a narrow dilation of the selected support.
- Prove BiRefNet runs only after reconstruction is complete.

### Step 25: Refactor object-layer extraction

- Consume `DetectedObject` records instead of synchronized masks and labels.
- Use the same canvas that was used for each object's matting pass.
- Use modal bounds for ordinary/fallback objects.
- Calculate and use amodal bounds for successfully reconstructed objects.
- Preserve display labels and full-image offsets.
- Prevent foreground-color refinement from reintroducing pixels from the
  original occluder-filled image.

### Step 26: Preserve final-background semantics

- Keep final background inpainting based on the original source image.
- Base the removal union on original modal object coverage and resulting soft
  alpha coverage.
- Do not blindly add amodal completion holes to the global removal mask.
- Ensure per-object reconstruction canvases never become the final-background
  input.
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
- Amodal versus modal layer bounds.
- Final background isolation from reconstruction canvases.

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
- Ordinary and fallback objects preserve their existing modal behavior.
- Completed layers use reconstructed RGB, amodal alpha support, and amodal
  bounds.
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

Current verified result when this document was created:

```text
67 passed
```
