# Amodal Occlusion Integration: Coding Progress

## 1. Final Goal

Extend the existing image-layer pipeline so that it can reconstruct an object
that is partially hidden by another detected object.

The required runtime order is:

1. Segment raw masks with SAM3 without grouping them yet.
2. Preserve every raw modal mask and calculate its original modal bounding
   box before completion.
3. Find positive-area cross-class overlaps from those original raw modal
   bounding boxes and skip amodal completion completely when none exist.
4. Complete every unique raw mask involved in a potential cross-class overlap
   once, before any same-class grouping.
5. Validate every raw completion result independently. Filter disconnected
   amodal components that do not touch the source modal mask, log each invalid
   case, and use the modal mask as the safe fallback for invalid results.
6. Recheck each candidate cross-class pair after validation. Keep the pair only
   when its two validated amodal masks have positive pixel overlap; otherwise
   log the reason and skip depth ordering and reconstruction for that pair.
7. Calculate each raw object's completion hole and effective hole area before
   grouping, then compare those individual areas independently for every
   retained cross-class pair.
8. Treat the raw object with the larger meaningful hole as occluded and the
   other as its occluder; keep ties or near-ties ambiguous.
9. Build one binary reconstruction mask per occluded raw object from its
   completion hole and the spatially relevant parts of all assigned occluders.
10. Reconstruct each selected raw object's hidden RGB once with the dedicated
    object-reconstruction model, before same-class grouping.
11. Group same-class raw objects only after reconstruction. Group membership
    is based purely on positive-area overlap between their original modal
    bounding boxes; no modal- or amodal-mask pixel-overlap test participates.
    Union-find preserves transitive bbox groups.
12. Build each final grouped object by unioning member modal masks and aligned
    validated amodal masks, while retaining member provenance and reconstruction
    results.
13. Compose one authoritative RGB source for each final group. Original visible
    pixels from every member always override reconstructed RGB. If two
    reconstructed regions overlap where no member has visible modal pixels,
    use stable raw-object order and keep the first reconstruction only; do not
    blend or overwrite it with later reconstructions.
14. Run BiRefNet only on the final grouped objects. Ordinary and fallback
    groups use original RGB with modal support; groups containing successful
    reconstruction use the composed grouped RGB source with aligned amodal
    support.
15. Recover clean foreground colors and extract RGBA layers only after
    same-class grouping is complete. Temporary background inpainting therefore
    runs once per final draggable group rather than once per raw component.
16. Preserve the existing original-image final-background behavior with one
    background-inpainting pass.

The binary reconstruction mask in step 9 remains:

```text
completion_hole OR spatially_relevant_occluder_pixels
```

It must be built only from retained amodal-overlapping cross-class pairs. The
grouping in step 11 is a downstream layer-packaging optimization and must not
change any individual completion, depth-order, or reconstruction decision.

The workflow must not use an explicit occlusion signal, class heuristic, or
global depth ordering.

## 2. Current Verified State

- Steps 19 through 25 are implemented and were accepted as completed roadmap
  checkpoints.
- Step 26 is implemented in the current working tree and awaits user
  verification.
- The model architecture now has separate `object_reconstruction` and
  `background_inpainting` categories. LaMa and SDXL are background-only
  adapters; neither can silently serve as hidden-object reconstruction.
- No concrete object-reconstruction adapter has been selected or integrated.
  Until that future checkpoint is implemented, raw objects that require hidden
  RGB reconstruction must take a safe modal/original-RGB fallback.
- The segmentation stage now preserves raw instances, and cross-class overlap
  detection/completion runs on their immutable original modal bounding boxes
  before any same-class grouping.
- Same-class merging now has a two-stage bounding-box-plus-mask overlap check.
  This reflects the current code only. The target workflow removes the
  pixel-level condition and groups same-class raw objects purely from their
  original modal bounding-box overlaps.
- Completion validation now filters disconnected predicted components that do
  not overlap the corresponding source modal mask and logs invalid-output
  reasons before applying modal fallback.
- Potential bbox pairs are now filtered by positive overlap between both
  validated amodal masks before depth ordering or reconstruction-mask logic.
- Completion and pair reasoning are no longer gated by object-reconstruction
  model availability; only the actual hidden-RGB model call is conditional.
- BiRefNet still processes the original image and modal masks for every object.
- Matting and per-layer background estimation still run on full-image inputs;
  expanded square per-object ROIs have not been implemented yet.
- The image-processing implementation has been split into explicit
  `backend/pipeline/` stages. `backend/image_processor.py` is now only the
  small public compatibility facade.
- The configured server checkpoint path is intentionally preserved; the stale
  local test expectation must not drive a production-path change.
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

The current transitional implementation still calculates bounding boxes after
same-class mask grouping. The revised target flow calculates and preserves one
original modal bounding box per raw mask, performs all occlusion reasoning and
reconstruction on raw objects, and groups them only afterward.

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

`assign_pair_roles()` compares noise-filtered effective hole areas independently
for every overlap edge. Raw completion-hole areas remain stored separately for
diagnostics:

- Holes below the configured absolute or modal-area-relative floor receive an
  effective area of zero.
- A larger first effective area outside the configured tolerance makes the
  first object occluded.
- A larger second effective area outside the configured tolerance makes the
  second object occluded.
- Equal or near-equal effective areas are ambiguous and receive no roles.

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

### 3.13 Two-stage overlap validation during grouping

The new grouping helper uses a two-stage overlap test instead of grouping from
bounding boxes alone:

1. The supplied bounding boxes must have positive-area intersection.
2. The supplied masks must share at least one positive pixel.

Only pairs passing both checks are joined. Union-find still makes the grouping
transitive. This documents the newly committed current implementation, not the
final target behavior. Under the revised workflow, the pixel-level condition
must be removed from same-class grouping: original modal bounding-box overlap
alone forms the final draggable groups after individual reconstruction.

Pixel overlap remains relevant in a different place: after completion
validation, each potential cross-class occlusion pair must be retained only if
its two validated amodal masks actually overlap. That pair-level gate is not
currently implemented by `build_reconstruction_masks()` and must be added
explicitly before depth ordering.

### 3.14 Disconnected-component filtering and validation logging

Completion validation now splits each predicted amodal mask into connected
components and discards components that do not overlap the source modal mask.
It rebuilds the candidate from the remaining connected components before the
existing modal-coverage, area-growth, and bounding-box-growth checks.

The completion stage logs the reason for every rejected output, including
invalid type or dtype, shape mismatch, non-finite values, empty masks, missing
modal coverage, excessive area growth, excessive bounding-box growth, and the
absence of any connected component touching the modal mask. Batch inference
and output-count failures retain their safe modal fallback behavior.

### 3.15 Square-ROI reconstruction mechanics

The Step 19 reconstruction infrastructure is present:

- Reusable ROI utilities derive a support box, expand context, create a square
  crop, pad at image boundaries, crop aligned arrays/images, and restore
  coordinates.
- `DetectedObject` can retain a square `reconstruction_canvas` and its aligned
  `reconstruction_roi` without allocating one full-image RGB copy per object.
- `reconstruct_objects()` selects only non-empty reconstruction masks, creates
  a binary `0/255` PIL mask, invokes the injected reconstruction callable, and
  normalizes the returned RGB crop.

These mechanics are reusable in the revised raw-object workflow, but the
current downstream matting/layer stages do not yet consume the reconstructed
canvas.

### 3.16 Completion fallback, noise floors, and tie tolerance

Steps 20 and 21 added deterministic safeguards that remain applicable to each
raw object:

- Batch output-count, type, shape, finite-value, non-empty, modal-preservation,
  area-growth, and bbox-growth validation.
- Per-object modal fallback for invalid outputs and batch-wide fallback for
  shared inference failures.
- Absolute/relative minimum meaningful hole area and configurable pairwise tie
  tolerance.
- Separate raw and effective completion-hole areas for diagnostics.

### 3.17 Separate reconstruction and background-inpainting roles

Step 22 is implemented in the current working tree:

- `BaseObjectReconstructionModel` and `BaseBackgroundInpaintingModel` expose
  distinct callable contracts.
- The registry and manager contain separate `object_reconstruction` and
  `background_inpainting` categories/APIs.
- LaMa and SDXL live under `background_inpainting`; LaMa remains the default.
- `object_reconstruction.active` is empty until a concrete future model is
  selected.
- The orchestrator injects background inpainting only into layer/background
  stages and never uses LaMa/SDXL as a reconstruction fallback.
- Notebook examples and deterministic architecture tests were updated to prove
  role isolation.

This section records the Step 22 behavior verified by the user.

### 3.18 Current behaviors intentionally scheduled for replacement

The following implemented behaviors are transitional rather than target
requirements:

- The old same-class helper still implements bbox-plus-modal-pixel overlap, but
  Step 23 no longer invokes it during segmentation; final bbox-only grouping is
  not implemented yet.
- Candidate cross-class pairs are not yet filtered by validated-amodal overlap.
- BiRefNet and layer extraction always use the original full image and modal
  masks, even when a reconstruction canvas exists.

The remaining steps below replace these behaviors incrementally; they must not
be mistaken for finished parts of the revised workflow.

### 3.19 Immutable raw SAM-object extraction

Step 23 now establishes the revised pipeline boundary:

- `extract_raw_objects()` returns one `DetectedObject` per non-empty SAM3 raw
  mask without same-class grouping.
- Overlapping same-class raw masks remain separate records.
- Each record stores stable global `segmentation_index`/`object_id` ordering and
  an `original_modal_bbox` snapshot calculated from its own normalized modal
  mask.
- `original_modal_bbox` remains available independently if a later stage
  changes the working `bbox`.
- Production orchestration calls the explicit raw-object extractor, while a
  temporary `extract_objects` alias keeps the partial notebook compatible until
  its scheduled parity step.
- Focused tests prove raw ordering, labels, mask normalization, individual
  bboxes, absence of grouping, and orchestration-boundary use.

During Step 23 verification, the recently added completion logger was also
corrected to use the existing `display_label` contract rather than the
nonexistent `label` attribute. Its existing validation tests cover the fix.

### 3.20 Raw cross-class candidate completion

Step 24 establishes completion before reconstruction-model selection:

- `link_overlap_partners()` detects positive-area cross-class overlaps from
  each raw object's immutable `original_modal_bbox` and stores partner IDs in
  deterministic raw-object order.
- `get_completion_candidates()` continues to batch every participating raw
  object once, even when it belongs to several potential pairs.
- Both `process_masks()` and `process_image()` now link and complete candidates
  regardless of whether an object-reconstruction model is configured.
- Empty candidate sets still bypass completion-model resolution, so DIFT and
  SDAmodal are not loaded when no cross-class bbox overlap exists.
- Existing completion validation, disconnected-component filtering, and
  per-object modal fallback remain the completion boundary.
- Only the later hidden-RGB reconstruction call checks
  `has_object_reconstruction_model()`; LaMa and SDXL remain background-only.

### 3.21 Validated-amodal pair filtering

Step 25 adds the explicit pair gate before depth ordering:

- `filter_pairs_by_amodal_overlap()` preserves potential bbox-pair order and
  returns a separate retained-pair list only when both validated/fallback
  amodal masks share at least one positive pixel.
- Rejected pairs log both object IDs and explicitly skip depth ordering and
  reconstruction for that relationship.
- Potential `overlap_partner_ids` and each object's validated completion result
  remain unchanged, including when an object participates in both retained and
  rejected pairs.
- Modal-fallback pairs with no pixel intersection and disconnected completion
  artifacts are rejected safely.
- `process_image()` passes only retained pairs into effective-area comparison,
  role assignment, occluder aggregation, and reconstruction-mask creation.

### 3.22 Individual raw-object depth and reconstruction masks

Step 26 consolidates the individual-object depth stage behind
`prepare_raw_reconstruction_masks()`:

- Raw completion-hole areas remain unchanged while noise-filtered effective
  areas are stored independently on each raw object.
- Only Step 25 retained pairs enter pairwise depth decisions, preserving the
  configured noise floor, tie tolerance, ambiguous ties, and independent
  mixed roles without constructing a global depth order.
- Decisive relationships aggregate unique occluder IDs on the occluded raw
  object.
- One pass builds at most one hard boolean reconstruction mask per raw object
  from its completion hole and the spatially relevant modal pixels of all
  assigned occluders.
- The orchestrator now calls this explicit raw-object boundary instead of
  assembling effective areas, decisions, and masks piecemeal.

## 4. Important Gaps in the Current Implementation

The following requirements are not complete and must not be treated as
finished:

### 4.1 Raw-object extraction and identity gaps

- Raw extraction now preserves segmentation order and immutable original modal
  bboxes, but raw and final grouped objects still share one transitional
  `DetectedObject` type; final-group provenance fields do not exist yet.
- The final same-class grouping entry point is intentionally absent until Step
  29; only the unused transitional helper remains.
- The public/process-mask path has not been redefined for final grouped-object
  output under the revised ordering.

### 4.2 Completion and cross-class pair gaps

- Potential and retained pairs are separate transient lists, but they are not
  yet exposed in a dedicated diagnostics/result record.

### 4.3 Individual depth and reconstruction gaps

- A concrete object-reconstruction model adapter has not been selected or
  added. LaMa and SDXL must remain background-only.
- Reconstruction-output validation and per-object modal/original-RGB fallback
  are incomplete.

### 4.4 Post-reconstruction grouping and compositing gaps

- Same-class grouping does not yet run after individual reconstruction.
- The current helper requires pixel-level modal-mask overlap; the revised final
  grouping must use only positive-area overlap of original modal bboxes.
- Final grouped records do not retain ordered raw member IDs/provenance.
- There is no aligned group-level union of member modal/amodal masks and no
  authoritative composed group RGB source.
- Original modal pixels are not yet protected from reconstructed output during
  group composition.
- Reconstruction-to-reconstruction overlap does not yet use deterministic
  stable-order, first-reconstruction-wins behavior.

### 4.5 Downstream final-group gaps

- BiRefNet still runs on original full-image RGB and modal masks instead of
  cropped final-group sources/support.
- `reconstruction_canvas` or the composed group source is not consumed by
  matting and foreground-color recovery.
- Layer extraction still accepts synchronized masks/labels and runs temporary
  background inpainting per current object rather than once per final group.
- Inference ROIs are not yet separated from tight final refined-alpha export
  bounds.
- Final-background visible-coverage semantics have not been reverified after
  the raw-to-group refactor.

### 4.6 Verification and operational gaps

- Raw count, candidate/retained pair counts, final group membership, conflict
  winners, stage timings, and fallback reasons are not fully observable.
- The example notebook still needs the final raw-to-group execution order.
- Deterministic end-to-end tests do not yet cover the revised order and conflict
  rules.
- Real SDAmodal/object-reconstruction acceptance scenes, latency, and GPU-memory
  measurements remain outstanding.

Also note that the `input_size` and `enlarge_box` values under
`backend/config.yaml` must be reconciled with the values read from
`config_SDAmodal.yaml`; there should be one authoritative runtime source.

## 5. Remaining Implementation Steps

Each step below should remain a separate review checkpoint: present the small
plan, obtain approval, implement with tests, and wait for verification before
continuing.

Steps 19-25 are completed historical checkpoints. Step 26 is implemented and
tested in the current working tree but still requires the user's checkpoint
verification; Step 27 must not begin until that verification is received.

### Step 19: Reconstruct hidden RGB — Completed

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

### Step 20: Harden completion output validation and fallback — Completed

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

### Step 21: Add noise-floor and tie-tolerance configuration — Completed

- Add an absolute and/or modal-area-relative minimum meaningful hole area.
- Add a configurable tie tolerance.
- Compare effective areas rather than raw boundary-growth noise.
- Mark equal or near-equal pairs ambiguous.
- Keep raw areas available for diagnostics.

### Step 22: Separate object reconstruction from background inpainting — Completed

- Replace the single generic inpainting dependency with two explicit model
  categories and two explicit manager APIs:
  - `object_reconstruction`: reconstructs hidden RGB belonging to an occluded
    object.
  - `background_inpainting`: removes objects and fills only background RGB.
- Give each category its own active-model selection and model-specific
  configuration. Do not share prompts, mask expansion, blending, or generation
  settings across the two categories.
- Move LaMa and SDXL under `background_inpainting`. LaMa remains the default
  background backend; SDXL is an alternative background backend, not an
  object-reconstruction model.
- Reserve `object_reconstruction` configuration for a future model, with no
  active concrete adapter until that model is added. Its future configuration
  may include object-aware prompt/context and reconstruction-specific mask or
  blending policies.
- Define separate callable contracts and dependency-injection boundaries so
  reconstruction code cannot accidentally receive a background inpainter, and
  background code cannot receive the object-reconstruction model.
- Use `background_inpainting` for both the temporary per-final-group background
  estimate used during foreground-color recovery and the single final
  background-removal pass.
- When no object-reconstruction model is configured, skip hidden-RGB
  reconstruction safely and retain the object's modal fallback path. Never
  silently fall back to LaMa or SDXL for object reconstruction.
- Update configuration, registry/manager wiring, orchestrator wiring, notebook
  examples, and deterministic tests to prove the two model paths stay isolated.

### Step 23: Extract immutable raw SAM objects — Completed

- Split `pipeline/segmentation.py` into raw mask extraction/normalization and a
  separate grouping entry point; do not call `_merge_overlapping_masks()` while
  creating raw objects.
- Add/evolve a typed raw-object record with stable segmentation order,
  `object_id`, semantic class, display label, original full-image modal mask,
  and original tight modal bbox.
- Calculate the bbox immediately from each normalized raw mask and preserve it
  unchanged through completion, depth ordering, reconstruction, and grouping.
- Keep empty-mask handling and deterministic labels/order.
- Update `process_masks()`/orchestrator boundaries so later stages receive raw
  records without accidentally grouping them.
- Add focused tests proving one record per SAM3 raw mask, immutable original
  bboxes, class/order preservation, and no grouping call during extraction.

Stop after Step 23 and obtain user verification before continuing.

### Step 24: Complete raw cross-class bbox candidates — Completed

- Run cross-class positive-area bbox-overlap detection on the Step 23 raw
  records. Same-class pairs remain excluded and edge/corner contact remains a
  non-overlap.
- Store potential partner IDs on raw records and select each participating raw
  object once even when it belongs to multiple pairs.
- Run SDAmodal completion and the complete existing validation/fallback workflow
  on each unique candidate before any grouping.
- Preserve the no-cross-class-overlap bypass so DIFT/SDAmodal are not loaded or
  called when the candidate list is empty.
- Remove the current coupling that skips the entire completion branch when no
  object-reconstruction model is configured. Under the revised workflow,
  completion/pair reasoning runs first; only the later RGB reconstruction call
  depends on that model's availability.
- Add tests for raw bbox pairs, unique candidate batching, no-overlap bypass,
  per-object validation/fallback, and completion execution without a configured
  object-reconstruction model.

Stop after Step 24 and obtain user verification before continuing.

### Step 25: Filter pairs by validated amodal overlap — Completed

- After both members have validated/fallback amodal masks, calculate:

  ```text
  amodal_pair_overlap = validated_amodal_A AND validated_amodal_B
  ```

- Retain a potential cross-class pair only when this intersection has at least
  one positive pixel. Log object IDs and a non-image skip reason for rejected
  pairs.
- Keep potential bbox pairs separate from retained amodal-overlap pairs for
  diagnostics and do not delete per-object completion results when only one
  pair is rejected.
- Ensure rejected pairs cannot assign depth roles, occluders, reconstruction
  masks, or trigger reconstruction.
- Do not treat the current expanded-amodal-support intersection inside
  `build_reconstruction_masks()` as this validation; it is a later spatial
  constraint, not a two-amodal-mask pair test.
- Add tests for positive overlap, no overlap, modal-fallback pairs, one object
  in mixed retained/rejected pairs, and disconnected-component artifacts that
  must not preserve a false pair.

Stop after Step 25 and obtain user verification before continuing.

### Step 26: Decide individual depth and build raw reconstruction masks — Implemented, pending user verification

- Calculate completion holes and raw/effective hole areas independently on
  every raw object, never on grouped unions.
- Run pairwise role decisions only on Step 25 retained pairs, preserving the
  existing noise floors, tie tolerance, ambiguity behavior, and absence of a
  global depth order.
- Apply decisions to raw objects and aggregate all unique assigned occluders.
- Build one hard boolean reconstruction mask per occluded raw object using its
  completion hole plus spatially relevant modal pixels from all assigned
  occluders.
- Guarantee that one raw object involved in multiple retained pairs can trigger
  at most one later reconstruction call.
- Add tests for either member becoming occluded, exact/near ties, mixed pair
  roles, multiple occluders, overlap chains, and rejected-pair isolation.

Stop after Step 26 and obtain user verification before continuing.

### Step 27: Integrate the future object-reconstruction model

- Add a concrete adapter only after the object-reconstruction model has been
  selected and its inference requirements are known.
- Implement the dedicated reconstruction contract using the original RGB crop,
  aligned binary reconstruction mask, semantic/object context where supported,
  and reconstruction-specific configuration.
- Lazy-load this model only when at least one object has a non-empty
  reconstruction mask.
- Keep the model output crop aligned with the existing square reconstruction
  ROI and store it without creating a full-image copy per object.
- Do not change or reuse background-inpainting configuration while integrating
  this model.

Step 27 is intentionally blocked until the user selects/provides the concrete
model and its inference contract. Do not substitute LaMa or SDXL. Stop and
request those model details when this checkpoint is reached.

### Step 28: Validate individual reconstruction and provide modal fallback

- Validate returned image type, ROI dimensions, and RGB conversion.
- Ensure pixels outside the permitted hard-mask/blend region remain unchanged
  inside the crop.
- Confirm the completion-hole region contains usable RGB data.
- On failure, clear the reconstruction crop/result and mark the object for its
  ordinary original-image/modal-mask path.
- Ensure one object's failure does not discard other successful objects.

- Add tests for invalid types/sizes, changes outside the permitted region,
  unusable completion-hole RGB, independent per-object failure, and the missing-
  model modal fallback.

Stop after Step 28 and obtain user verification before continuing.

### Step 29: Group raw objects by original same-class bbox overlap

- Run grouping only after all raw-object reconstruction attempts and fallbacks
  are finalized.
- Group only objects with the same semantic class and positive-area overlap
  between their immutable original modal bboxes. Do not check modal, amodal, or
  reconstructed pixel overlap.
- Preserve transitive union-find behavior and stable raw segmentation/object-ID
  order within every group.
- Introduce a final grouped-object contract containing ordered member IDs and
  enough provenance to map every member mask, reconstruction ROI, and fallback
  state.
- Build aligned geometry:

  ```text
  grouped_modal_mask = union(member original modal masks)
  grouped_amodal_mask = union(member validated amodal masks)
  grouped_bbox = bbox(grouped_modal_mask)
  ```

- Members without completion/reconstruction contribute their safe modal state
  without removing successful sibling data.
- Add tests for bbox-only grouping despite pixel-disjoint masks, non-overlapping
  bboxes remaining separate, same-class-only behavior, transitive groups,
  deterministic member order, aligned unions, and mixed fallback members.

Stop after Step 29 and obtain user verification before continuing.

### Step 30: Compose final-group RGB with deterministic conflict priority

- Create one authoritative RGB source per Step 29 group in full-image
  coordinates or one aligned group ROI, initialized from the original image.
- Map every successful member `reconstruction_canvas` and
  `reconstruction_roi` into the group coordinate system.
- Protect `grouped_modal_mask` completely so reconstructed RGB never overwrites
  any member's original visible pixels.
- Paste reconstruction only within that member's permitted reconstruction
  region and outside protected original modal coverage.
- Process members in stable group order and maintain
  `already_filled_reconstruction_region`. Where reconstructed regions overlap
  outside all modal coverage, the first accepted reconstruction wins; later
  results neither overwrite nor blend those pixels.
- Retain untouched original-image pixels everywhere no accepted reconstruction
  is permitted.
- Emit only final groups with their authoritative composed source to downstream
  matting/layer stages; raw members remain diagnostics, not draggable layers.
- Add tests for ROI mapping, original-pixel priority, first-reconstruction-wins,
  no blending, untouched-source preservation, reconstruction/fallback mixtures,
  and deterministic output independent of dictionary/set iteration.

This post-reconstruction grouping reduces BiRefNet, temporary background-
inpainting, foreground-recovery, and exported-layer counts. It does not reduce
individual amodal-completion or object-reconstruction calls because those must
precede grouping for correct per-object depth reasoning.

Stop after Step 30 and obtain user verification before continuing.

### Step 31: Refactor BiRefNet to final grouped-object inputs

- Replace the shared full-image/raw-mask-list matting interface with final
  grouped-object cropped inputs. Never run this stage on raw members after
  grouping.
- Ordinary or all-fallback groups use an expanded square crop from the original
  image and grouped modal support.
- Groups containing accepted reconstruction use their composed group RGB
  source and aligned grouped amodal support.
- Treat the composed group source as authoritative. Do not pass the full
  original `image_np` to BiRefNet for a reconstructed group.
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
- Prove final groups enter BiRefNet only after all member reconstruction and
  RGB-conflict resolution are complete.

Stop after Step 31 and obtain user verification before continuing.

### Step 32: Refactor final grouped-object layer extraction

- Consume final grouped-object records instead of synchronized masks/labels or
  raw member records.
- Select one authoritative `source_rgb` for each object and use exactly the
  same source for matting and foreground-color recovery:
  - all-fallback groups use the aligned crop from the original image;
  - groups containing accepted reconstruction use their already conflict-
    resolved composed group RGB source.
- Remove the current unconditional dependency on the full original
  `image_np` inside layer extraction. A reconstructed group's foreground RGB
  must come from its composed group source, not from original pixels that still
  contain an occluder.
- Build the temporary background-estimation inpainting mask as a hard binary
  mask from the selected modal/amodal support, optionally unioned with
  thresholded alpha coverage. Do not pass fractional `soft_alpha` directly to
  inpainting.
- Run this temporary background inpainting with the selected
  `background_inpainting` backend once per final group ROI instead of
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

Stop after Step 32 and obtain user verification before continuing.

### Step 33: Preserve final-background semantics

- Keep final background inpainting based on the original source image and the
  selected `background_inpainting` backend.
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

Stop after Step 33 and obtain user verification before continuing.

### Step 34: Add diagnostics, notebook parity, and observability

- Record raw-object count, final grouped-object count, potential bbox-overlap
  edge count, retained amodal-overlap edge count, and unique completion-
  candidate count.
- Record modal, amodal, raw-hole, and effective-hole areas.
- Record pair decisions and ambiguity reasons.
- Record same-class group membership and reconstruction conflict winners by
  object ID without logging pixel data.
- Record completion, object-reconstruction, group composition,
  background-inpainting, and matting timings separately.
- Record fallback stage and reason without logging image data or features.
- Optionally record peak GPU memory in diagnostic mode.
- Update the partial-pipeline notebook to expose raw objects, potential bbox
  pairs, retained amodal pairs, individual reconstruction, final groups, RGB
  conflict resolution, and downstream final-group processing in production
  order.

Stop after Step 34 and obtain user verification before continuing.

### Step 35: Add end-to-end deterministic tests

Cover at minimum:

- No-overlap bypass.
- Raw-mask bounding boxes and cross-class overlap detection before grouping.
- Completion and validation on individual raw objects.
- Potential cross-class pairs being rejected when their two validated amodal
  masks do not have positive pixel overlap.
- Disconnected amodal components being filtered before the cross-class
  amodal-overlap check so they cannot preserve a false occlusion pair.
- Completion holes, effective areas, depth roles, occluders, reconstruction
  masks, and reconstruction calls remaining individual-object operations.
- Same-class grouping occurring only after individual reconstruction.
- Same-class original modal-bbox overlap merging even when modal and amodal
  masks have no pixel overlap.
- Same-class masks with non-overlapping original modal boxes remaining separate
  even if their later reconstructed regions overlap.
- Transitive bbox-only grouping and aligned modal/amodal unions.
- Original modal pixels overriding every reconstructed result during group RGB
  composition.
- Stable first-reconstruction-wins behavior when reconstructed regions overlap
  outside all member modal masks, with no blending or later overwrite.
- Only final grouped objects entering matting, temporary background inpainting,
  foreground recovery, and layer export.
- Either member of a two-object pair becoming occluded.
- Exact and near ties.
- Same-class overlap bypass.
- One object with multiple occluders.
- Three-object overlap chains.
- Invalid completion output.
- Completion-batch failure.
- Reconstruction failure.
- Missing object-reconstruction model producing a safe modal fallback without
  invoking LaMa or SDXL.
- Strict isolation between object-reconstruction and background-inpainting
  model calls and configuration.
- Ordinary and reconstructed matting paths.
- Reconstructed final-group matting and foreground recovery reading RGB from
  the conflict-resolved composed group source instead of the original image.
- Expanded square ROI creation, border padding, and full-image coordinate
  restoration.
- Amodal versus modal layer bounds.
- Hard-mask enforcement at both reconstruction and background-estimation
  inpainting boundaries.
- Final refined-alpha export bounds without clipping newly recovered edges.
- Final background isolation from per-object reconstruction results.

Stop after Step 35 and obtain user verification before continuing.

### Step 36: Run real-model acceptance and performance evaluation

- Use a fixed set of representative occlusion scenes.
- Inspect modal masks, amodal masks, holes, role decisions, reconstruction
  masks, reconstructed canvases, alpha mattes, layers, and final backgrounds.
- Tune only documented thresholds and expansion/blending settings.
- Measure completion latency, object-reconstruction latency,
  background-inpainting latency, total request latency, and peak GPU memory.
- Consider mixed precision or offloading only after output-equivalence checks.

## 6. Final Acceptance Criteria

The integration is complete only when all of the following are true:

- Positive-area cross-class overlap between original raw modal bounding boxes
  is the only completion trigger.
- Completion, validation, depth ordering, reconstruction-mask construction,
  and hidden-RGB reconstruction happen on individual eligible raw objects
  before same-class grouping.
- Raw completion eligibility is determined only from original modal geometry;
  model-generated amodal growth cannot create a completion candidate.
- Both members of every raw cross-class overlap pair are completed once per
  request.
- No-overlap requests do not initialize or run DIFT/SDAmodal.
- After validation, a potential cross-class pair reaches depth ordering and
  reconstruction only when its two validated amodal masks have positive pixel
  overlap.
- Completion holes and depth roles are calculated from individual raw masks,
  never from grouped unions.
- Same-class group membership is evaluated only after individual reconstruction
  and requires only positive-area overlap between original modal bounding
  boxes. No modal-, amodal-, or reconstructed-mask pixel overlap is required.
- Grouped modal masks union original members, while grouped amodal masks union
  those same members' validated completion results.
- Disconnected amodal components that do not touch their source modal mask are
  filtered before cross-class pair validation, and invalid completion cases are
  logged with safe modal fallback.
- Occluded/occluder roles depend only on validated completion-hole areas.
- Ambiguous pairs do not trigger destructive reconstruction.
- Multiple assigned occluders produce one constrained reconstruction pass per
  occluded object.
- Hidden-object RGB reconstruction uses only the configured
  `object_reconstruction` model. LaMa and SDXL are never used for this purpose.
- LaMa and SDXL are selectable only within `background_inpainting`; that model
  category owns temporary background estimation and final-background removal.
- Missing or failed object reconstruction produces a safe modal fallback and
  never crosses over to a background-inpainting backend.
- Successful hidden-object reconstruction and same-class RGB composition both
  happen before BiRefNet.
- Every final grouped RGB source protects all original member modal pixels from
  reconstructed output.
- Reconstruction-to-reconstruction conflicts outside visible modal coverage
  use stable raw-object order: the first accepted reconstruction wins, later
  results do not overwrite it, and the predictions are not blended.
- Every inpainting adapter receives a binary hard mask; `soft_alpha` is reserved
  for alpha refinement, foreground recovery, and RGBA compositing.
- Matting uses expanded square per-final-group ROIs and restores alpha to
  full-image coordinates.
- Ordinary and all-fallback final groups use original RGB with grouped modal
  support.
- Final groups containing successful reconstruction use their conflict-resolved
  composed RGB source for both BiRefNet and foreground-color recovery, with
  grouped amodal support.
- Matting, temporary background inpainting, foreground recovery, and layer
  export execute only on final grouped objects, producing one draggable layer
  per group rather than one layer per raw component.
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
115 passed with the 1 known stale checkpoint-path expectation deselected
```

The remaining failure reflects a stale local expected checkpoint path. The
runtime path is intentionally configured to match the server and should remain
unchanged.
