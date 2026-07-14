# Amodal Completion and Occlusion-Reconstruction Integration Plan

## 1. Purpose and Scope

This document replaces `first_plan.md` as the current architectural plan for integrating SDAmodal amodal completion into the existing image-to-components pipeline.

The target workflow is:

1. Segment objects with SAM3 and group detections by semantic class.
2. Build a merged bounding box for each grouped object.
3. Compare bounding boxes belonging to different classes.
4. Treat any cross-class bounding-box overlap as the trigger for possible occlusion.
5. Run amodal completion on both members of every overlapping pair.
6. Measure how much new mask area each completion added.
7. Select the object with the larger completed area as the occluded object.
8. Remove the other object from the source image and reconstruct the selected object's hidden appearance with the inpainting model.
9. Run BiRefNet only after reconstruction, using the reconstructed object image and completed mask.
10. Produce the completed RGBA object layer and the final object-free background.

This is an architectural and implementation-sequencing plan only. It contains no Python implementation. The DIFT extractor at `backend/models/completion/dift/extract_dift_amodal.py` and the compatible pretrained checkpoints are treated as present, correctly loaded, and ready for inference.

## 2. Confirmed Design Decisions

The following decisions are fixed for this integration:

- No explicit occlusion detector will be added.
- No depth-order model or order matrix will be used.
- Cross-class bounding-box overlap is the only amodal-completion trigger.
- Both objects in an overlapping pair must be completed before deciding which is occluded.
- The newly completed area, not the total completed-mask area, determines which object is treated as occluded.
- The object with the larger valid completed area is the occluded object.
- The other member of that pair is treated as the occluding object for reconstruction-mask construction.
- BiRefNet must run after hidden RGB reconstruction for an occluded object.
- Objects that do not participate in any cross-class bounding-box overlap bypass completion and continue through the ordinary matting path.
- Same-class overlap does not trigger this workflow.
- Modal and amodal masks remain separate throughout the pipeline.

## 3. Current Pipeline and Target Pipeline

### 3.1 Current pipeline

The current orchestrator in `backend/image_processor.py` performs:

```text
Input image and keywords
  -> SAM3 segmentation
  -> same-keyword mask merging
  -> BiRefNet matting on each visible mask
  -> per-object color/alpha refinement
  -> final background inpainting
```

### 3.2 Target pipeline

The target pipeline is:

```text
Input image and keywords
  -> SAM3 segmentation
  -> semantic grouping and merged object records
  -> bounding-box calculation
  -> cross-class overlap graph
       -> no overlap: ordinary object path
       -> overlap: shared DIFT extraction
                    -> SDAmodal completion for both/all involved objects
                    -> completion validation
                    -> hole-area comparison per overlap pair
                    -> occluded/occluder role assignment
                    -> occluder-removal mask construction
                    -> hidden RGB reconstruction by inpainting
  -> BiRefNet matting
       -> original image for ordinary objects
       -> reconstructed object canvas for occluded objects
  -> completed RGBA object layers
  -> final background inpainting from original modal-object coverage
```

The key injection point is after `_extract_raw_masks()` and before `_refine_masks()`. The current direct transition from raw masks to matting must be replaced by an analysis-and-reconstruction stage that chooses the correct image and mask inputs for each object.

## 4. Core Data Model

Replace parallel mask and label lists inside the orchestrator with an internal object record. The record is not initially part of the public API.

Each object record should carry:

| Field | Purpose |
| --- | --- |
| Stable object ID | Identifies the object independently of list position. |
| Semantic class | The normalized keyword used for cross-class comparisons. |
| Display label | Preserves the existing output label, including instance suffixes. |
| Modal mask | Original visible SAM3 mask at full image resolution. |
| Modal bounding box | Tight box around the modal mask. |
| Amodal mask | Validated completion result; defaults to the modal mask. |
| Completion hole | Pixels in the amodal mask but not in the modal mask. |
| Completion-hole area | Count of valid newly completed pixels. |
| Completion status | Not required, completed-valid, completed-fallback, or failed. |
| Overlap partners | Stable IDs of cross-class objects whose boxes overlap. |
| Pair decisions | Per-partner area comparison and assigned role. |
| Reconstruction mask | Region supplied to the inpainting model for hidden-content recovery. |
| Reconstruction canvas | Image used as the BiRefNet input for this object. |
| Matting support mask | Modal mask for ordinary objects; amodal mask for reconstructed objects. |
| Soft alpha | BiRefNet result constrained by the appropriate support mask. |
| Diagnostics | Areas, ratios, timing, validation results, and fallback reason. |

The record must preserve full-image coordinates until final layer cropping. This prevents mismatches between DIFT crops, completion output, inpainting output, matting output, and returned layer offsets.

## 5. Semantic Grouping and Bounding-Box Overlap

### 5.1 Preserve semantic class separately from display labels

The current pipeline turns multiple detections into labels such as `person_0` and `person_1`. Overlap filtering must compare their underlying semantic class (`person`), not the suffixed display label. The extraction stage should therefore return or construct both values explicitly.

### 5.2 Grouping boundary

Retain the existing same-keyword merging behavior as the initial semantic grouping rule. Each merged output becomes one logical object record and receives one merged modal bounding box.

Before implementation, verify that `_merge_overlapping_masks()` returns masks in a stable order and that it merges only detections within the keyword currently being processed. Do not merge objects across different keywords.

### 5.3 Bounding-box convention

Use one canonical box representation throughout the new orchestration layer. The existing project convention is `[x, y, width, height]`; retain it at the model-adapter boundary. Pairwise overlap calculations should derive right and bottom edges consistently from this representation.

Two boxes overlap only when their intersection has positive width and positive height. Edge or corner contact alone should not trigger completion unless a later requirement explicitly changes this rule.

### 5.4 Pair generation

After all object records exist:

1. Iterate over each unordered object pair once.
2. Skip pairs whose normalized semantic classes are equal.
3. Calculate the intersection of their modal bounding boxes.
4. Skip pairs with zero intersection area.
5. Record each positive-overlap pair as an edge in an overlap graph.
6. Add each object ID to the other's overlap-partner set.
7. Mark every object incident to at least one edge as requiring completion.

The graph representation is important for scenes containing three or more overlapping classes. It prevents duplicate completion calls and preserves every pairwise decision.

### 5.5 No-overlap bypass

If the graph has no edges:

- Do not initialize SDAmodal solely for this request.
- Do not initialize or run DIFT solely for this request.
- Set each amodal mask equal to its modal mask.
- Set completion holes and hole areas to zero.
- Use the original source image and modal mask for BiRefNet.
- Continue through the current layer and background paths.

## 6. Detailed Resolution of Package Import Conflicts (Previous Point 2.2)

### 6.1 Goal

Make all completion research modules importable as an isolated application package without allowing names such as `models`, `utils`, `inference`, or `src` to resolve to unrelated repository or installed packages.

### 6.2 Package-boundary inventory

Before changing imports, inventory every Python module below `backend/models/completion/` and classify each import as:

- Python standard library.
- Third-party dependency.
- Application package import.
- Completion-package-local import.
- DIFT-package-local import.
- Training-only or debugging-only import.

Pay particular attention to:

- `backend/models/completion/inference.py`
- `backend/models/completion/run_inference.py`
- `backend/models/completion/models/`
- `backend/models/completion/utils/`
- `backend/models/completion/dift/extract_dift_amodal.py`
- `backend/models/completion/dift/dift_sd.py`
- `backend/models/completion/dift/src/models/`

### 6.3 Establish explicit package roots

1. Treat `backend.models.completion` as the root package for SDAmodal.
2. Treat the DIFT implementation as a subpackage of that root rather than a standalone working-directory-dependent project.
3. Add package initializers where needed so imports do not depend on invoking a script from a particular directory.
4. Decide one canonical DIFT featurizer location. The repository currently contains both `dift/dift_sd.py` and `dift/src/models/dift_sd.py`; the adapter must reference exactly one and document why it matches `extract_dift_amodal.py` and the checkpoint expectations.
5. Keep command-line entry points separate from modules imported by the API.

### 6.4 Convert ambiguous imports

For each local import:

1. Resolve the intended file manually from the completion directory.
2. Replace top-level research names with explicit package-relative or fully qualified application-package imports.
3. Apply the same rule recursively inside the imported module; fixing only the first import layer is insufficient.
4. Remove any runtime manipulation of `sys.path` or assumptions about the current working directory.
5. Ensure imports succeed when the backend starts from the repository root and when tests import modules directly.

Examples of names requiring review include `models`, `utils`, `inference`, `src.models`, and `src.utils`. The plan intentionally does not prescribe Python syntax, but none may remain ambiguous in the production import path.

### 6.5 Separate production inference from research utilities

1. Identify the minimal SDAmodal inference dependency chain.
2. Prevent production adapter imports from loading training datasets, loss functions, distributed training utilities, visualization modules, notebooks, or CLI parsers.
3. Remove debugging-only runtime dependencies such as `pdb`, `ipdb`, and `matplotlib` from the production chain.
4. Remove `pycocotools` from the production chain if none of the selected inference operations require it.
5. Keep original research scripts available for reference, but route the application through a focused adapter module.

### 6.6 Import verification

Verify the refactor at four levels:

1. Import the completion base interface without loading model weights.
2. Import the SDAmodal adapter without changing the process working directory.
3. Instantiate the adapter through `ModelRegistry` and `ModelManager`.
4. Start the backend from the normal application entry point and confirm that `backend.models` is never mistaken for SDAmodal's internal `models` package.

Regression tests should deliberately pre-import `backend.models` before importing the completion adapter, because that reproduces the collision most likely to occur in the application.

## 7. Detailed Resolution of Device Handling (Previous Point 2.3)

### 7.1 Goal

Make SDAmodal, DIFT, all input tensors, feature tensors, intermediate tensors, and checkpoint loading follow the device selected by the existing `ConfigManager` and `ModelManager`.

### 7.2 Define device ownership

1. `ConfigManager.device` remains the application source of truth.
2. `ModelManager` passes the resolved device into the completion adapter, as it does for existing model categories.
3. The completion adapter owns the resolved device object and exposes no hard-coded accelerator choice to the orchestrator.
4. SDAmodal and DIFT use the same device by default to avoid repeated transfers.
5. Any future split-device or offload behavior must be explicit configuration, not implicit code behavior.

### 7.3 Remove hard-coded placement

Audit the complete production dependency chain for:

- Direct CUDA convenience calls.
- Literal `cuda`, `cuda:0`, or device index values.
- Tensor constructors that silently default to CPU.
- Checkpoint loads that deserialize directly onto a GPU.
- Internal helper tensors created without matching the input tensor's device.

Every discovered location must be changed conceptually to one of these ownership rules:

- Model parameters move to the adapter's selected device during load.
- Request tensors move to that device at the adapter boundary.
- Helper tensors inherit device and dtype from the tensor they accompany.
- Outputs needed by NumPy are detached and transferred to CPU only at the final conversion boundary.

### 7.4 Device-aware checkpoint lifecycle

1. Load checkpoints through a device-aware mapping so checkpoints saved on a different GPU remain portable.
2. Construct the model architecture before loading its state.
3. Validate checkpoint keys and architecture compatibility.
4. Move the fully initialized network to the selected runtime device.
5. Switch it to evaluation mode.
6. Prevent gradients during all request-time completion and DIFT inference.

The plan assumes checkpoint availability and compatibility; failures here are operational errors, not missing-prerequisite work.

### 7.5 CPU behavior and capability validation

Even if production will use CUDA, the adapter should fail clearly when a selected device is unsupported. At initialization:

1. Confirm the requested accelerator exists.
2. Confirm SDAmodal and DIFT dependencies support that device.
3. Emit a precise initialization error if the selected mode cannot run.
4. Do not silently choose a different GPU or silently fall back to CPU if that changes expected latency or memory behavior.

### 7.6 Precision policy

Start with the checkpoint's known-safe precision. Do not enable mixed precision until real-model validation confirms mask stability. If mixed precision is introduced later, configure it independently for SDAmodal and DIFT because their numerical requirements may differ.

### 7.7 GPU memory and concurrency

1. Keep completion lazy-loaded so requests with no overlapping boxes do not pay its memory cost.
2. Extract DIFT features once for a request and release request-scoped references after all relevant objects are completed.
3. Avoid retaining full feature pyramids in object records or API results.
4. Use the application's inference context and, if required by measured behavior, a shared GPU inference lock.
5. Measure peak memory with SAM3, DIFT, SDAmodal, BiRefNet, and the inpainting model participating in one request.
6. Add explicit offload configuration only if measurements show it is necessary.

### 7.8 Device verification matrix

Verify:

- Model parameters and request tensors are colocated.
- Each DIFT feature level arrives at SDAmodal on the expected device.
- No implicit CPU/GPU copies occur inside the per-object loop.
- Outputs convert to full-resolution NumPy masks only after inference completes.
- A configured non-default CUDA device is respected.
- Device mismatch errors are caught by tests with fake components before real-model testing.

## 8. Detailed In-Memory DIFT Feature Flow (Previous Point 2.4)

### 8.1 Goal

Eliminate the current request-time dependency on `feature/pth0` through `feature/pth3`. DIFT features must be extracted once from the uploaded source image and reused directly in memory for every object requiring completion.

### 8.2 Adapter boundary

The application-facing completion model should accept:

- The original RGB image.
- A collection of full-resolution modal masks for objects requiring completion.
- Their expanded completion boxes.
- Stable object IDs or preserved input ordering.

It should return one full-resolution validated amodal mask per input object, preserving identity and order. The orchestrator must not know about feature levels, Stable Diffusion timesteps, patch dimensions, or SDAmodal tensor formatting.

### 8.3 Source-image preparation

1. Use the original RGB image, before any object removal, guided masking, background inpainting, or reconstruction.
2. Convert it once to the normalization and image size expected by the pretrained DIFT extractor.
3. Record the transform between original image coordinates and DIFT feature coordinates.
4. Ensure that all four feature levels refer to the same transformed image.
5. Do not encode request identity through a filename; the in-memory image is the source of truth.

### 8.4 One extraction per request

1. Build the cross-class overlap graph first.
2. Collect the unique object IDs incident to at least one overlap edge.
3. If this set is empty, skip DIFT.
4. If it is non-empty, run the extractor exactly once on the source image.
5. Retain the four feature tensors in a request-scoped feature-pyramid object.
6. Reuse that object for every relevant completion call.
7. Release it after all required amodal masks have been produced and validated.

### 8.5 Feature-pyramid contract

The request-scoped pyramid must explicitly define:

- Required levels: 0, 1, 2, and 3.
- Tensor layout at its boundary.
- Batch dimension policy.
- Spatial size per level before instance cropping.
- Channel count validation.
- Dtype and device.
- Coordinate transform from source-image pixels to each feature map.

Reject a pyramid with a missing level, empty spatial extent, unexpected rank, non-finite values, or inconsistent batch size before starting per-instance completion.

### 8.6 Per-object feature preparation

For each unique completion object:

1. Start from its expanded square completion box, not merely its tight modal box.
2. Project that box independently into each DIFT level's coordinate system.
3. Clamp or pad the projected crop consistently when it extends beyond a feature-map boundary.
4. Reject a truly invalid or empty crop before model inference.
5. Resize the level crops to SDAmodal's expected spatial sizes: level 0 to 24 by 24, level 1 to 48 by 48, and levels 2 and 3 to 96 by 96.
6. Preserve the feature channels and expected layout.
7. Keep the prepared feature tensors on the selected inference device.
8. Pass them directly to SDAmodal without converting to files or using an image filename lookup.

### 8.7 Modal patch preparation

For the same expanded box:

1. Crop the binary modal mask with identical padding semantics.
2. Resize it to the configured SDAmodal input size using nearest-neighbor interpolation.
3. Preserve binary values.
4. Apply the category value expected by the pretrained model; the initial integration continues to use category value 1 for every object.
5. Keep enough box metadata to restore the predicted patch to full-image coordinates.

### 8.8 Full-image restoration and validation

1. Convert model logits to a binary predicted patch using the behavior verified against the supplied inference guide.
2. Resize the patch back to its expanded box.
3. Crop away any out-of-image padding.
4. Place it into a full-resolution mask.
5. Union it with the modal mask so visible pixels cannot disappear.
6. Calculate the completion hole as the validated amodal mask minus the modal mask.
7. Apply growth and association validation.
8. Fall back to the modal mask if validation fails.

### 8.9 Eliminate disk feature I/O

The production inference path must contain no request-time:

- Feature-directory construction.
- Image-basename manipulation.
- Feature serialization.
- Feature deserialization.
- Temporary feature files.

The existing disk-based behavior may remain only in standalone research or offline diagnostic scripts that are not imported by the API.

### 8.10 DIFT verification

Use fake extractor and fake SDAmodal components to prove:

- Zero extraction calls when no cross-class boxes overlap.
- Exactly one extraction call when one or many objects require completion.
- Each required object receives crops derived from the same pyramid.
- Each unique object is completed once even if it has multiple overlap partners.
- No file access occurs in the production completion path.
- Output object identity and order remain stable.

## 9. Completion Model Architecture Integration

### 9.1 Base interface

Add a completion-model abstraction alongside the existing segmentation, matting, and inpainting abstractions in `backend/models/base.py`. Its contract should express batch completion of multiple object masks from one source image, enabling the adapter to share DIFT extraction across the batch.

### 9.2 Registry

Extend `ModelRegistry` with a `completion` category and register the SDAmodal adapter under a stable name. Registry import behavior must not eagerly load weights.

### 9.3 Manager

Extend `ModelManager` with:

- A nullable completion-model slot.
- A lazy completion getter.
- Completion-category instantiation through the existing configuration path.
- Optional warm-up behavior controlled by configuration.

The ordinary no-overlap request path should not call the completion getter.

### 9.4 Configuration

Add a completion configuration section containing at least:

- Enabled state.
- Active adapter name.
- SDAmodal model configuration path.
- Checkpoint location.
- DIFT model identity and inference parameters.
- Completion input size.
- Expanded-box scale.
- Output decision method or threshold, matching verified model behavior.
- Mask-growth validation limits.
- Minimum meaningful completion-hole area.
- Hole-area tie tolerance.
- Lazy-load and warm-up behavior.
- Device/offload settings if they differ from global defaults.

Do not retain the previous `off`, `all`, or `heuristic` completion policy. Cross-class box overlap is now the fixed trigger.

## 10. Expanded Completion Boxes

For every object requiring completion:

1. Reject an empty modal mask.
2. Calculate the tight modal bounding box.
3. Expand it into the square region expected by SDAmodal, using the configured enlargement factor and the model guide's minimum enlargement behavior.
4. Preserve the unclamped box so crop-padding and full-image restoration use consistent geometry.
5. Validate positive dimensions.
6. Handle negative origins and boxes extending beyond the image through padding rather than changing the object coordinate system unpredictably.
7. Store both the tight modal box and expanded completion box for diagnostics.

The overlap trigger must use merged modal bounding boxes, not these expanded boxes. Expanded boxes exist only to provide completion context; using them for trigger detection would create false overlap edges.

## 11. Amodal Prediction Validation and Hole Measurement

### 11.1 Validation sequence

For each predicted amodal mask:

1. Confirm full-image shape.
2. Convert it to the canonical binary representation.
3. Reject non-finite or malformed output before conversion.
4. Union predicted pixels with modal pixels.
5. Confirm the result is non-empty.
6. Confirm every modal pixel remains present.
7. Confirm area growth is below a configured maximum.
8. Confirm bounding-box growth is below a configured maximum.
9. Confirm the completed region remains spatially connected or plausibly associated with the modal object according to a narrowly defined validation rule.
10. On any failure, set the amodal mask to the modal mask, hole area to zero, and status to fallback.

### 11.2 Hole definition

The completed area for an object is exactly:

```text
completion hole = validated amodal mask AND NOT modal mask
```

The hole area is the count of positive pixels in this difference. It is not:

- The full amodal area.
- The difference between bounding-box areas.
- The model patch area.
- The overlap between two object masks.

### 11.3 Noise floor

Small boundary growth may be ordinary prediction noise rather than recovered hidden shape. Add a configurable minimum meaningful hole area, preferably supporting both an absolute pixel floor and a modal-area-relative floor. Values below the effective floor count as zero for pair role assignment while remaining available in diagnostics.

## 12. Pairwise Occluded-Object Selection

For each cross-class overlap edge between objects A and B:

1. Retrieve A's validated hole area.
2. Retrieve B's validated hole area.
3. Apply the configured noise floor to both.
4. Compare the effective areas.
5. If A is larger beyond the tie tolerance, assign A as occluded and B as occluder for that pair.
6. If B is larger beyond the tie tolerance, assign B as occluded and A as occluder.
7. If the areas are equal or within tolerance, record the pair as ambiguous.
8. Do not infer a role from total mask size, class name, vertical position, mask contact, or a depth-order heuristic.

For an ambiguous pair, the safe default is to skip hidden-content reconstruction for that pair. Both validated completions may be retained for diagnostics, but neither should receive a potentially destructive occluder-removal mask based on an unsupported tie-break.

## 13. Multi-Object Overlap Policy

A single object may overlap multiple classes. Complete each unique object only once, then make decisions per graph edge.

For each object selected as occluded on one or more edges:

1. Collect the partner objects that were assigned as its occluders.
2. Build a union of only those partners' modal masks.
3. Limit that union to the reconstruction-relevant region defined in Section 14.
4. Run one reconstruction for the occluded object using the union mask, rather than repeatedly inpainting it once per partner.

If an object is occluded on one edge and occluder on another, retain both pairwise roles. Its own reconstruction uses only partners that occlude it. Its modal mask may separately contribute to reconstruction masks of objects it occludes. This avoids forcing a global depth ordering that the design explicitly does not require.

## 14. Hidden RGB Reconstruction with Inpainting

### 14.1 Reconstruction objective

SDAmodal supplies shape, not hidden RGB. The inpainting stage must produce a source canvas in which the selected occluded object visually continues through the area formerly occupied by its occluder.

### 14.2 Reconstruction region

For an occluded object:

1. Start with its completion-hole mask.
2. Form the union of modal masks belonging to pairwise-assigned occluders.
3. Intersect or otherwise constrain the occluder union to the occluded object's validated amodal support and a narrowly expanded seam region.
4. Include the completion hole itself so all unknown hidden pixels are eligible for generation.
5. Expand/feather the final reconstruction mask only enough to remove occluder remnants and blend boundaries.
6. Exclude unrelated portions of a large occluding object outside the amodal support.

The conceptual reconstruction mask is therefore the hidden area plus relevant occluder pixels inside or immediately adjacent to the occluded object's completed support, not the occluder's entire full-image mask.

### 14.3 Reconstruction input and output

The inpainting adapter receives:

- The original RGB image.
- The reconstruction mask.
- Context describing the occluded object's semantic class if the active inpainting model supports prompting.

The returned image remains full resolution. It is the reconstructed object canvas for that object only; it must not replace the global source image used for other objects.

### 14.4 Inpainting model suitability

The current LaMa/SDXL interface is used for background removal as well as per-object color refinement. Hidden-object reconstruction has different semantics. The integration should initially reuse the registered inpainting interface but add a reconstruction-specific invocation policy:

- LaMa receives the constrained reconstruction mask and surrounding visible object context.
- A prompt-capable model receives an object-continuation prompt based on the semantic class, not the current empty-background prompt.
- Background-removal and object-reconstruction configuration must remain separate even if they use the same model class.

If evaluation shows the configured inpainting backend cannot reconstruct object appearance reliably, introduce a separate reconstruction model category later. That is an evaluation-driven extension, not a prerequisite for the first integration.

### 14.5 Reconstruction validation and fallback

Validate:

- Output dimensions and RGB format.
- No unintended changes outside the allowed reconstruction/blend region.
- Retention of known visible object pixels outside the blend band.
- Availability of RGB content throughout the completion hole.

If reconstruction fails, do not run BiRefNet against the original occluder-filled pixels using the amodal mask. Fall back to the object's ordinary modal path and mark the completion as non-user-visible for that object.

## 15. BiRefNet After Reconstruction

### 15.1 Ordinary objects

For an object that is not selected as occluded, or whose reconstruction falls back:

- Use the original source image.
- Black out pixels outside the modal support as the current guided-matting path does.
- Run BiRefNet.
- Constrain the resulting alpha to a small dilation of the modal mask.

### 15.2 Successfully reconstructed occluded objects

For a successfully reconstructed object:

1. Use its reconstructed object canvas, never the original occluder-filled source.
2. Black out pixels outside the validated amodal support, with the same narrow support dilation policy used for edge recovery.
3. Run BiRefNet after reconstruction.
4. Constrain its alpha to the amodal support dilation.
5. Preserve alpha over both visible and reconstructed portions.
6. Use the completed object's amodal bounding box when cropping the returned layer.

This sequencing enforces the required order:

```text
amodal completion -> hole comparison -> occluder removal and reconstruction -> BiRefNet
```

### 15.3 Per-object matting inputs

Refactor `_refine_masks()` conceptually from one shared image plus a mask list into per-object matting inputs. Each object record chooses its own source canvas and support mask. This is required because multiple occluded objects may each have a different reconstructed canvas.

## 16. Object-Layer Extraction

The layer-extraction stage must consume object records rather than synchronized lists.

For each object:

- Use its selected canvas: original or reconstructed.
- Use its validated soft alpha.
- Use modal bounds for ordinary objects.
- Use amodal bounds for successfully reconstructed objects.
- Preserve the existing display label.
- Ensure the RGB crop comes from the same canvas used for matting.
- Continue the existing foreground/background color refinement only if it does not overwrite reconstructed RGB with pixels derived from the original occluder-filled source.

The existing `_extract_object_layers()` internally inpaints a component background to refine colors. Review this logic carefully during implementation: for completed objects, all calculations involving foreground RGB must use the reconstruction canvas, while any auxiliary background estimate must remain isolated to color/alpha solving.

## 17. Final Background Generation

Final background generation remains a separate concern from hidden-object reconstruction.

Use the original source image as the background-inpainting input. Build the removal union primarily from:

- Every object's original modal mask.
- Its resulting visible/soft-alpha coverage as appropriate.

Do not automatically add every amodal completion hole to the global background-removal mask. Those pixels were hidden behind another modal object and will already be removed when that modal occluder is removed. Blindly adding amodal regions could erase genuine background outside the visible-object union.

The final background pass should still run once after all layers are prepared.

## 18. File-by-File Modification Plan

### 18.1 `backend/models/base.py`

- Add the completion abstraction.
- Define a batch-oriented, image-plus-masks contract.
- Keep SDAmodal/DIFT-specific concepts out of the base interface.

### 18.2 `backend/models/registry.py`

- Add the completion category.
- Preserve current registration behavior for all existing categories.

### 18.3 `backend/models/manager.py`

- Import the completion adapter registration module.
- Add completion-model lifecycle state and lazy getter.
- Make warm-up configurable so completion is not necessarily loaded at startup.

### 18.4 `backend/config.yaml` and `backend/config.py`

- Add the completion model selection and adapter parameters.
- Add overlap/completion validation thresholds and tie policy.
- Add reconstruction-specific inpainting configuration separate from background-removal prompts/settings.
- Reuse the existing generic configuration lookup where possible.

### 18.5 New focused completion adapter module under `backend/models/completion/`

- Own SDAmodal construction, checkpoint loading, DIFT initialization, image normalization, shared feature extraction, per-object patch preparation, inference, restoration, and output validation.
- Register itself in the completion category.
- Hide the research-code interface from `backend/image_processor.py`.

### 18.6 `backend/models/completion/inference.py` and imported research modules

- Repair package imports.
- Remove hard-coded device placement.
- Accept in-memory features.
- Remove production dependence on filenames and feature directories.
- Remove deprecated NumPy aliases and production-path debug dependencies.
- Keep unrelated order-estimation utilities out of the application call path.

### 18.7 `backend/models/completion/dift/extract_dift_amodal.py` and DIFT modules

- Convert the CLI-oriented extractor into or wrap it with a reusable in-memory component.
- Preserve CLI functionality only as an optional standalone path.
- Resolve the canonical featurizer import within the package.
- Accept the application-selected device.
- Return all required feature levels without writing them to disk.

### 18.8 `backend/image_processor.py`

- Introduce the internal object record.
- Preserve semantic class separately from output label.
- Calculate merged modal boxes.
- Build the cross-class overlap graph.
- Batch-complete unique incident objects.
- Validate predictions and calculate hole areas.
- Make pairwise role decisions.
- Build per-object reconstruction masks.
- Run reconstruction before BiRefNet.
- Refactor matting and layer extraction to use per-object canvases and supports.
- Keep final background generation based on original visible-object removal.

### 18.9 Tests

- Extend `tests/test_image_processor.py` for orchestration behavior.
- Add focused completion-adapter tests in a completion-specific test module.
- Add import-isolation, device-routing, and in-memory DIFT tests.
- Keep real-model acceptance tests separate from deterministic unit tests.

## 19. Detailed Implementation Sequence

### Phase 1: Freeze contracts and diagnostics

1. Document the object-record fields and canonical mask/box formats.
2. Define overlap semantics, including positive-area intersection and same-class exclusion.
3. Define completion validation, noise floor, and tie tolerance configuration.
4. Define adapter batch input/output identity guarantees.
5. Define diagnostic fields needed to explain every completion and fallback.

Deliverable: stable internal contracts that prevent later adapter and orchestrator work from diverging.

### Phase 2: Isolate the research packages

1. Inventory all imports in the production inference chain.
2. Establish package initializers and canonical package roots.
3. Convert ambiguous SDAmodal imports.
4. Resolve the canonical DIFT featurizer module.
5. Remove production-path debug, visualization, training, and optional evaluation imports.
6. Verify imports under the normal backend startup path.

Deliverable: import-safe completion modules with no working-directory dependency.

### Phase 3: Normalize device behavior

1. Trace tensor and model placement end to end.
2. Remove direct CUDA and device-index assumptions.
3. Make checkpoint loading portable.
4. Ensure DIFT and SDAmodal share the selected device by default.
5. Ensure helper tensors inherit device/dtype correctly.
6. Verify fake inference on CPU and configured-device routing without real weights.

Deliverable: one explicit device owner and no hidden placement decisions.

### Phase 4: Build the in-memory DIFT/SDAmodal adapter

1. Add the completion base abstraction.
2. Register and lazily manage the SDAmodal adapter.
3. Wrap DIFT as an in-memory extractor.
4. Define and validate the four-level feature pyramid.
5. Extract once per source image.
6. Project completion boxes into each feature level.
7. Prepare feature and modal patches.
8. Run SDAmodal for each unique requested object.
9. Restore full-image masks and preserve modal pixels.
10. Return results by stable object identity.

Deliverable: a standalone batch adapter that performs no feature-file I/O.

### Phase 5: Add overlap triggering and hole comparison

1. Convert segmentation outputs into object records.
2. Calculate modal bounding boxes after semantic grouping.
3. Build the cross-class overlap graph.
4. Verify no-overlap bypass without loading completion.
5. Complete all unique graph vertices once.
6. Validate outputs and calculate completion holes.
7. Compare hole areas per graph edge.
8. Record occluded, occluder, or ambiguous pair decisions.

Deliverable: deterministic role assignments based only on the approved logic.

### Phase 6: Add hidden-object reconstruction

1. Collect assigned occluders per occluded object.
2. Build constrained reconstruction masks.
3. Add reconstruction-specific inpainting settings.
4. Run one inpainting pass per selected occluded object.
5. Validate reconstructed canvases.
6. Fall back to modal processing on reconstruction failure.

Deliverable: per-object canvases containing reconstructed hidden appearance.

### Phase 7: Move matting after reconstruction

1. Refactor matting to accept a source canvas and support mask per object.
2. Keep ordinary objects on original-image/modal-mask inputs.
3. Use reconstruction-canvas/amodal-mask inputs for successfully reconstructed objects.
4. Constrain alpha to the appropriate support.
5. Crop reconstructed objects using amodal bounds.
6. Verify visible objects retain current behavior.

Deliverable: complete RGBA layers for reconstructed objects and unchanged modal layers for bypassed objects.

### Phase 8: Preserve final background semantics

1. Keep final background inpainting based on original visible objects.
2. Ensure reconstruction canvases never leak into the final background input.
3. Confirm amodal holes do not unnecessarily enlarge the global removal mask.
4. Run the existing final background pass once.

Deliverable: completed object layers plus a clean object-free background.

### Phase 9: Real-model evaluation and tuning

1. Run the supplied checkpoints on a fixed acceptance set.
2. Review modal masks, amodal masks, completion holes, reconstruction masks, reconstructed canvases, and final alpha mattes.
3. Tune only validation thresholds, noise floor, tie tolerance, box expansion, and blend expansion.
4. Measure latency and peak GPU memory.
5. Enable mixed precision or offloading only after output-equivalence checks.

Deliverable: measured configuration suitable for enabling the workflow in normal requests.

## 20. Testing and Acceptance Plan

### 20.1 Overlap-trigger unit tests

- Different-class boxes with positive intersection create one edge.
- Same-class boxes never create an edge.
- Edge-only contact does not create an edge.
- No-overlap scenes bypass completion and DIFT.
- Each unordered pair is evaluated once.
- Three-object overlap produces the correct graph without duplicate vertices.

### 20.2 Completion-area tests

- Hole area counts only amodal-minus-modal pixels.
- Modal pixels are always retained.
- Invalid predictions fall back to zero hole area.
- Noise-floor growth is treated as zero for decisions.
- Larger hole assigns the corresponding object as occluded.
- Equal or near-equal holes produce an ambiguous decision and no reconstruction for that edge.

### 20.3 Import tests

- Completion imports succeed after `backend.models` is already imported.
- Imports do not depend on the current working directory.
- Production adapter imports do not require debugging or visualization packages.
- API startup does not import training-only modules.

### 20.4 Device tests

- The selected device reaches DIFT and SDAmodal.
- Parameters, modal patches, and feature tensors are colocated.
- No literal device index controls placement.
- Checkpoint deserialization is device portable.
- NumPy conversion occurs only after output transfer to CPU.

### 20.5 DIFT tests

- No extraction for no-overlap input.
- One extraction for any non-empty completion batch.
- Four required levels are validated.
- Feature crops correspond to each expanded box.
- A multi-partner object is completed once.
- No feature files are read or written.

### 20.6 Reconstruction tests

- Reconstruction masks contain the completion hole.
- Only pairwise-assigned occluder masks contribute.
- Unrelated parts of an occluder outside amodal support are excluded.
- Multiple occluders are unioned into one per-object reconstruction.
- Reconstruction failure returns the object to the modal path.
- Original visible pixels outside the blend region remain unchanged.

### 20.7 Matting-order tests

- BiRefNet is not called for an occluded object before its reconstruction completes.
- Successfully reconstructed objects use the reconstruction canvas and amodal support.
- Ordinary objects use the original image and modal support.
- Returned completed layers use amodal bounds.
- Failed or ambiguous cases use modal bounds.

### 20.8 End-to-end acceptance scenes

Evaluate at minimum:

- Two non-overlapping objects of different classes.
- Two overlapping objects of different classes with one clearly larger completion hole.
- Reverse of the preceding case.
- Equal or near-equal completion holes.
- Two overlapping objects of the same class.
- One object overlapping two different classes.
- A three-object overlap chain.
- Small and border-touching objects.
- Invalid completion output.
- Inpainting failure.
- Already-complete masks whose bounding boxes overlap.

For each scene, retain request diagnostics and visual artifacts from every stage for manual inspection.

## 21. Error Handling and Fallback Rules

| Failure | Required behavior |
| --- | --- |
| No detections | Return the original background as today. |
| Empty object mask | Drop that object before box comparison. |
| No cross-class overlap | Skip DIFT and SDAmodal entirely. |
| DIFT batch failure | Fall back all completion candidates to modal processing; continue request if possible. |
| One SDAmodal object failure | Fall back that object to its modal mask; preserve other valid results. |
| Invalid amodal growth | Reject that completion and set hole area to zero. |
| Hole-area tie | Mark pair ambiguous and do not reconstruct from that edge. |
| Reconstruction failure | Use original image and modal mask for that object. |
| BiRefNet failure | Follow the existing request-level error contract; do not emit a malformed layer. |
| Final background failure | Follow the existing inpainting error contract independently of object reconstruction. |

Fallbacks must be visible in structured logs and diagnostics. A request must never silently claim that an object is complete when it actually returned through the modal fallback path.

## 22. Observability and Performance Metrics

Record per request:

- Number of segmented/grouped objects.
- Number of cross-class overlap edges.
- Number of unique completion candidates.
- Whether DIFT ran and its elapsed time.
- SDAmodal elapsed time per object and total.
- Modal area, amodal area, raw hole area, and effective hole area per object.
- Pairwise role decisions and tie reasons.
- Reconstruction-mask area and inpainting time per occluded object.
- Matting time per object.
- Fallback reason per stage.
- Peak GPU memory where available.

Do not log raw image data or feature tensors in normal operation. Optional visual artifact saving should be restricted to an explicit diagnostic mode.

## 23. Non-Goals

The following are explicitly outside this integration:

- Predicting a global depth ordering.
- Training or fine-tuning SDAmodal, DIFT, BiRefNet, or the inpainting model.
- Discovering a learned occlusion signal.
- Running completion on non-overlapping objects.
- Using same-class overlap as a trigger.
- Changing the external API schema unless later required for diagnostics.
- Replacing the existing final-background generation architecture.
- Treating amodal shape prediction alone as completed RGB output.

## 24. Approval and Implementation Gate

No Python implementation should begin until this plan is reviewed and approved. Approval should confirm:

1. Positive-area cross-class bounding-box overlap is the sole completion trigger.
2. Both objects in every overlap pair are completed.
3. Validated amodal-minus-modal pixel count is the comparison metric.
4. The larger completed area identifies the occluded object for that pair.
5. Ambiguous ties skip reconstruction rather than using a depth heuristic.
6. Hidden RGB reconstruction precedes BiRefNet.
7. Multi-object scenes use pairwise decisions without constructing a global depth order.
8. Modal fallback remains the safe response to completion or reconstruction failure.
9. DIFT features are extracted once per relevant request and remain entirely in memory.
10. Checkpoints and `extract_dift_amodal.py` are assumed available and ready, so no acquisition phase is required.
