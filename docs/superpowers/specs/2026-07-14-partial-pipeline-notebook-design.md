# Partial Pipeline V1 Notebook Design

## Objective

Update `examples/partial_pipeline_v1.ipynb` into an executable, visually
inspectable walkthrough of the workflow currently implemented by
`backend.image_processor.process_image()`.

The notebook will cover segmentation, grouped bounding boxes, cross-class
overlap detection, conditional amodal completion, pair-role decisions,
reconstruction-mask construction, the current modal BiRefNet path, RGBA layer
extraction, and final-background inpainting.

## Scope Boundary

The named CLI, `backend/run_mask_completion.py`, currently calls
`process_masks()` and stops after returning and saving modal or amodal masks.
The broader matting and inpainting workflow is implemented in
`backend.image_processor.process_image()` and is the approved source for the
notebook's execution order.

Hidden RGB reconstruction is not implemented yet. The notebook will visualize
each `reconstruction_mask`, then explicitly show that the current runtime still
sends the original RGB image and modal mask to BiRefNet. It will not present a
reconstruction canvas or amodal layer as if either already worked.

## Chosen Approach

The notebook will call the same private orchestration helpers used by
`process_image()` in the same order. This provides access to intermediate state
without monkey-patching production code or running the GPU-heavy pipeline a
second time.

Notebook-only visualization and decoding helpers may be defined locally. They
must not replace or modify pipeline decisions. When the notebook needs to show
an internal mask that a production helper does not return, it will reproduce
the corresponding expression from `image_processor.py` and label it as a
diagnostic view.

## Notebook Structure and Data Flow

### 1. Setup and imports

- Resolve the repository root from either the repository root or the
  `examples/` directory and add it to `sys.path` only when necessary.
- Import PIL, NumPy, OpenCV, Matplotlib, base64/bytes utilities, and the exact
  backend functions used by `process_image()`.
- Enable informative logging.
- Define small notebook-only helpers for subplot normalization, mask overlay,
  bounding-box drawing, and base64 image decoding.

### 2. Model configuration

- Display the configured device and active model names.
- Preserve `ModelManager` lazy loading instead of warming every model upfront.
- Explain which stage first initializes SAM3, SDAmodal/DIFT, BiRefNet, and the
  configured inpainting model.

### 3. Input loading

- Keep `image_path` and `keywords` in one editable configuration cell.
- Require the input file to exist and raise a clear `FileNotFoundError`; do not
  silently substitute a blank image that cannot validate model behavior.
- Convert the image once to RGB and create the same `uint8` NumPy array used by
  `process_image()`.
- Display the source image and basic dimensions.

### 4. SAM3 extraction and grouping

- Call `_extract_objects(image, keywords)`.
- Display object ID, semantic class, display label, modal area, and grouped
  modal bounding box.
- Visualize every modal mask separately and as a colored overlay on the source
  image with its bounding box.
- Stop downstream execution clearly if no objects are detected.

### 5. Cross-class overlap detection

- Call `_link_overlap_partners(objects)`.
- Display every overlap pair and each object's partner IDs.
- Draw grouped bounding boxes and positive-area intersection rectangles.
- Explain that overlap is evaluated after same-class grouping and only across
  different semantic classes.

### 6. Conditional amodal completion

- Call `_complete_overlapping_objects(image, objects)`.
- Show modal mask, amodal mask, and completion-hole mask side by side for each
  completed object.
- Display completion-hole pixel area.
- For objects without partners, show that SDAmodal was skipped and the modal
  mask remains the effective mask.
- If there are no cross-class overlaps, proceed through the ordinary modal path
  without requiring the completion model.

### 7. Pair decisions

- Build the same hole-area mapping as `process_image()`.
- Call `assign_pair_roles(overlap_pairs, hole_areas)` and then
  `_apply_pair_decisions(objects, pair_decisions)`.
- Display each pair's two areas, selected occluded object, selected occluder,
  and ambiguity state.
- Treat equal areas as ambiguous exactly as production code does.

### 8. Reconstruction masks

- Calculate `kernel_size` with `_calc_kernel_size(image_np)`.
- Call `_build_reconstruction_masks(objects, kernel_size)`.
- Display each reconstruction mask on its own and over the source image.
- State explicitly that these masks are currently diagnostic outputs and are
  not consumed by later production stages.

### 9. Current BiRefNet matting path

- Build `raw_masks` from `DetectedObject.modal_mask` and labels from
  `DetectedObject.display_label`, exactly as `process_image()` currently does.
- Call `_refine_masks(image_np, raw_masks)`.
- Display every soft alpha matte, its numeric range, and an alpha overlay on
  the original image.
- Explain that amodal support will only enter this stage after hidden RGB
  reconstruction is implemented.

### 10. RGBA object layers

- Call `_extract_object_layers(image, image_np, soft_alphas, labels,
  kernel_size)`.
- Decode each returned `png_base64` value in memory.
- Display the RGBA crop over a checkerboard or neutral background and report
  its label, full-image offset, width, and height.
- Do not call the inpainting model separately; the production helper already
  performs its component-background pass and color/alpha refinement.

### 11. Final background

- Build a diagnostic removal union using the same modal masks, soft-alpha
  threshold, and expansion logic used by `_generate_final_background()`.
- Display the unexpanded coverage, expanded inpainting mask, and source-image
  overlay.
- Call `_generate_final_background(image, raw_masks, soft_alphas,
  kernel_size)` once and display the resulting background.

### 12. Result summary and optional export

- Construct the same `ProcessResult` fields from the already computed
  background and layers without rerunning `process_image()`.
- Display a compact summary of image dimensions and layer metadata.
- Provide an optional, disabled-by-default output cell that writes the final
  background and decoded RGBA layers to a user-selected directory.

## Execution and State Rules

- Cells are intended to run top to bottom in one kernel session.
- Every processing code cell has a preceding Markdown cell that explains the
  production operation, inputs, outputs, and fallback behavior.
- Later cells will use explicit guards for empty object, overlap, completion,
  decision, and reconstruction collections.
- The notebook will avoid calling `process_image()` after the staged run,
  because doing so would repeat all expensive model inference and increase GPU
  memory pressure.
- Model initialization remains lazy and occurs at the same first-use stage as
  the production pipeline.

## Error Handling

- Missing images produce an actionable file-path error.
- Empty keywords produce a validation error before model loading.
- No detections produce a clear terminal visualization and prevent invalid
  reductions over empty mask lists.
- No-overlap scenes skip SDAmodal and continue with modal masks.
- Ambiguous pairs do not receive occluder assignments or reconstruction masks.
- Model exceptions remain visible with their full traceback so the notebook is
  useful for debugging rather than silently hiding inference failures.

## Verification

Static notebook verification will check that:

- The notebook JSON is valid and contains alternating explanatory Markdown and
  executable code sections.
- All code cells compile without executing heavyweight models.
- Imports correspond to the current backend package layout.
- The ordered helper calls match `process_image()`.
- The old dummy-image fallback and outdated end-of-pipeline statement are
  absent.

The real-model acceptance run must be performed on the configured server/GPU by
executing all cells from a clean kernel. Acceptance requires visible outputs
for the source, grouped masks and boxes, overlaps, completion masks and holes,
decisions, reconstruction masks when applicable, alpha mattes, RGBA layers,
the final removal mask, and final background.

## Non-Goals

- Implement hidden RGB reconstruction.
- Change production pipeline behavior or model configuration.
- Treat reconstruction masks as reconstructed images.
- Add completion validation, fallback policy, tie tolerance, or other remaining
  steps from `coding_progress.md`.
- Benchmark latency or GPU memory in this notebook revision.
