# Image Processor Option 3 Refactor Design

## Objective

Refactor `backend/image_processor.py` into focused, function-based pipeline
modules with explicit model dependencies. Preserve the current observable
behavior of `process_image()` and `process_masks()`, including lazy loading,
no-overlap completion bypass, modal BiRefNet input, layer output, and final
background generation.

This refactor prepares a clear insertion point for hidden RGB reconstruction,
but does not implement that feature.

## Current Problems

`backend/image_processor.py` currently owns data contracts, model lookup,
segmentation, completion, occlusion state, reconstruction masks, matting,
layer rendering, background inpainting, and orchestration in one 414-line
module.

Stage functions obtain models through the global `model_manager`, so their
real dependencies are not visible in their signatures. Tests must replace
global manager behavior instead of passing fake dependencies directly.

Later stages also rely on synchronized lists of masks, labels, and alpha
mattes. Those lists are correct only while their length and ordering remain
identical.

## Chosen Architecture

Create a `backend/pipeline/` package containing functional stages. The stage
functions receive the processor, model adapter, or inference callable they
need. Only the orchestrator and compatibility facade resolve dependencies from
`model_manager`.

Do not create service classes for stateless image transformations. Existing
model adapters remain the stateful model-owning objects.

```text
backend/
├── image_processor.py       # compatibility facade
└── pipeline/
    ├── __init__.py
    ├── types.py             # pipeline dataclasses
    ├── segmentation.py      # SAM3 extraction and grouping
    ├── completion.py        # overlap linking and SDAmodal completion
    ├── reconstruction.py    # pair application and reconstruction masks
    ├── matting.py           # BiRefNet alpha generation
    ├── layers.py            # per-object RGBA layer generation
    ├── background.py        # final removal mask and inpainting
    └── orchestrator.py      # process_masks and process_image
```

## Data Contracts

Move these dataclasses without changing their existing constructor fields:

- `ObjectLayer`
- `ProcessResult`
- `DetectedObject`

Add `soft_alpha: Optional[np.ndarray] = None` to `DetectedObject`. Matting
results will be stored on the corresponding object, eliminating the parallel
`soft_alphas` list in the new orchestrator.

Do not add `reconstruction_canvas`, `matting_canvas`, or amodal layer behavior
in this refactor. Those belong to the hidden RGB reconstruction feature.

## Stage Interfaces

### Segmentation

`segmentation.extract_objects(image, keywords, processor)` receives an already
resolved SAM3 processor. It performs the same prompt reset, segmentation,
normalization, same-class grouping, bounding-box calculation, labeling, and
logging as the current `_extract_objects()`.

It must not import or access `model_manager`.

### Completion

`completion.link_overlap_partners(objects)` remains model-independent and
preserves bidirectional partner linking.

`completion.get_completion_candidates(objects)` returns each object with at
least one partner exactly once, preserving object order.

`completion.complete_objects(image, candidates, completion_model)` calls the
supplied completion adapter, maps outputs to candidates, and calculates the
same completion-hole masks and areas.

The orchestrator calls `get_completion_candidates()` first. It resolves the
completion model only when candidates exist, preserving the current no-overlap
lazy-loading behavior.

### Reconstruction

`reconstruction.apply_pair_decisions(objects, decisions)` preserves current
role assignment and ambiguous-pair handling.

`reconstruction.build_reconstruction_masks(objects, kernel_size)` preserves
the current mask formula and mutates only `reconstruction_mask`.

These functions remain independent of model lookup. Hidden RGB inpainting is
not added.

### Matting

`matting.refine_objects(image_np, objects, matte)` receives the bound matting
callable, equivalent to the current adapter's `process` method. It processes
each object's modal mask against the original image and stores the resulting
NumPy alpha in `detected.soft_alpha`.

This deliberately preserves the current modal behavior. Objects with no
usable alpha remain represented by their object record and may be skipped by
layer construction in the same way an empty alpha is currently skipped.

### Layers

`layers.extract_object_layers(image, image_np, objects, kernel_size, inpaint)`
receives the bound inpainting callable. It reads each object's `soft_alpha`
and `display_label`, performs the existing component-background inpainting,
color refinement, crop, encoding, and logging, and returns `ObjectLayer`
records in object order.

The function must validate that a usable `soft_alpha` is present before using
it. It must not resolve the inpainting model itself.

### Background

`background.generate_final_background(image, objects, kernel_size, inpaint)`
builds the removal union from every object's modal mask and stored soft alpha,
then performs the existing expansion, inpainting, resizing, and background
refinement.

It must not include amodal holes or reconstruction masks in the removal union.

## Orchestration

`pipeline.orchestrator.process_masks(image, keywords, manager=model_manager)`
will:

1. Convert the image to RGB.
2. Resolve the segmentation processor and extract objects.
3. Return an empty list when no objects exist.
4. Link cross-class overlaps.
5. Build completion candidates.
6. Resolve and call the completion model only when candidates exist.
7. Return amodal masks for completed objects and modal masks otherwise.

`pipeline.orchestrator.process_image(image, keywords, manager=model_manager)`
will preserve the current runtime order:

1. Normalize RGB data.
2. Resolve SAM3 and extract objects.
3. Preserve the existing no-detection result.
4. Link overlap partners.
5. Lazily resolve SDAmodal only when completion candidates exist.
6. Assign pair roles and build reconstruction masks.
7. Resolve BiRefNet and store one soft alpha per object.
8. Resolve the inpainting adapter and reuse its bound `process` callable for
   layer extraction and final-background generation.
9. Return the same `ProcessResult` contract.

The optional `manager` parameter supports direct dependency injection at the
orchestration boundary while retaining the global manager as the production
default.

## Final Public Facade

Keep `backend/image_processor.py` as a minimal public facade. It will re-export
the three dataclasses and the two public entry points only.

Migrate `examples/partial_pipeline_v1.ipynb` and deterministic tests to import
stage functions from their owning `backend.pipeline` modules. The notebook
will explicitly resolve SAM3, SDAmodal, BiRefNet, and inpainting dependencies
at first use and preserve the orchestrator's lazy-loading order.

Remove the transitional private wrappers after all internal consumers are
migrated. Historical notebooks that imported old private constants or helpers
are not supported compatibility APIs.

## Dependency Rules

- Stage modules must not import `model_manager`.
- `orchestrator.py` may import the default `model_manager`.
- `image_processor.py` must not resolve models or import stage internals.
- Pure geometry and mask calculations must remain separate from model lookup.
- Model adapters and configuration remain unchanged.

## Error and Fallback Behavior

This is a behavior-preserving refactor. It must retain:

- Empty output from `process_masks()` when segmentation finds no objects.
- Original background from `process_image()` when segmentation finds no
  objects.
- No SDAmodal initialization when there is no overlap candidate.
- Exact-tie ambiguity behavior.
- Modal-mask BiRefNet support.
- Modal and soft-alpha final-background removal semantics.
- Existing exception propagation from model inference.

Completion-output validation and recovery are not introduced here; they
remain separate work documented in `coding_progress.md`.

## Testing Strategy

Follow a characterization-first migration:

1. Preserve every existing behavioral assertion in
   `tests/test_image_processor.py`. Replace tests that require the public
   orchestrator to call monkey-patched facade-private functions with tests
   against `pipeline.orchestrator` using an injected fake manager.
2. Add direct unit tests for stage functions using explicitly supplied fake
   processor/model callables.
3. Verify stage modules do not reference `model_manager`.
4. Verify no-overlap orchestration does not request the completion model.
5. Verify the orchestrator resolves the inpainting model once and reuses it.
6. Migrate direct private-wrapper tests and notebook structural tests to the
   owning stage modules before removing compatibility exports.
7. Run the full deterministic suite while retaining the acknowledged server
   checkpoint-path test mismatch unchanged.

## Migration Sequence

1. Create `pipeline/types.py` and keep facade re-exports.
2. Extract segmentation with explicit processor input.
3. Extract completion and candidate selection.
4. Extract reconstruction functions.
5. Extract object-based matting.
6. Extract object-based layer and background rendering.
7. Add the new orchestrator using explicit stage dependencies.
8. Migrate the debugging notebook and tests to explicit stage imports.
9. Reduce `image_processor.py` to public entry points and type exports.
10. Run focused and repository-level verification after each extraction.

## Non-Goals

- Hidden RGB reconstruction.
- Amodal BiRefNet input or amodal layer crops.
- Completion validation and fallback changes.
- New model classes or model configuration changes.
- Changes to the existing checkpoint path selected for the server.
