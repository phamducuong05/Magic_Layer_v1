# SmartEraser Background Inpainting Integration Design

## Goal

Integrate SmartEraser as an alternative background-inpainting strategy while
keeping `original_lama` as the default and preserving the existing LaMa
behavior. Refactor the upstream single-image inference path into small,
typed, production-oriented components that fit the existing model registry
and lifecycle.

## Scope

The integration covers SmartEraser model loading, image and mask
preprocessing, inference, postprocessing, model registration, configuration,
unloading, and automated tests.

The upstream dataset synthesis, training, distributed evaluation, shell
launchers, and Gradio applications remain reference code and are not imported
by the backend runtime. The `SmartEraser/Model_framework/ckpts` directory and
all files below it remain unchanged. The backend never downloads weights.

## Existing Architecture

Background inpainters implement
`BaseBackgroundInpaintingModel.process(image, mask, prompt="",
artifact_callback=None)`. `ModelRegistry` maps the configured strategy name
to an adapter, `ModelManager` lazy-loads the active adapter, and the pipeline
passes the adapter's `process` method into layer extraction and final
background generation.

The strategy switch already exists at:

```yaml
models:
  background_inpainting:
    active: original_lama
```

SmartEraser will use this mechanism. The pipeline orchestrator will not gain
model-specific branches.

## Upstream SmartEraser Inference Analysis

SmartEraser uses a Stable Diffusion 1.5 inpainting pipeline modified for
masked-region guidance. Unlike conventional mask-and-inpaint behavior, the
masked pixels are retained in the image supplied to the inpainting UNet. A
CLIP vision encoder embeds a crop of the masked object, and a learned MLP
inserts that visual feature into the text-conditioning sequence.

The runtime requires two local weight roots:

- `SmartEraser/Model_framework/ckpts/smarteraser-weights`
- `SmartEraser/Model_framework/ckpts/clip-vit-large-patch14`

The first root also contains `clip_mlp_weight.pth`. Both roots are already
present on the server.

For a single image, upstream inference:

1. Converts the source to RGB and the mask to binary grayscale.
2. Selects crop mode when the mask bounding box fits within the shorter image
   dimension; otherwise it selects padding mode.
3. Produces a 512 by 512 source and mask.
4. Builds a CLIP guidance crop by placing the masked object on white and
   cropping to its bounding box.
5. Tokenizes the removal prompt and unconditional prompt.
6. Replaces a text token with the learned visual feature.
7. Runs the custom diffusion pipeline with the original masked pixels retained
   as masked-image guidance.
8. Restores the generated image to the source resolution and composites only
   the requested region over the original image.

The upstream `app_remove.py` and `demo_app.py` duplicate preprocessing and
controller logic and mix it with Gradio, diagnostics, global random state, and
CUDA cache operations. `inference_dis.py` additionally mixes dataset I/O,
multiprocessing, distributed device setup, and output persistence. These
responsibilities will not enter the backend adapter.

## Selected Architecture

Create a SmartEraser strategy package under the existing background
inpainting package:

```text
backend/models/background_inpainting/smarteraser/
├── __init__.py
├── adapter.py
├── geometry.py
└── runtime.py
```

### Geometry

`geometry.py` owns deterministic, dependency-light image transformations:

- normalize source and mask modes and sizes;
- binarize masks with nearest-neighbor semantics;
- select crop or padding mode;
- transform source and mask to the configured square resolution;
- create the white-background CLIP guidance crop;
- restore generated RGB to the original image size.

Geometry functions return typed immutable metadata rather than loose tuples.
They do not load models, mutate global random state, or know about registry
configuration.

Empty masks are detected before bounding-box calculations. The adapter returns
a copy of the original RGB image without invoking inference.

### Runtime

`runtime.py` owns heavyweight resources and one inference call. It:

- resolves local paths relative to the repository root;
- validates that both model directories and `clip_mlp_weight.pth` exist;
- loads the custom SmartEraser diffusion pipeline and CLIP visual-token
  components once;
- uses the configured device and dtype, forcing float32 on CPU;
- moves input IDs and CLIP pixels to the correct device and dtype;
- creates a per-call `torch.Generator` when the pipeline supports it, avoiding
  mutation of process-wide random state;
- runs under inference mode;
- releases pipeline, CLIP model, processor, and tokenizer references on close.

The runtime imports the vendored custom pipeline and visual-token model
directly. It does not import Gradio, training code, dataset classes,
multiprocessing code, or launch scripts.

### Adapter

`adapter.py` implements `BaseBackgroundInpaintingModel` and registers
`smarteraser` in `ModelRegistry`.

Its `process` method:

1. Converts the source to RGB.
2. Uses shared background-inpainting mask helpers to construct generation and
   composition masks.
3. Returns the source unchanged when the composition mask is empty.
4. Uses geometry preprocessing to produce one 512-square inference request.
5. Calls the runtime.
6. Restores the generated crop or padded image.
7. Uses `preserve_unmasked_pixels` so pixels outside the composition mask are
   identical to the input.
8. Emits the same diagnostic callback stages currently consumed by the
   background diagnostic writer, preserving compatibility.

`unload` closes runtime-owned resources before delegating to the base cleanup.

## Configuration and Switching

Keep the current default:

```yaml
models:
  background_inpainting:
    active: original_lama
```

Add:

```yaml
    smarteraser:
      checkpoint_dir: SmartEraser/Model_framework/ckpts/smarteraser-weights
      clip_dir: SmartEraser/Model_framework/ckpts/clip-vit-large-patch14
      resolution: 512
      num_inference_steps: 50
      guidance_scale: 1.5
      seed: 42
      dtype: float16
      prompt: Remove the instance of object
      negative_prompt: ""
      generation_mask_expansion: 1
      composition_mask_expansion: 1
      feather_radius: 5.0
```

Selecting SmartEraser requires only changing `active` to `smarteraser`.
Changing it back restores the existing `original_lama` path. No weight path is
created or modified by code.

`backend/models/manager.py` imports the SmartEraser strategy package for
registration, matching existing strategies. The orchestrator remains generic
and unchanged unless a test exposes an existing assumption tied specifically
to LaMa.

## Error Handling

Initialization fails with a precise `FileNotFoundError` when a required local
weight directory or `clip_mlp_weight.pth` is absent. It never falls back to a
Hugging Face model ID and never attempts a download.

Invalid configuration values such as a non-positive resolution, step count,
or guidance scale fail during adapter initialization. Unsupported dtype names
raise `ValueError`. CPU execution uses float32 even when float16 is configured.

An empty mask is a successful no-op. A mask with a different size is resized
to the source dimensions with nearest-neighbor resampling. Results are always
RGB and have the same dimensions as the source.

Inference exceptions are allowed to propagate to the existing pipeline error
boundary. The adapter does not silently fall back to LaMa because silent
fallback would hide deployment and quality problems.

## Testing Strategy

Tests will be written before production changes.

Geometry tests cover:

- binary mask normalization and size alignment;
- crop selection and restoration;
- padding selection and restoration;
- guidance crop construction;
- empty-mask behavior;
- output size and mode.

Adapter tests use a fake runtime dependency and cover:

- no runtime call for an empty mask;
- configured arguments forwarded to runtime;
- preservation of all pixels outside the composition mask;
- artifact callback compatibility;
- runtime close during unload.

Runtime tests replace heavyweight loaders with fakes and cover:

- local path validation without network fallback;
- device and dtype selection;
- prompt and CLIP visual-token wiring;
- configured steps, guidance, seed, image, and mask passed to the pipeline;
- release of owned resources.

Integration tests cover:

- registry contains the `smarteraser` strategy;
- config contains SmartEraser while `original_lama` remains active;
- manager can instantiate the selected strategy when configuration is
  temporarily overridden;
- existing Original LaMa adapter, runtime, lifecycle, pipeline architecture,
  and package tests remain green.

## Acceptance Criteria

- `original_lama` remains the configured default.
- `active: smarteraser` selects SmartEraser through the existing registry and
  manager without an orchestrator branch.
- The backend uses only local server weights and never changes or downloads
  anything under `ckpts`.
- SmartEraser accepts the same PIL image and mask interface as LaMa and returns
  an RGB PIL image at the original size.
- Pixels outside the requested composition mask remain unchanged.
- Runtime responsibilities are separated according to SRP and public
  functions and methods have explicit type hints.
- SmartEraser-specific tests and the existing LaMa regression tests pass.
