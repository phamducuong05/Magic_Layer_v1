# SmartEraser Local-Context Inpainting Design

## Goal

Reduce SmartEraser's tendency to invent unrelated shapes, text, objects, and
decorations when removing objects from Canva-style artwork.

The implementation will limit each SmartEraser inference call to adaptive
local context around a group of nearby mask components and use more
conservative text guidance.

## Scope

This change applies only inside the SmartEraser adapter and geometry package.

The shared background-mask workflow remains unchanged:

1. combine visible modal masks;
2. include visible soft-alpha coverage;
3. fill holes;
4. apply the configured common dilation;
5. pass one final binary mask to the selected background model.

Original LaMa and SimpleLaMa continue to consume that final mask exactly once.
They do not perform component splitting, grouping, adaptive cropping, or
multiple inference calls.

An automatic model router that selects LaMa for small regions and SmartEraser
for large regions is a future feature and is explicitly outside this
implementation.

## Root Cause

The existing SmartEraser crop scales the source so its shorter side is 512,
then extracts a 512-by-512 window centered near the mask. In original-image
coordinates, the model therefore sees a square approximately as large as the
entire shorter side of the artwork even when the masked object is small.

That wide view can contain unrelated text, icons, decorations, and objects.
Those semantic cues increase the chance that the diffusion model generates
new content inside the inpainted region.

The final background workflow already preserves pixels outside the composition
mask and palette-refines generated pixels. Those safeguards limit where
hallucinations appear but cannot prevent invented structures inside the hole.

## Region Extraction

SmartEraser receives the same final binary mask produced by the shared
pipeline. It labels that mask using 8-connectivity.

Each connected component has:

- its compact binary mask cropped to the component bounding box;
- its axis-aligned bounding box;
- an adaptive candidate context box.

The implementation keeps one shared integer label map while extracting
components. It does not allocate one full-resolution array per component.
Compact masks are expanded to full image size only for the final groups that
are about to be inpainted.

No component analysis is performed for LaMa.

## Adaptive Context Formula

For an image of size `width` by `height` and one component bounding box:

```python
short_side = min(width, height)
object_span = max(component_width, component_height)
crop_side = min(
    short_side,
    max(
        object_span * context_scale,
        short_side * minimum_context_ratio,
    ),
)
```

The initial values are:

```yaml
context_scale: 3.0
minimum_context_ratio: 0.25
```

`context_scale: 3.0` normally leaves approximately one object span of context
on each side. `minimum_context_ratio: 0.25` prevents tiny components from
creating undersampled crops with too little background evidence.

The square is centered on the component bounding-box center. If it crosses an
image boundary, it is shifted inside the image instead of padded with
synthetic pixels.

If a component cannot fit in a square bounded by the image's shorter side,
the existing full-image padding path remains the fallback.

## Grouping Nearby Components

Before inference, components whose adaptive candidate context boxes intersect
are merged using transitive grouping. This means if A overlaps B and B
overlaps C, all three belong to one group.

Intersection candidates are looked up through a bounded uniform spatial grid.
The grid tracks one union-find root per group in each occupied cell, so a
dense cluster that has already become one group does not repeatedly compare
every member with every new component.

Grouping has two purposes:

- nearby fragments of the same visual object are inpainted together;
- distant objects do not force one crop to include most of the artwork.

The final group mask is the union of its component masks. A group's adaptive
crop is recomputed from the bounding box of that union.

Groups remain independent even if their final context boxes share some
unmasked context. Only their group masks are written into the accumulated
result, so one group cannot overwrite another group's output.

## Per-Group Inference

For every group:

1. build the group's generation and composition masks using SmartEraser's
   existing expansion and feather settings;
2. calculate and extract the adaptive local crop;
3. resize the image crop to 512-by-512 with bilinear interpolation;
4. resize the binary mask with nearest-neighbor interpolation;
5. run one SmartEraser inference call;
6. resize and restore the generated crop to its original coordinates;
7. copy generated pixels into a raw diagnostic accumulator using the
   generation mask;
8. blend generated pixels into the final accumulator using the composition
   mask.

Each inference call uses the original source image as context. Previously
generated groups are not fed back into later inference calls.

## Conservative Guidance

SmartEraser was trained and demonstrated with a short seven-token conditioning
sequence. Longer prose is truncated and increasing the token length would
move inference away from the training distribution.

The selected defaults are therefore short:

```yaml
num_inference_steps: 50
guidance_scale: 1.2
prompt: "Remove the instance of"
negative_prompt: "objects, text, decorations, artifacts"
seed: 42
```

Only `guidance_scale`, `prompt`, and `negative_prompt` change. Resolution,
steps, seed, dtype, mask expansion, feathering, CLIP loading, and model weights
remain unchanged.

## Diagnostics

The external diagnostic sequence remains:

- `00_input_mask.png`: the unchanged shared union mask;
- `01_after_smarteraser.png`: all group generations accumulated through their
  hard generation masks;
- `02_after_composition_blend.png`: all groups accumulated through their
  composition masks;
- `03_after_palette_refine.png`: the existing final palette refinement.

Per-group files are not introduced in this change.

## Components and Responsibilities

### `smarteraser/regions.py`

Owns:

- 8-connected component extraction;
- adaptive context-box calculation;
- transitive grouping of intersecting candidate boxes;
- immutable typed region/group descriptions.

It contains no model loading, inference, or image composition.

### `smarteraser/geometry.py`

Owns:

- extracting a supplied local crop;
- resizing an image and mask to the configured inference resolution;
- recording typed transform metadata;
- restoring a generated square to original-image coordinates;
- the existing visual-guidance crop.

### `smarteraser/adapter.py`

Owns:

- converting the incoming mask to binary;
- asking `regions.py` for local groups;
- building per-group generation and composition masks;
- orchestrating inference calls;
- accumulating raw diagnostic and final composition images.

### `smarteraser/runtime.py`

Continues to own model resources, prompt tokenization, and one-square-image
inference. It does not know about connected components or source coordinates.

### Shared pipeline and LaMa adapters

Remain behaviorally unchanged.

## Validation and Error Handling

- `context_scale` must be at least `1.0`.
- `minimum_context_ratio` must be greater than `0.0` and at most `1.0`.
- Empty masks continue to bypass inference.
- Every group crop must contain its complete group bounding box.
- Restored output must match the original image size.
- A mismatch between runtime output and configured resolution raises the
  existing explicit error.

## Testing

Unit coverage will verify:

- diagonal mask pixels form one 8-connected component;
- distant components form separate groups;
- components with overlapping candidate context boxes form one transitive
  group;
- tiny masks respect the minimum crop ratio;
- normal masks use `context_scale`;
- edge-adjacent crops shift inside the image without padding;
- large masks use the existing padding fallback;
- SmartEraser performs one inference per group and accumulates results;
- later groups receive the original source rather than earlier generated
  output;
- SmartEraser emits one combined diagnostic sequence;
- SmartEraser forwards the conservative guidance configuration;
- Original LaMa continues to receive and process one unchanged union mask.

## Acceptance Criteria

- Distant masked objects no longer force SmartEraser to see the whole artwork.
- Nearby mask fragments are processed together.
- SmartEraser uses the approved adaptive context and conservative guidance
  defaults.
- Pixels outside each composition mask remain identical to the source.
- The shared mask creation pipeline and both LaMa adapters remain unchanged.
- The current diagnostic filenames remain available.
- No automatic LaMa/SmartEraser routing is implemented.
