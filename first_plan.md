# Amodal Completion Integration Plan

## 1. Objective

Integrate the **Amodal Completion in the Wild / SDAmodal** model into the existing
image-to-components pipeline. It will receive a visible or fragmented object
mask and predict the object's complete, unoccluded shape.

This is a planning document only. No implementation is included in this phase.

## 2. Codebase Findings

### 2.1 Missing prerequisites

The supplied guide requires two assets that are not currently present under
`backend/models/completion/`:

1. The DIFT feature extractor referenced as `dift/extract_dift_amodal.py`.
2. A pretrained checkpoint such as `ckpt_SDAmodal.pth`.

The model cannot perform useful inference until both assets are available and
their versions are confirmed to match `config_SDAmodal.yaml`.

### 2.2 Package import conflicts

The imported research code contains top-level imports such as:

```python
import models
import utils
import inference
from models import backbone
```

Within this application, `models` can resolve to `backend.models` rather than the
completion model's internal package. These imports must eventually be converted
to explicit package-relative imports.

### 2.3 Hard-coded CUDA assumptions

The completion source uses `.cuda()` and `cuda:0` directly. This conflicts with
the application's configurable `device: auto` behavior. The production adapter
must consistently use the device selected by the model manager.

### 2.4 Disk-based feature loading

`infer_amodal_aw_sdm()` currently loads four feature files:

```text
feature/pth0/<image>.pt
feature/pth1/<image>.pt
feature/pth2/<image>.pt
feature/pth3/<image>.pt
```

This is unsuitable for an API request. The production path should extract DIFT
features once per uploaded image and pass them directly in memory to every
instance-completion call.

### 2.5 No current occlusion signal

The previous depth-order and occlusion feature was removed. The current pipeline
has no explicit signal that says which instances require completion.

A configurable completion policy is therefore needed:

```text
off       -> never run completion
all       -> run completion for every valid instance
heuristic -> run only for masks that appear fragmented or occluded
```

The recommended first policy is `all` in a non-user-visible validation mode.
After evaluating real outputs, it can be replaced with `heuristic`.

### 2.6 Shape completion is not appearance completion

SDAmodal predicts a full mask, not hidden RGB pixels. The pipeline must retain:

- `modal_mask`: visible pixels detected by SAM3.
- `amodal_mask`: predicted full object shape.
- `completion_holes`: `amodal_mask AND NOT modal_mask`.

An amodal mask must not be passed directly to BiRefNet over the original image.
Pixels in the completed area still belong to the occluder or background, so the
resulting alpha and RGB would be incorrect. A separate hidden-content
reconstruction stage will ultimately be required.

## 3. Proposed Data Flow

```text
Input image + keywords
          |
          v
      SAM3 detection
          |
          v
Visible per-instance masks
          |
          v
Safe fragment association
          |
          v
Completion policy (off / all / heuristic)
          |
          +-- completion disabled/not needed -> amodal = modal
          |
          +-- completion needed
                    |
                    v
          Extract DIFT features once
                    |
                    v
             SDAmodal inference
                    |
                    v
           Validate completed masks
                    |
                    v
 modal + amodal + completion-hole masks
          |
          +---------------------------+
          |                           |
          v                           v
Visible component path      Future hidden RGB reconstruction
(BiRefNet/refinement)        over completion holes
          |                           |
          +-------------+-------------+
                        |
                        v
                Completed RGBA layer
                        |
                        v
              Final background inpainting
```

## 4. Injection Point

The completion stage should run after SAM3 extraction:

```python
raw_masks, labels = _extract_raw_masks(image, keywords)
```

and before matting:

```python
soft_alphas = _refine_masks(image_np, raw_masks)
```

Conceptually, the orchestrator becomes:

```text
_extract_raw_masks()
        -> _complete_masks()
        -> _refine_masks()
        -> _extract_object_layers()
        -> _generate_final_background()
```

Initially, `_refine_masks()` must continue to use modal masks. Amodal masks must
travel alongside them until hidden RGB reconstruction exists.

## 5. Instance Association Before Completion

SDAmodal expects one incomplete mask per logical object. The current behavior of
merging same-keyword masks when bounding boxes overlap may combine two distinct
overlapping objects, such as two people, into one completion input.

Before completion is enabled, grouping should consider:

- Keyword equality.
- Bounding-box overlap.
- Actual mask proximity.
- Connected-component structure.
- SAM confidence.
- Area growth caused by merging.
- Whether the merged mask contains multiple large disconnected regions.

The completion model must receive one logical instance per mask.

## 6. Model Architecture Changes

### 6.1 Completion base interface

Add a completion abstraction parallel to the existing model types:

```text
BaseSegmentationModel
BaseMattingModel
BaseInpaintingModel
BaseCompletionModel
```

Conceptual completion contract:

```text
Input:
    original RGB image
    modal masks

Output:
    amodal masks in the same order
```

The adapter must hide SDAmodal, DIFT, patch extraction, and checkpoint details
from `backend/image_processor.py`.

### 6.2 Registry and configuration

Extend `ModelRegistry` with a `completion` category. The expected configuration
shape is:

```yaml
models:
  completion:
    enabled: true
    active: sdamodal

    sdamodal:
      config_path: "backend/models/completion/config_SDAmodal.yaml"
      checkpoint_path: "checkpoints/ckpt_SDAmodal.pth"
      threshold: 0.5
      input_size: 512
      enlarge_box: 3.0
      policy: "all"
      lazy_load: true
```

### 6.3 Model manager

Add:

```text
_completion_model
get_completion_model()
```

Completion should only import and load when enabled. Startup warm-up should be
optional because SDAmodal and the Stable Diffusion DIFT extractor may consume a
large amount of GPU memory.

## 7. SDAmodal Adapter Responsibilities

The application-facing adapter should:

1. Load `config_SDAmodal.yaml`.
2. Construct `AWSDM`.
3. Load the matching checkpoint.
4. Move the network to the configured device.
5. Initialize the DIFT extractor.
6. Extract one DIFT feature pyramid per source image.
7. Prepare modal masks and expanded bounding boxes.
8. Complete each instance using the shared feature pyramid.
9. Restore completed patches to full-image coordinates.
10. Validate predictions and apply safe fallbacks.
11. Return binary NumPy masks in input order.

The orchestrator should not import the research inference module directly.

## 8. Input and Output Processing

### 8.1 Source image

The current source representations are:

```text
PIL RGB
NumPy uint8 (H, W, 3)
```

DIFT must receive the original image before guided masking or inpainting. Its
required normalization and resizing must match the pretrained extractor.

### 8.2 Modal masks

Current SAM masks:

```text
shape: (H, W)
dtype: uint8
values: 0 or 255
```

SDAmodal input:

```text
shape: (N, H, W)
dtype: uint8 or float32
values: 0 or 1
```

The conversion is conceptually:

```text
modal = (raw_mask > 0).astype(uint8)
```

Empty masks must be rejected before bounding-box extraction.

### 8.3 Bounding boxes

Visible boxes use `[x, y, width, height]`. The guide expands each into a square
using `enlarge_box = 3.0`, while ensuring at least approximately 1.1 times the
original dimensions.

The adapter must safely handle negative coordinates, boxes extending beyond the
image, very small masks, empty crops, and boxes larger than the image.

### 8.4 Category values

The supplied inference example uses category value `1` for every instance. The
first integration should preserve that behavior rather than mapping SAM keywords
to class IDs.

### 8.5 DIFT features

The current code expects feature levels `0`, `1`, `2`, and `3`. Per instance,
they are resized to:

```text
level 0 -> 24 x 24
level 1 -> 48 x 48
level 2 -> 96 x 96
level 3 -> 96 x 96
```

Production inference should accept these features as an in-memory dictionary.
They should be computed once and reused across every object in the image.

### 8.6 Modal patch

The cropped modal mask is resized to `512 x 512` with nearest-neighbor
interpolation. The completion network receives a `(1, 1, 512, 512)` modal tensor
plus the aligned DIFT feature dictionary.

### 8.7 Model output

The model produces two-class logits. The current code selects the result using
channel `argmax`. Although the guide exposes a threshold of `0.5`, the provided
forward path does not directly use the threshold argument. This behavior should
be made explicit during implementation.

The predicted patch is resized into its expanded box, restored to full image
size, and returned as a binary `(N, H, W)` array.

## 9. Prediction Validation

Known visible pixels must always survive completion:

```text
amodal_mask = predicted_mask OR modal_mask
```

Additional validation should ensure:

- Output shape matches the image.
- Output is binary and non-empty.
- Modal pixels are a subset of amodal pixels.
- Area growth stays below a configurable limit.
- Bounding-box growth stays below a configurable limit.
- The completed mask remains associated with the visible component.
- Invalid results fall back to the modal mask.

Record these metrics for evaluation:

```text
modal area
amodal area
completion-hole area
area growth ratio
bbox growth ratio
```

Growth limits should be tuned from real samples instead of assumed in advance.

## 10. Internal Instance Record

Parallel lists will become error-prone once completion is added. Introduce an
internal per-instance record containing:

```text
label
keyword
modal_mask
amodal_mask
completion_holes
soft_alpha
completion_applied
completion_valid
```

This internal structure does not need to change the external API initially.

## 11. Mask Usage Rules

| Operation | Mask |
| --- | --- |
| BiRefNet on the original image | Modal mask |
| Visible RGB extraction | Modal mask |
| Existing alpha extraction | Modal mask / soft alpha |
| Hidden reconstruction region | `amodal AND NOT modal` |
| Completed layer bounding box | Amodal mask |
| Final background removal | Primarily modal-mask union |
| Completion validation | Both modal and amodal masks |

The final background should not automatically use the complete amodal union.
Amodal predictions may extend into genuine background and erase valid pixels.

## 12. Hidden RGB Reconstruction

After mask completion:

```text
modal mask       -> visible RGB is known
completion holes -> hidden RGB is unknown
```

The existing inpainting backend is configured to generate an empty background,
not reconstruct a hidden object. A future object-content reconstruction stage
will need the completed mask, visible crop, object keyword, completion-hole mask,
and a separate reconstruction model or prompt policy.

Until then, completion should run in shadow mode and should not claim to produce
complete drag-and-droppable RGB layers.

## 13. Required Research-Code Refactoring

Before model registration:

1. Replace ambiguous imports with package-relative imports.
2. Remove unused production imports such as `ipdb`, `pdb`, and `matplotlib`.
3. Remove `pycocotools` from the runtime path if unnecessary.
4. Replace `.cuda()` with `.to(device)`.
5. Remove hard-coded `cuda:0` placement.
6. Make checkpoint loading device-aware.
7. Accept DIFT features in memory instead of reading files.
8. Keep CLI scripts out of the API runtime path.
9. Avoid loading training-only modules during startup.
10. Replace deprecated `np.int` and `np.bool` aliases.
11. Use `torch.inference_mode()` for inference.

## 14. GPU and Performance Plan

The full application may hold SAM3, BiRefNet, LaMa or Stable Diffusion,
the DIFT feature extractor, and SDAmodal at the same time. This may exceed GPU
memory.

Recommended controls:

```yaml
models:
  completion:
    enabled: true
    lazy_load: true
    feature_device: cuda
    model_device: cuda
    offload_features_after_completion: true
```

Operational rules:

- Compute DIFT once per image.
- Reuse features across instances.
- Avoid feature files on disk.
- Release temporary features after completion.
- Consider a GPU inference lock for concurrent requests.
- Benchmark memory before enabling startup warm-up.
- Validate model precision before enabling mixed precision.

## 15. Completion Trigger Policy

### Phase-one: `all`

Complete every valid mask in shadow mode. This is simple and provides a baseline,
but may alter masks that were already complete and increases latency.

### Phase-two: `heuristic`

Potential trigger signals:

- Multiple disconnected fragments.
- Contact or near-contact with another mask.
- Strong mask concavity.
- A visible contour terminating against another mask.
- Overlapping boxes with separated masks.
- Suspiciously small visible area.

The heuristic decides whether completion runs; it does not infer depth order.

## 16. Testing Plan

### Unit tests

- Empty-mask rejection.
- `0/255` to `0/1` normalization.
- Bounding-box calculation and expansion.
- Out-of-image padding.
- Patch restoration.
- Visible-mask preservation.
- Invalid-result fallback.
- Area and box growth validation.
- Device propagation.
- In-memory DIFT routing.

### Adapter tests

Use fake DIFT tensors and a fake completion network to verify:

- DIFT executes once per image.
- Instances share one feature pyramid.
- Output order matches input order.
- No disk feature I/O occurs.
- Modal masks remain subsets of completed masks.

### Pipeline integration tests

- Disabled completion preserves current behavior.
- Empty detections bypass completion.
- Completion runs after segmentation.
- Modal masks still drive visible alpha extraction.
- Amodal masks do not enlarge background removal unexpectedly.
- One failed completion does not fail the request.

### Real-model acceptance set

Evaluate:

```text
simple partial object
multiple fragments
heavy occlusion
already-complete object
small object
border-touching object
overlapping same-keyword instances
```

Save modal masks, amodal predictions, and completion holes for manual review.

## 17. Implementation Phases

### Phase 0: Acquire prerequisites

- DIFT extraction source.
- SDAmodal checkpoint.
- Checkpoint/config compatibility confirmation.
- Model-author dependency specification.

### Phase 1: Standalone adapter

- Repair imports and device handling.
- Load the config and checkpoint.
- Pass DIFT features in memory.
- Verify one image and one mask outside FastAPI.

### Phase 2: Architecture integration

- Add `BaseCompletionModel`.
- Add the `completion` registry category.
- Add lazy completion lifecycle management.
- Add configuration.

### Phase 3: Orchestrator integration

- Add `_complete_masks()`.
- Insert it after SAM3 extraction.
- Preserve modal and amodal masks separately.
- Add validation and fallback behavior.

### Phase 4: Shadow mode

Run completion without changing returned layers. Record masks, growth metrics,
latency, and GPU memory usage.

### Phase 5: Hidden RGB reconstruction

Introduce a separate reconstruction stage for:

```text
amodal_mask AND NOT modal_mask
```

Only after this phase should amodal masks drive complete component layers.

### Phase 6: Heuristic activation

Replace `policy: all` with a validated completion trigger.

## 18. Approval Gate

No implementation should begin until approval is given for:

1. The dual modal/amodal mask architecture.
2. Shadow-mode integration before changing user-visible output.
3. A registered `completion` model category.
4. Refactoring completion imports, device placement, and DIFT handling.
5. Supplying the missing DIFT source and pretrained checkpoint.
