# SmartEraser Local-Context Inpainting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce SmartEraser hallucinations on Canva artwork by grouping nearby mask components, inpainting each group from adaptive local context, and using conservative guidance.

**Architecture:** A new pure `regions.py` module extracts and groups 8-connected mask components. `geometry.py` transforms one explicitly supplied group crop to and from 512-by-512 inference space, while the SmartEraser adapter independently runs and composes every group; shared mask creation and both LaMa adapters remain unchanged.

**Tech Stack:** Python 3.12, Pillow, NumPy, SciPy `ndimage.label`, PyTorch, pytest, YAML.

## Global Constraints

- Work directly on `feature/vlm`; do not use the old integration worktree.
- Preserve the user's existing uncommitted `active: smarteraser` configuration.
- Apply connected-components, grouping, and adaptive cropping only inside the SmartEraser package.
- Do not modify `backend/pipeline/background.py`, Original LaMa, SimpleLaMa, model weights, or the `ckpts` directory.
- Do not implement automatic LaMa/SmartEraser routing.
- Keep `resolution: 512`, `num_inference_steps: 50`, `seed: 42`, dtype, mask expansion, feathering, diagnostics filenames, and CLIP loading unchanged.
- Change only SmartEraser's guidance defaults to `guidance_scale: 1.2`, `prompt: "Remove the instance of"`, and `negative_prompt: "objects, text, decorations, artifacts"`.
- Do not commit or push unless the user separately approves the Git action.

---

## File Map

- Create `backend/models/background_inpainting/smarteraser/regions.py`: connected-component extraction, adaptive context boxes, and transitive grouping.
- Modify `backend/models/background_inpainting/smarteraser/geometry.py`: prepare and restore one explicit local crop or the existing padding fallback.
- Modify `backend/models/background_inpainting/smarteraser/adapter.py`: validate context settings, run groups independently, and accumulate diagnostics/composition.
- Modify `backend/config.yaml`: add context settings and conservative SmartEraser guidance while preserving `active: smarteraser`.
- Create `tests/test_smarteraser_regions.py`: pure grouping and crop-box behavior.
- Modify `tests/test_smarteraser_geometry.py`: local crop/restore and padding behavior.
- Modify `tests/test_smarteraser_adapter.py`: multi-group orchestration, source independence, diagnostics, and config forwarding.
- Modify `tests/test_background_inpainting_packages.py`: assert SmartEraser context/guidance values without changing model-selection policy.
- Modify `tests/test_original_lama_adapter.py`: regression proof that a disconnected union mask still causes one LaMa inference call.

### Task 1: Pure Mask Regions and Adaptive Context

**Files:**
- Create: `backend/models/background_inpainting/smarteraser/regions.py`
- Create: `tests/test_smarteraser_regions.py`

**Interfaces:**
- Consumes: a binary-compatible `PIL.Image.Image`, `context_scale: float`, and `minimum_context_ratio: float`.
- Produces:

```python
BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class MaskGroup:
    mask: Image.Image
    bounding_box: BoundingBox


def adaptive_context_box(
    bounding_box: BoundingBox,
    image_size: tuple[int, int],
    context_scale: float,
    minimum_context_ratio: float,
) -> BoundingBox | None: ...


def group_mask_components(
    mask: Image.Image,
    context_scale: float,
    minimum_context_ratio: float,
) -> list[MaskGroup]: ...
```

`None` from `adaptive_context_box()` means the bounding box cannot fit inside a square no larger than the image's shorter side and geometry must use the full-image padding fallback.

- [ ] **Step 1: Write failing tests for validation and adaptive crop sizing**

Create tests with these exact cases:

```python
def test_tiny_component_uses_minimum_context_ratio():
    box = adaptive_context_box(
        (100, 100, 120, 120),
        image_size=(1000, 600),
        context_scale=3.0,
        minimum_context_ratio=0.25,
    )
    assert box is not None
    assert box[2] - box[0] == 150
    assert box[3] - box[1] == 150


def test_normal_component_uses_context_scale():
    box = adaptive_context_box(
        (300, 200, 400, 280),
        image_size=(1000, 600),
        context_scale=3.0,
        minimum_context_ratio=0.25,
    )
    assert box is not None
    assert box[2] - box[0] == 300


def test_edge_component_shifts_crop_inside_image():
    box = adaptive_context_box(
        (0, 10, 100, 90),
        image_size=(1000, 600),
        context_scale=3.0,
        minimum_context_ratio=0.25,
    )
    assert box == (0, 0, 300, 300)


def test_component_wider_than_short_side_uses_padding_fallback():
    assert adaptive_context_box(
        (50, 100, 750, 200),
        image_size=(800, 400),
        context_scale=3.0,
        minimum_context_ratio=0.25,
    ) is None
```

Parametrize invalid `context_scale=0.99` and invalid minimum ratios `0.0` and `1.01`, asserting `ValueError` names the invalid field.

- [ ] **Step 2: Run crop-box tests and verify RED**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_smarteraser_regions.py -k "context or edge or padding" -v
```

Expected: import failure because `smarteraser.regions` does not exist.

- [ ] **Step 3: Implement immutable types, validation, and crop-box calculation**

Implement:

```python
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import label


BoundingBox = tuple[int, int, int, int]
_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class MaskGroup:
    mask: Image.Image
    bounding_box: BoundingBox


def _validate_context(
    context_scale: float,
    minimum_context_ratio: float,
) -> None:
    if context_scale < 1.0:
        raise ValueError("SmartEraser context_scale must be at least 1.0")
    if not 0.0 < minimum_context_ratio <= 1.0:
        raise ValueError(
            "SmartEraser minimum_context_ratio must be in (0.0, 1.0]"
        )
```

In `adaptive_context_box()`, use integer `ceil()` for the requested side,
clamp it to `short_side`, center it on the bounding-box center, then clamp
`left` and `top` to `[0, width - side]` and `[0, height - side]`. Return
`None` before sizing if either bounding-box dimension exceeds `short_side`.

- [ ] **Step 4: Write failing connected-component and grouping tests**

Cover:

```python
def test_diagonal_pixels_form_one_eight_connected_group():
    mask = Image.new("L", (10, 10), 0)
    mask.putpixel((3, 3), 255)
    mask.putpixel((4, 4), 255)
    groups = group_mask_components(mask, 3.0, 0.25)
    assert len(groups) == 1


def test_distant_components_form_separate_groups():
    mask = Image.new("L", (100, 100), 0)
    mask.putpixel((10, 10), 255)
    mask.putpixel((90, 90), 255)
    groups = group_mask_components(mask, 3.0, 0.1)
    assert [group.bounding_box for group in groups] == [
        (10, 10, 11, 11),
        (90, 90, 91, 91),
    ]


def test_overlapping_context_boxes_merge_transitively():
    mask = Image.new("L", (120, 40), 0)
    for x in (20, 32, 44):
        mask.paste(255, (x, 15, x + 5, 20))
    groups = group_mask_components(mask, 3.0, 0.1)
    assert len(groups) == 1
    assert groups[0].bounding_box == (20, 15, 49, 20)
```

Also assert an empty mask returns `[]` and each returned group mask contains
only its member components.

- [ ] **Step 5: Run grouping tests and verify RED**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_smarteraser_regions.py -k "group or component or empty" -v
```

Expected: failures because component extraction and grouping are incomplete.

- [ ] **Step 6: Implement 8-connected extraction and transitive grouping**

Implementation requirements:

1. Convert the input to `L`, threshold at 127, and call:

```python
labels, component_count = label(
    np.asarray(binary_mask, dtype=np.uint8) > 0,
    structure=_EIGHT_CONNECTED,
)
```

2. For each label ID, build an isolated `uint8` mask, derive its PIL
   `getbbox()`, and calculate its candidate context box. Treat a `None`
   candidate as the full image box for grouping.
3. Use union-find. Two half-open boxes intersect or touch when:

```python
not (
    first[2] < second[0]
    or second[2] < first[0]
    or first[3] < second[1]
    or second[3] < first[1]
)
```

4. Union every intersecting pair. This naturally provides transitive grouping.
5. OR all component arrays belonging to one root, convert to `L`, and compute
   the union bounding box.
6. Return groups sorted by `(top, left, bottom, right)` for deterministic
   inference order.

- [ ] **Step 7: Run all region tests and verify GREEN**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_smarteraser_regions.py -v
```

Expected: all region tests pass.

### Task 2: Explicit Local-Crop Geometry

**Files:**
- Modify: `backend/models/background_inpainting/smarteraser/geometry.py`
- Modify: `tests/test_smarteraser_geometry.py`

**Interfaces:**
- Consumes the Task 1 `BoundingBox | None`.
- Produces:

```python
def prepare_inputs(
    image: Image.Image,
    mask: Image.Image,
    resolution: int,
    crop_box: BoundingBox | None,
) -> PreparedInputs: ...
```

When `crop_box` is present, it is in original-image coordinates. When it is
`None`, geometry uses the existing full-image padding path.

- [ ] **Step 1: Replace implicit-crop expectations with explicit local-crop tests**

Add a test using a `1000×600` coordinate image, a mask at
`(300, 200, 400, 280)`, and crop box `(200, 90, 500, 390)`. Assert:

```python
assert prepared.image.size == (512, 512)
assert prepared.mask.size == (512, 512)
assert prepared.metadata.mode == "crop"
assert prepared.metadata.crop_box == (200, 90, 500, 390)
```

Generate a solid red `512×512` output, restore it, and assert:

- result size is `1000×600`;
- pixel `(250, 150)` inside the crop is red;
- pixel `(0, 0)` outside the crop remains the original color.

Add failures for a crop outside image bounds, a non-square crop, and a crop
that does not fully contain the mask bounding box.

- [ ] **Step 2: Run local-crop geometry tests and verify RED**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_smarteraser_geometry.py -k "local_crop or crop_box" -v
```

Expected: `TypeError` or assertion failures because `prepare_inputs()` does
not accept an explicit crop box and metadata uses scaled-image coordinates.

- [ ] **Step 3: Refactor crop metadata and implementation**

Keep `TransformMode = Literal["crop", "padding"]`. For crop mode:

```python
cropped_image = image.crop(crop_box)
cropped_mask = mask.crop(crop_box)
prepared_image = cropped_image.resize(
    (resolution, resolution),
    Image.Resampling.BILINEAR,
)
prepared_mask = cropped_mask.resize(
    (resolution, resolution),
    Image.Resampling.NEAREST,
)
```

Store `crop_box` in original-image coordinates. `scaled_size` remains required
only by padding metadata and becomes optional:

```python
scaled_size: tuple[int, int] | None = None
```

For crop restoration:

```python
restored = original.convert("RGB").copy()
left, top, right, bottom = metadata.crop_box
restored.paste(
    generated_rgb.resize(
        (right - left, bottom - top),
        Image.Resampling.BILINEAR,
    ),
    (left, top),
)
return restored
```

Keep padding preparation/restoration behavior unchanged.

- [ ] **Step 4: Update existing geometry tests to pass explicit crop boxes**

For tests that exercise crop mode, compute or supply a square original-space
box. For the wide-mask test, pass `crop_box=None` and retain the assertion
that padding restores original size. Keep empty mask, invalid resolution,
guidance crop, and wrong generated-size coverage.

- [ ] **Step 5: Run geometry tests and verify GREEN**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_smarteraser_geometry.py -v
```

Expected: all geometry tests pass.

### Task 3: Per-Group SmartEraser Orchestration

**Files:**
- Modify: `backend/models/background_inpainting/smarteraser/adapter.py`
- Modify: `tests/test_smarteraser_adapter.py`

**Interfaces:**
- Consumes `group_mask_components()` and `adaptive_context_box()` from Task 1.
- Consumes explicit-crop `prepare_inputs()` from Task 2.
- Produces one final RGB image and at most two callbacks:
  `after_smarteraser`, then `after_composition_blend`.

- [ ] **Step 1: Update the adapter fixture and write failing validation tests**

Set:

```python
"clip_model_id": "openai/clip-vit-large-patch14",
"clip_auto_download": True,
"context_scale": 3.0,
"minimum_context_ratio": 0.25,
"guidance_scale": 1.2,
"prompt": "Remove the instance of",
"negative_prompt": "objects, text, decorations, artifacts",
```

Parametrize invalid `context_scale` and `minimum_context_ratio`, instantiate
the adapter, and assert it raises before any runtime inference.

- [ ] **Step 2: Write a failing two-group orchestration test**

Use a `100×100` source with two single-component masks at `(15, 15)` and
`(85, 85)`, `context_scale=3.0`, and `minimum_context_ratio=0.1`.

Make `FakeRuntime.inpaint()`:

- record a copy of each image and mask input;
- return a red square on the first call and a blue square on the second.

Assert:

```python
assert len(inpaint_calls) == 2
assert all(call.image.size == (8, 8) for call in inpaint_calls)
assert result.getpixel((15, 15)) == (255, 0, 0)
assert result.getpixel((85, 85)) == (0, 0, 255)
assert result.getpixel((50, 50)) == source.getpixel((50, 50))
```

Wrap or monkeypatch `prepare_inputs()` to record its `image` argument before
delegating to the real function. Assert every recorded image array equals the
original source array, proving the first generated result was not fed into the
second call.

- [ ] **Step 3: Run adapter grouping tests and verify RED**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_smarteraser_adapter.py -k "group or context" -v
```

Expected: failures because the adapter still performs one whole-mask inference.

- [ ] **Step 4: Implement context configuration and per-group processing**

In `_load_model()`:

```python
self.context_scale = float(self.config.get("context_scale", 3.0))
self.minimum_context_ratio = float(
    self.config.get("minimum_context_ratio", 0.25)
)
```

Validate using the Task 1 helper before creating the runtime.

Replace the single-inference body after the empty-mask guard with:

```python
groups = group_mask_components(
    binary_mask,
    context_scale=self.context_scale,
    minimum_context_ratio=self.minimum_context_ratio,
)
raw_result = source.copy()
composed = source.copy()

for group in groups:
    generation_mask, blend_mask = prepare_inpaint_masks(
        group.mask,
        generation_expansion=self.generation_mask_expansion,
        composition_expansion=self.composition_mask_expansion,
        feather_radius=self.feather_radius,
    )
    generation_box = generation_mask.getbbox()
    if generation_box is None:
        continue
    crop_box = adaptive_context_box(
        generation_box,
        source.size,
        self.context_scale,
        self.minimum_context_ratio,
    )
    prepared = prepare_inputs(
        source,
        generation_mask,
        self.resolution,
        crop_box=crop_box,
    )
    generated_square = self.runtime.inpaint(
        prepared.image,
        prepared.mask,
    )
    generated = restore_output(
        generated_square,
        source,
        prepared.metadata,
    )
    raw_result = preserve_unmasked_pixels(
        raw_result,
        generated,
        generation_mask,
    )
    composed = preserve_unmasked_pixels(
        composed,
        generated,
        blend_mask,
    )
```

Emit diagnostics once, after the loop:

```python
if artifact_callback is not None:
    artifact_callback("after_smarteraser", raw_result)
    artifact_callback("after_composition_blend", composed)
```

- [ ] **Step 5: Update existing adapter tests**

Keep coverage for:

- empty mask bypass;
- pixels outside composition masks;
- runtime config forwarding;
- diagnostic stage order;
- unload;
- invalid resolution.

Change the expected SmartEraser diagnostic stage from any stale
`after_lama` expectation to `after_smarteraser`. Do not modify LaMa diagnostic
expectations.

- [ ] **Step 6: Run adapter and region/geometry integration tests**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_smarteraser_regions.py tests/test_smarteraser_geometry.py tests/test_smarteraser_adapter.py -v
```

Expected: all SmartEraser local-context tests pass.

### Task 4: Conservative Defaults and LaMa Isolation

**Files:**
- Modify: `backend/config.yaml`
- Modify: `tests/test_background_inpainting_packages.py`
- Modify: `tests/test_original_lama_adapter.py`

**Interfaces:**
- SmartEraser config adds `context_scale: float` and
  `minimum_context_ratio: float`.
- LaMa public adapter interface and call count remain unchanged.

- [ ] **Step 1: Write failing config assertions**

Read `backend/config.yaml` and assert:

```python
smart = config["models"]["background_inpainting"]["smarteraser"]
assert smart["context_scale"] == 3.0
assert smart["minimum_context_ratio"] == 0.25
assert smart["guidance_scale"] == 1.2
assert smart["prompt"] == "Remove the instance of"
assert smart["negative_prompt"] == (
    "objects, text, decorations, artifacts"
)
```

Do not assert or change `background_inpainting.active`; preserve the user's
current local value.

- [ ] **Step 2: Write the LaMa disconnected-mask regression test**

In `tests/test_original_lama_adapter.py`, create a mask with two distant
pixels, run the adapter once, and assert:

```python
assert len(runtime_masks) == 1
assert runtime_masks[0][1, 1] > 0
assert runtime_masks[0][-2, -2] > 0
```

This proves component splitting was not introduced into LaMa.

- [ ] **Step 3: Run config and LaMa tests and verify RED/GREEN boundary**

Run before YAML changes:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_background_inpainting_packages.py -k smarteraser_config tests/test_original_lama_adapter.py -v
```

Expected: the new SmartEraser config assertions fail; the LaMa regression
already passes because LaMa remains unchanged.

- [ ] **Step 4: Update only the SmartEraser YAML block**

Preserve `active: smarteraser` and set:

```yaml
context_scale: 3.0
minimum_context_ratio: 0.25
num_inference_steps: 50
guidance_scale: 1.2
seed: 42
prompt: "Remove the instance of"
negative_prompt: "objects, text, decorations, artifacts"
```

Do not change any LaMa keys.

- [ ] **Step 5: Run focused SmartEraser and LaMa regression**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_smarteraser_regions.py tests/test_smarteraser_geometry.py tests/test_smarteraser_adapter.py tests/test_smarteraser_runtime.py tests/test_background_inpainting_packages.py tests/test_original_lama_adapter.py tests/test_original_lama_runtime.py -v
```

Expected: all focused tests pass.

### Task 5: Final Verification and Handoff

**Files:**
- Verify only; no production files added in this task.

**Interfaces:**
- Confirms the acceptance criteria across SmartEraser and LaMa.

- [ ] **Step 1: Confirm shared and LaMa production files are untouched**

Run:

```powershell
git diff --name-only
git diff -- backend/pipeline/background.py backend/models/background_inpainting/original_lama backend/models/background_inpainting/simple_lama
```

Expected: the second command has no output. The name list contains only the
SmartEraser package, its tests, YAML config, and planning/spec documents.

- [ ] **Step 2: Run diff and syntax checks**

Run:

```powershell
git diff --check
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m compileall -q backend/models/background_inpainting/smarteraser
```

Expected: both commands exit zero.

- [ ] **Step 3: Run the full suite and record the exact baseline**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest -q
```

Expected: no SmartEraser or LaMa failures. If the repository's known unrelated
completion, grouping, diagnostics, or reconstruction tests still fail, report
their exact names and counts rather than modifying them in this task.

- [ ] **Step 4: Inspect final status without committing**

Run:

```powershell
git status --short --branch
git diff --stat
```

Confirm:

- no files under `SmartEraser/Model_framework/ckpts` were changed;
- the `SmartEraser` submodule's pre-existing untracked server weights were not
  staged or removed;
- no commit or push was performed.

## Implementation Refinement

The implemented region representation is intentionally more memory-efficient
than the illustrative Task 1 pseudocode:

- `scipy.ndimage.find_objects()` derives component boxes from one shared label
  map;
- `MaskGroup.mask` stores only the pixels inside `MaskGroup.bounding_box`;
- `MaskGroup.to_image(image_size)` materializes a full-size mask only when the
  adapter processes that final group;
- candidate context boxes are indexed in a bounded uniform spatial grid;
- each grid cell tracks union-find roots and their member boxes, allowing dense
  overlapping fragments to connect through one root without unconditional
  all-pairs comparisons.

These refinements preserve the documented grouping behavior and public adapter
flow while avoiding memory proportional to
`component_count * image_width * image_height`.
