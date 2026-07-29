# SmartEraser Background Inpainting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SmartEraser as a locally loaded, configuration-selectable background inpainter without changing the default Original LaMa behavior.

**Architecture:** A typed geometry module performs deterministic PIL preprocessing and restoration, a runtime module owns heavyweight SmartEraser resources and inference, and a thin registered adapter implements the project's background-inpainting interface. The existing registry and `models.background_inpainting.active` setting remain the only orchestration mechanism.

**Tech Stack:** Python 3.10+, Pillow, NumPy, PyTorch, Diffusers, Transformers, PyYAML, pytest.

## Global Constraints

- Do not modify the structure or contents of any `ckpts` directory.
- Do not add model-download behavior or network fallback.
- Keep `models.background_inpainting.active: original_lama`.
- Preserve existing Original LaMa behavior and tests.
- Apply Single Responsibility Principle boundaries and explicit type hints.
- Write each production behavior only after its failing test has been observed.

## File Map

- Create `backend/models/background_inpainting/smarteraser/__init__.py`: package export.
- Create `backend/models/background_inpainting/smarteraser/geometry.py`: image/mask preprocessing, guidance crop, and restoration.
- Create `backend/models/background_inpainting/smarteraser/runtime.py`: local resource validation, model loading, inference, and cleanup.
- Create `backend/models/background_inpainting/smarteraser/adapter.py`: registry adapter and shared-mask composition.
- Modify `backend/models/manager.py`: import the strategy package for registration.
- Modify `backend/config.yaml`: add SmartEraser configuration while retaining Original LaMa as active.
- Modify `tests/test_background_inpainting_packages.py`: assert package/config registration and unchanged default.
- Create `tests/test_smarteraser_geometry.py`: unit tests for pure transformations.
- Create `tests/test_smarteraser_runtime.py`: unit tests for runtime validation and inference wiring.
- Create `tests/test_smarteraser_adapter.py`: unit tests for interface behavior, composition, callbacks, and unload.

---

### Task 1: Deterministic SmartEraser Geometry

**Files:**
- Create: `backend/models/background_inpainting/smarteraser/geometry.py`
- Create: `tests/test_smarteraser_geometry.py`

**Interfaces:**
- Produces:
  - `TransformMode = Literal["crop", "padding"]`
  - `TransformMetadata(mode, original_size, scaled_size, crop_box, padding)`
  - `PreparedInputs(image, mask, metadata)`
  - `prepare_inputs(image: Image.Image, mask: Image.Image, resolution: int) -> PreparedInputs`
  - `build_guidance_crop(image: Image.Image, mask: Image.Image) -> Image.Image`
  - `restore_output(generated: Image.Image, original: Image.Image, metadata: TransformMetadata) -> Image.Image`

- [ ] **Step 1: Write failing normalization and empty-mask tests**

```python
import numpy as np
import pytest
from PIL import Image


def test_prepare_inputs_aligns_and_binarizes_mask():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
    )

    image = Image.new("RGBA", (20, 10), (10, 20, 30, 255))
    mask = Image.fromarray(np.array([[0, 200]], dtype=np.uint8), mode="L")
    prepared = prepare_inputs(image, mask, resolution=8)

    assert prepared.image.mode == "RGB"
    assert prepared.image.size == (8, 8)
    assert prepared.mask.size == (8, 8)
    assert set(np.unique(np.asarray(prepared.mask))) <= {0, 255}


def test_prepare_inputs_rejects_empty_mask():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
    )

    with pytest.raises(ValueError, match="empty"):
        prepare_inputs(
            Image.new("RGB", (8, 8)),
            Image.new("L", (8, 8), 0),
            resolution=8,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
pytest tests/test_smarteraser_geometry.py -q
```

Expected: collection fails with `ModuleNotFoundError` for the SmartEraser geometry module.

- [ ] **Step 3: Implement typed metadata and input preparation**

Implement frozen dataclasses, positive-resolution validation, RGB conversion,
nearest-neighbor mask alignment and thresholding, bounding-box measurement,
and the upstream crop-versus-padding rule. Use bilinear image resizing and
nearest-neighbor mask resizing.

Crop mode scales the shorter image edge to `resolution`, centers the square
crop around the mask bounding box, and clamps the crop within the scaled
image. Padding mode scales the longer edge to `resolution`, centers it on a
white RGB canvas, and centers the mask on a black grayscale canvas.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
pytest tests/test_smarteraser_geometry.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: Add failing crop, padding, guidance, and restoration tests**

Add tests that assert:

```python
def test_small_mask_uses_crop_and_restores_original_size():
    image = Image.new("RGB", (12, 8), "blue")
    mask = Image.new("L", image.size, 0)
    mask.putpixel((6, 4), 255)
    prepared = prepare_inputs(image, mask, resolution=8)
    restored = restore_output(
        Image.new("RGB", (8, 8), "red"), image, prepared.metadata
    )
    assert prepared.metadata.mode == "crop"
    assert restored.size == image.size
    assert restored.mode == "RGB"


def test_wide_mask_uses_padding_and_restores_original_size():
    image = Image.new("RGB", (12, 8), "blue")
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (1, 2, 11, 6))
    prepared = prepare_inputs(image, mask, resolution=8)
    restored = restore_output(
        Image.new("RGB", (8, 8), "red"), image, prepared.metadata
    )
    assert prepared.metadata.mode == "padding"
    assert restored.size == image.size


def test_guidance_crop_keeps_masked_object_on_white():
    image = Image.new("RGB", (5, 5), "blue")
    mask = Image.new("L", image.size, 0)
    mask.putpixel((2, 2), 255)
    guidance = build_guidance_crop(image, mask)
    assert guidance.size == (1, 1)
    assert guidance.getpixel((0, 0)) == (0, 0, 255)
```

- [ ] **Step 6: Run the new tests and verify RED**

Run:

```powershell
pytest tests/test_smarteraser_geometry.py -q
```

Expected: failures show missing `restore_output` and
`build_guidance_crop` behavior.

- [ ] **Step 7: Implement guidance construction and inverse transforms**

For padding, remove the recorded borders then resize to the original size. For
crop, paste the generated square at the recorded crop box on a scaled copy of
the original, then resize the full canvas to the original size. Always return
RGB.

- [ ] **Step 8: Run geometry tests and commit**

Run:

```powershell
pytest tests/test_smarteraser_geometry.py -q
git add backend/models/background_inpainting/smarteraser/geometry.py tests/test_smarteraser_geometry.py
git commit -m "feat: add SmartEraser image geometry"
```

Expected: all geometry tests pass.

---

### Task 2: Local-Only SmartEraser Runtime

**Files:**
- Create: `backend/models/background_inpainting/smarteraser/runtime.py`
- Create: `tests/test_smarteraser_runtime.py`

**Interfaces:**
- Consumes:
  - `build_guidance_crop(image, mask) -> Image.Image`
- Produces:
  - `SmartEraserRuntime(checkpoint_dir: str | Path, clip_dir: str | Path, device: str, dtype: str, num_inference_steps: int, guidance_scale: float, seed: int, prompt: str, negative_prompt: str)`
  - `inpaint(image: Image.Image, mask: Image.Image) -> Image.Image`
  - `close() -> None`

- [ ] **Step 1: Write failing local-path validation tests**

```python
from pathlib import Path
import pytest


def test_runtime_rejects_missing_local_checkpoint(tmp_path):
    from backend.models.background_inpainting.smarteraser.runtime import (
        SmartEraserRuntime,
    )

    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        SmartEraserRuntime(
            checkpoint_dir=tmp_path / "missing",
            clip_dir=clip_dir,
            device="cpu",
            dtype="float16",
            num_inference_steps=1,
            guidance_scale=1.5,
            seed=42,
            prompt="Remove the instance of object",
            negative_prompt="",
        )


def test_runtime_rejects_missing_clip_mlp_weight(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    clip_dir = tmp_path / "clip"
    checkpoint_dir.mkdir()
    clip_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="clip_mlp_weight"):
        SmartEraserRuntime(
            checkpoint_dir=checkpoint_dir,
            clip_dir=clip_dir,
            device="cpu",
            dtype="float32",
            num_inference_steps=1,
            guidance_scale=1.5,
            seed=42,
            prompt="Remove the instance of object",
            negative_prompt="",
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
pytest tests/test_smarteraser_runtime.py -q
```

Expected: import failure because `runtime.py` does not exist.

- [ ] **Step 3: Implement configuration and path validation**

Resolve relative paths against `Path(__file__).resolve().parents[4]`. Validate
the checkpoint directory, CLIP directory, and
`checkpoint_dir / "clip_mlp_weight.pth"` before importing or invoking
heavyweight loaders. Validate positive step count and guidance scale and allow
only `float16`, `bfloat16`, and `float32`. Map CPU to float32.

- [ ] **Step 4: Run path tests and verify GREEN**

Run:

```powershell
pytest tests/test_smarteraser_runtime.py -q
```

Expected: both validation tests pass without loading a real model.

- [ ] **Step 5: Add a failing inference-wiring test with fake loaders**

Use monkeypatch to replace:

- `StableDiffusionInpaintRegionPipeline.from_pretrained`;
- `CLIPVisualPrompt`;
- `CLIPImageProcessor.from_pretrained`;
- `CLIPTokenizer.from_pretrained`.

The fakes record the local paths, dtype, device, prompt IDs, CLIP tensor,
generator, image, mask, `num_inference_steps`, and `guidance_scale`. Assert
that CPU forces float32, `inpaint` returns the first pipeline image as RGB,
and no model ID or download argument is substituted.

- [ ] **Step 6: Run the inference-wiring test and verify RED**

Run:

```powershell
pytest tests/test_smarteraser_runtime.py -q
```

Expected: failure because model loading and `inpaint` are not implemented.

- [ ] **Step 7: Implement model loading, visual tokens, inference, and close**

Import:

```python
from SmartEraser.Model_framework.modules.clip_visual_token import (
    CLIPVisualPrompt,
)
from SmartEraser.Model_framework.modules.pipeline.pipeline_stable_diffusion_inpaint_region import (
    StableDiffusionInpaintRegionPipeline,
)
```

Load only the supplied local paths. Move the pipeline and three CLIP modules
to the configured device/dtype and set them to eval. Tokenize the configured
positive and negative prompts with `max_length=7`. Build prompt embeddings
with `inference_vtoken`, call the pipeline with a device-local seeded
`torch.Generator`, and return `.images[0].convert("RGB")`.

`close` sets all heavyweight references to `None`; it does not clear CUDA
caches because `ModelManager` owns accelerator cleanup.

- [ ] **Step 8: Run runtime tests and commit**

Run:

```powershell
pytest tests/test_smarteraser_runtime.py -q
git add backend/models/background_inpainting/smarteraser/runtime.py tests/test_smarteraser_runtime.py
git commit -m "feat: add local SmartEraser runtime"
```

Expected: all runtime tests pass.

---

### Task 3: Registered Background-Inpainting Adapter

**Files:**
- Create: `backend/models/background_inpainting/smarteraser/__init__.py`
- Create: `backend/models/background_inpainting/smarteraser/adapter.py`
- Create: `tests/test_smarteraser_adapter.py`

**Interfaces:**
- Consumes:
  - `prepare_inputs(...) -> PreparedInputs`
  - `restore_output(...) -> Image.Image`
  - `SmartEraserRuntime.inpaint(...) -> Image.Image`
- Produces:
  - `SmartEraserBackgroundInpaintingModel.process(...) -> Image.Image`
  - registry entry `("background_inpainting", "smarteraser")`

- [ ] **Step 1: Write failing empty-mask and composition tests**

```python
import numpy as np
from PIL import Image


def _config():
    return {
        "checkpoint_dir": "checkpoint",
        "clip_dir": "clip",
        "resolution": 8,
        "num_inference_steps": 2,
        "guidance_scale": 1.5,
        "seed": 42,
        "dtype": "float16",
        "prompt": "Remove the instance of object",
        "negative_prompt": "",
        "generation_mask_expansion": 1,
        "composition_mask_expansion": 1,
        "feather_radius": 0.0,
    }


def test_adapter_skips_runtime_for_empty_mask(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "SmartEraserRuntime",
        lambda **kwargs: type(
            "Runtime",
            (),
            {
                "inpaint": lambda self, image, mask: calls.append((image, mask)),
                "close": lambda self: None,
            },
        )(),
    )
    model = adapter.SmartEraserBackgroundInpaintingModel(_config(), "cpu")
    source = Image.new("RGB", (8, 8), "white")
    result = model.process(source, Image.new("L", source.size, 0))
    assert np.array_equal(np.asarray(result), np.asarray(source))
    assert calls == []


def test_adapter_preserves_pixels_outside_mask(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    class FakeRuntime:
        def __init__(self, **kwargs):
            pass
        def inpaint(self, image, mask):
            return Image.new("RGB", image.size, "black")
        def close(self):
            pass

    monkeypatch.setattr(adapter, "SmartEraserRuntime", FakeRuntime)
    model = adapter.SmartEraserBackgroundInpaintingModel(_config(), "cpu")
    source = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", source.size, 0)
    mask.putpixel((4, 4), 255)
    result = np.asarray(model.process(source, mask))
    assert np.all(result[0, 0] == 255)
    assert np.all(result[4, 4] == 0)
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```powershell
pytest tests/test_smarteraser_adapter.py -q
```

Expected: import failure because the adapter does not exist.

- [ ] **Step 3: Implement adapter initialization and process**

Register with:

```python
@ModelRegistry.register("background_inpainting", "smarteraser")
class SmartEraserBackgroundInpaintingModel(
    BaseBackgroundInpaintingModel
):
    ...
```

Parse typed configuration in `_load_model`, instantiate the runtime once, and
use `prepare_inpaint_masks`, `prepare_inputs`, `restore_output`, and
`preserve_unmasked_pixels`. Return early for empty masks. Keep the base
signature including `prompt` and `artifact_callback`.

- [ ] **Step 4: Run focused adapter tests and verify GREEN**

Run:

```powershell
pytest tests/test_smarteraser_adapter.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: Add failing callback, config-forwarding, and unload tests**

Assert that:

- constructor values reach `SmartEraserRuntime` unchanged;
- callback stages are `after_lama` and `after_composition_blend` for diagnostic
  compatibility;
- `unload()` calls runtime `close()` exactly once and leaves an empty
  `__dict__`.

- [ ] **Step 6: Run the new adapter tests and verify RED**

Run:

```powershell
pytest tests/test_smarteraser_adapter.py -q
```

Expected: failures identify missing callback and unload behavior.

- [ ] **Step 7: Implement callbacks, package export, and cleanup**

Export `SmartEraserBackgroundInpaintingModel` from `__init__.py`. Invoke the
raw callback after restored generation and the composition callback after
source-preserving blending. Close the runtime before `super().unload()`.

- [ ] **Step 8: Run adapter tests and commit**

Run:

```powershell
pytest tests/test_smarteraser_adapter.py -q
git add backend/models/background_inpainting/smarteraser tests/test_smarteraser_adapter.py
git commit -m "feat: add SmartEraser background adapter"
```

Expected: all adapter tests pass.

---

### Task 4: Configuration, Registry Wiring, and LaMa Regression

**Files:**
- Modify: `backend/models/manager.py`
- Modify: `backend/config.yaml`
- Modify: `tests/test_background_inpainting_packages.py`

**Interfaces:**
- Consumes:
  - registry entry `smarteraser`
  - existing `ConfigManager.get_model_config("background_inpainting")`
- Produces:
  - selectable config block with `original_lama` still active
  - manager import that registers SmartEraser

- [ ] **Step 1: Add failing package and config tests**

Extend `tests/test_background_inpainting_packages.py`:

```python
def test_smarteraser_is_a_registered_background_strategy():
    from backend.models.background_inpainting.smarteraser import (
        SmartEraserBackgroundInpaintingModel,
    )
    from backend.models.registry import ModelRegistry

    assert (
        ModelRegistry.get_class("background_inpainting", "smarteraser")
        is SmartEraserBackgroundInpaintingModel
    )


def test_smarteraser_config_is_local_only_and_lama_stays_default():
    with (ROOT / "backend" / "config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    background = config["models"]["background_inpainting"]
    smart = background["smarteraser"]
    assert background["active"] == "original_lama"
    assert smart["checkpoint_dir"] == (
        "SmartEraser/Model_framework/ckpts/smarteraser-weights"
    )
    assert smart["clip_dir"] == (
        "SmartEraser/Model_framework/ckpts/clip-vit-large-patch14"
    )
    assert "model_id" not in smart
    assert "download" not in smart
```

- [ ] **Step 2: Run package/config tests and verify RED**

Run:

```powershell
pytest tests/test_background_inpainting_packages.py -q
```

Expected: config test fails because the SmartEraser block is absent.

- [ ] **Step 3: Add manager registration import and config**

Change the manager import to include `smarteraser`. Add the exact SmartEraser
block from the approved design with expansion kernels `1` and `1`. Do not
change the active strategy or any existing strategy values.

- [ ] **Step 4: Run SmartEraser and package tests**

Run:

```powershell
pytest tests/test_smarteraser_geometry.py tests/test_smarteraser_runtime.py tests/test_smarteraser_adapter.py tests/test_background_inpainting_packages.py -q
```

Expected: all SmartEraser and package tests pass.

- [ ] **Step 5: Run targeted Original LaMa and lifecycle regression**

Run:

```powershell
pytest tests/test_original_lama_adapter.py tests/test_original_lama_runtime.py tests/test_model_lifecycle.py tests/test_pipeline_architecture.py -q
```

Expected: all targeted existing tests pass with Original LaMa still active.

- [ ] **Step 6: Run static and full-suite verification**

Run:

```powershell
python -m compileall -q backend/models/background_inpainting/smarteraser
pytest -q
git diff --check
```

Expected: compilation exits 0, the full pytest suite has zero failures, and
`git diff --check` reports no whitespace errors.

- [ ] **Step 7: Review requirements against the diff**

Verify from `git diff --stat` and `git diff` that:

- no `ckpts` paths were modified;
- no downloader or network fallback was added;
- `active` remains `original_lama`;
- the orchestrator contains no SmartEraser-specific branch;
- all new public functions and methods have type hints;
- only inference-oriented SmartEraser primitives are imported.

- [ ] **Step 8: Commit integration wiring**

Run:

```powershell
git add backend/models/manager.py backend/config.yaml tests/test_background_inpainting_packages.py
git commit -m "feat: configure SmartEraser background inpainting"
```

Expected: the configuration and registration changes are committed without
staging the untracked upstream `SmartEraser` directory or any checkpoint.
