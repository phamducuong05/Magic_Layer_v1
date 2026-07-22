import importlib.util
import logging
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image


# Avoid importing backend.models.__init__, which eagerly imports optional models.
MODELS_PATH = Path(__file__).resolve().parents[1] / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package


class FakeIImage:
    def __init__(self, value):
        if isinstance(value, FakeIImage):
            self.image = value.image.copy()
        elif isinstance(value, Image.Image):
            self.image = value.copy()
        else:
            self.image = Image.fromarray(np.asarray(value, dtype=np.uint8))

    def pil(self):
        return self.image.copy()


def make_runtime(*, result_color=(20, 40, 60)):
    calls = []
    reset_calls = []

    def load_inpainting_model(**kwargs):
        calls.append(("load_inpainting", kwargs))
        return "inpainting-model"

    def load_sr_model(**kwargs):
        calls.append(("load_sr", kwargs))
        return "sr-model"

    def sd_run(**kwargs):
        calls.append(("sd_run", kwargs))
        return FakeIImage(Image.new("RGB", (512, 512), result_color))

    def rasg_run(**kwargs):
        calls.append(("rasg_run", kwargs))
        return FakeIImage(Image.new("RGB", (512, 512), result_color))

    def run_sr(**kwargs):
        calls.append(("run_sr", kwargs))
        return Image.new("RGB", (2048, 2048), (70, 80, 90))

    runtime = SimpleNamespace(
        IImage=FakeIImage,
        load_inpainting_model=load_inpainting_model,
        load_sr_model=load_sr_model,
        sd_run=sd_run,
        rasg_run=rasg_run,
        sr_run=run_sr,
        reset_state=lambda: reset_calls.append(True),
    )
    return runtime, calls, reset_calls


class FakeModule:
    def __init__(self, name, moves):
        self.name = name
        self.device = "cpu"
        self.moves = moves

    def to(self, device=None, **_kwargs):
        self.device = str(device)
        self.moves.append((self.name, self.device))
        return self

    def requires_grad_(self, _enabled):
        return self

    def zero_grad(self, **_kwargs):
        return None


class FakeDDIM:
    def __init__(self, name, moves, *, super_resolution=False):
        self.name = name
        self.vae = FakeModule(f"{name}.vae", moves)
        self.encoder = FakeModule(f"{name}.encoder", moves)
        self.unet = FakeModule(f"{name}.unet", moves)
        if super_resolution:
            self.low_scale_model = FakeModule(f"{name}.low_scale", moves)


class FakeOpenCLIPBlock(nn.Module):
    def __init__(self, *, batch_first):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=4,
            num_heads=1,
            batch_first=batch_first,
        )

    def forward(self, x, attn_mask=None):
        return self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]


class FakeOpenCLIPModel(nn.Module):
    def __init__(self, *, batch_first):
        super().__init__()
        self.token_embedding = nn.Embedding(16, 4)
        self.positional_embedding = nn.Parameter(torch.zeros(3, 4))
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            [FakeOpenCLIPBlock(batch_first=batch_first)]
        )
        self.transformer.grad_checkpointing = False
        self.register_buffer("attn_mask", torch.zeros(3, 3))
        self.ln_final = nn.Identity()


def load_open_clip_embedder_module(monkeypatch):
    """Load the vendored encoder without installing its optional dependency."""
    monkeypatch.setitem(sys.modules, "open_clip", types.ModuleType("open_clip"))
    module_path = (
        MODELS_PATH
        / "object_reconstruction"
        / "hd_painter"
        / "src"
        / "smplfusion"
        / "models"
        / "encoders"
        / "open_clip_embedder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_open_clip_embedder_runtime", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batch_first", [False, True])
def test_open_clip_embedder_supports_both_attention_layouts(
    monkeypatch, batch_first
):
    module = load_open_clip_embedder_module(monkeypatch)
    embedder = module.FrozenOpenCLIPEmbedder.__new__(
        module.FrozenOpenCLIPEmbedder
    )
    nn.Module.__init__(embedder)
    embedder.model = FakeOpenCLIPModel(batch_first=batch_first)
    embedder.layer_idx = 0
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    encoded = embedder.encode_with_transformer(tokens)

    assert encoded.shape == (2, 3, 4)


def build_adapter(monkeypatch, config=None, runtime=None):
    from backend.models.object_reconstruction import adapter as module

    runtime = runtime or make_runtime()[0]
    monkeypatch.setattr(module, "_load_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    return module.HDPainterObjectReconstruction(
        config=config or {"super_resolution": {"enabled": False}},
        device="cuda:0",
    )


def test_registration_is_lightweight_and_object_reconstruction_only():
    from backend.models.registry import ModelRegistry
    import backend.models.object_reconstruction.adapter  # noqa: F401

    heavy_prefix = "backend.models.object_reconstruction.hd_painter.src"
    assert not any(name.startswith(heavy_prefix) for name in sys.modules)
    assert (
        ModelRegistry.get_class("object_reconstruction", "hd_painter")
        .__name__
        == "HDPainterObjectReconstruction"
    )
    with pytest.raises(ValueError, match="not found"):
        ModelRegistry.get_class("background_inpainting", "hd_painter")


def test_model_package_import_does_not_require_unused_segment_anything(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.abc
import sys
import types

class BlockSegmentAnything(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'segment_anything':
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockSegmentAnything())
models_package = types.ModuleType("backend.models")
models_package.__path__ = [r"{MODELS_PATH}"]
sys.modules["backend.models"] = models_package
import backend.models.object_reconstruction.hd_painter.src.models
assert 'backend.models.object_reconstruction.hd_painter.src.models.sam' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(project_root)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_empty_mask_returns_rgb_without_running_inference(monkeypatch):
    runtime, calls, reset_calls = make_runtime()
    model = build_adapter(monkeypatch, runtime=runtime)
    image = Image.new("RGBA", (37, 37), (1, 2, 3, 255))

    result = model.reconstruct(image, Image.new("L", image.size, 0), "object")

    assert result.mode == "RGB"
    assert result.size == image.size
    assert [name for name, _ in calls] == ["load_inpainting"]
    assert reset_calls == []


@pytest.mark.parametrize(
    ("method", "runner_name"),
    [
        ("baseline", "sd_run"),
        ("painta", "sd_run"),
        ("rasg", "rasg_run"),
        ("painta+rasg", "rasg_run"),
    ],
)
def test_method_dispatch_and_binary_mask_preprocessing(
    monkeypatch, method, runner_name
):
    runtime, calls, reset_calls = make_runtime()
    config = {
        "method": method,
        "positive_prompt": "detailed continuation",
        "negative_prompt": "artifact",
        "num_steps": 50,
        "guidance_scale": 7.5,
        "rasg_eta": 0.1,
        "seed": 9,
        "super_resolution": {"enabled": False},
    }
    model = build_adapter(monkeypatch, config=config, runtime=runtime)
    image = Image.new("RGB", (63, 63), "white")
    mask_array = np.zeros((31, 31), dtype=np.uint8)
    mask_array[5:20, 8:22] = 127

    result = model.reconstruct(
        image,
        Image.fromarray(mask_array, mode="L"),
        "Continue the hidden bicycle",
    )

    run_calls = [entry for entry in calls if entry[0] == runner_name]
    assert len(run_calls) == 1
    kwargs = run_calls[0][1]
    assert kwargs["method"] == method
    assert kwargs["prompt"] == "Continue the hidden bicycle"
    assert kwargs["positive_prompt"] == "detailed continuation"
    assert kwargs["negative_prompt"] == "artifact"
    assert kwargs["image"].pil().size == (512, 512)
    inference_mask = np.asarray(kwargs["mask"].pil())
    assert set(np.unique(inference_mask)) <= {0, 255}
    assert np.all(inference_mask[250, 250] == 255)
    assert result.mode == "RGB"
    assert result.size == (63, 63)
    assert reset_calls == [True]


def test_optional_super_resolution_uses_original_crop_and_mask(monkeypatch):
    runtime, calls, _ = make_runtime()
    config = {
        "method": "painta+rasg",
        "super_resolution": {
            "enabled": True,
            "noise_level": 20,
            "denoising_stride": 50,
            "guidance_scale": 7.5,
            "blend_trick": True,
            "blend_output": True,
            "use_sam_mask": False,
        },
    }
    model = build_adapter(monkeypatch, config=config, runtime=runtime)
    image = Image.new("RGB", (45, 45), "blue")
    mask = Image.new("L", image.size, 0)
    mask.putpixel((20, 20), 255)

    result = model.reconstruct(image, mask, "hidden blue object")

    assert [name for name, _ in calls[:2]] == ["load_inpainting", "load_sr"]
    sr_kwargs = next(kwargs for name, kwargs in calls if name == "run_sr")
    assert sr_kwargs["sam_predictor"] is None
    assert sr_kwargs["use_sam_mask"] is False
    assert isinstance(sr_kwargs["lr_image"], Image.Image)
    assert isinstance(sr_kwargs["hr_image"], Image.Image)
    assert isinstance(sr_kwargs["hr_mask"], Image.Image)
    assert sr_kwargs["hr_image"].size == image.size
    assert sr_kwargs["hr_mask"].mode == "RGB"
    assert result.size == image.size
    assert result.getpixel((0, 0)) == (70, 80, 90)


def test_hd_painter_logs_generation_and_super_resolution_stages(
    monkeypatch, caplog
):
    runtime, _, _ = make_runtime()
    model = build_adapter(
        monkeypatch,
        config={"super_resolution": {"enabled": True}},
        runtime=runtime,
    )

    with caplog.at_level(logging.INFO):
        model.reconstruct(
            Image.new("RGB", (32, 32), "white"),
            Image.new("L", (32, 32), 255),
            "hidden object",
        )

    assert "[HD_PAINTER_GENERATION_512] START" in caplog.text
    assert "[HD_PAINTER_GENERATION_512] COMPLETE" in caplog.text
    assert "[HD_PAINTER_SUPER_RESOLUTION] START" in caplog.text
    assert "[HD_PAINTER_SUPER_RESOLUTION] COMPLETE" in caplog.text


def test_super_resolution_receives_pil_images_with_source_metadata(monkeypatch):
    """The bundled SR runner reads ``hr_image.info`` before wrapping inputs."""
    runtime, calls, _ = make_runtime()

    def run_sr(**kwargs):
        # Match the input contract used by HD-Painter's methods/sr.py.
        assert isinstance(kwargs["lr_image"], Image.Image)
        assert isinstance(kwargs["hr_image"], Image.Image)
        assert isinstance(kwargs["hr_mask"], Image.Image)
        assert kwargs["hr_image"].info["test-metadata"] == "preserved"
        calls.append(("run_sr", kwargs))
        return Image.new("RGB", (2048, 2048), (70, 80, 90))

    runtime.sr_run = run_sr
    model = build_adapter(
        monkeypatch,
        config={"super_resolution": {"enabled": True}},
        runtime=runtime,
    )
    image = Image.new("RGB", (45, 45), "blue")
    image.info["test-metadata"] = "preserved"
    mask = Image.new("L", image.size, 255)

    result = model.reconstruct(image, mask, "hidden blue object")

    assert result.size == image.size


@pytest.mark.parametrize(
    ("failing_call", "expected_stage", "expected_reset_count"),
    [
        ("rasg_run", "generation_512", 1),
        ("run_sr", "super_resolution", 2),
    ],
)
def test_inference_errors_report_the_failed_hd_painter_stage(
    monkeypatch, failing_call, expected_stage, expected_reset_count
):
    from backend.models.object_reconstruction import adapter as module

    runtime, _, reset_calls = make_runtime()

    def fail(**_kwargs):
        raise RuntimeError("synthetic stage failure")

    if failing_call == "rasg_run":
        runtime.rasg_run = fail
        config = {"super_resolution": {"enabled": False}}
    else:
        runtime.sr_run = fail
        config = {"super_resolution": {"enabled": True}}
    model = build_adapter(monkeypatch, config=config, runtime=runtime)
    mask = Image.new("L", (32, 32), 255)

    with pytest.raises(module.ObjectReconstructionError) as error:
        model.reconstruct(Image.new("RGB", (32, 32)), mask, "object")

    assert error.value.stage == expected_stage
    assert "synthetic stage failure" in str(error.value)
    assert len(reset_calls) == expected_reset_count


def test_sequential_cpu_offload_keeps_only_active_stage_on_cuda(monkeypatch):
    moves = []
    calls = []
    resets = []
    inpainting_model = FakeDDIM("inpainting", moves)
    sr_model = FakeDDIM("sr", moves, super_resolution=True)
    # Simulate research-level cache entries left resident by an older adapter.
    for model in (inpainting_model, sr_model):
        for name in ("vae", "encoder", "unet", "low_scale_model"):
            module = getattr(model, name, None)
            if module is not None:
                module.device = "cuda:0"

    def load_inpainting_model(**kwargs):
        calls.append(("load_inpainting", kwargs))
        return inpainting_model

    def load_sr_model(**kwargs):
        calls.append(("load_sr", kwargs))
        return sr_model

    def rasg_run(**kwargs):
        assert kwargs["ddim"] is inpainting_model
        assert inpainting_model.unet.device == "cuda:0"
        assert sr_model.unet.device == "cpu"
        calls.append(("generation", {}))
        return FakeIImage(Image.new("RGB", (512, 512), "green"))

    def sr_run(**kwargs):
        assert kwargs["ddim"] is sr_model
        assert inpainting_model.unet.device == "cpu"
        assert sr_model.unet.device == "cuda:0"
        calls.append(("super_resolution", {}))
        return Image.new("RGB", (2048, 2048), "blue")

    runtime = SimpleNamespace(
        IImage=FakeIImage,
        load_inpainting_model=load_inpainting_model,
        load_sr_model=load_sr_model,
        sd_run=rasg_run,
        rasg_run=rasg_run,
        sr_run=sr_run,
        reset_state=lambda: resets.append(len(calls)),
    )
    from backend.models.object_reconstruction import adapter as module

    monkeypatch.setattr(module, "_load_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    empty_cache = Mock()
    monkeypatch.setattr(module.torch.cuda, "empty_cache", empty_cache)
    model = module.HDPainterObjectReconstruction(
        config={
            "sequential_cpu_offload": True,
            "super_resolution": {"enabled": True},
        },
        device="cuda:0",
    )

    assert calls[0][1]["device"] == "cpu"
    assert calls[1][1]["device"] == "cpu"
    assert inpainting_model.unet.device == "cpu"
    assert sr_model.unet.device == "cpu"

    result = model.reconstruct(
        Image.new("RGB", (64, 64)),
        Image.new("L", (64, 64), 255),
        "hidden object",
    )

    assert result.size == (64, 64)
    assert [name for name, _ in calls[-2:]] == [
        "generation",
        "super_resolution",
    ]
    assert len(resets) >= 2
    assert inpainting_model.unet.device == "cpu"
    assert sr_model.unet.device == "cpu"
    assert empty_cache.call_count >= 2


def test_reconstruct_many_runs_generation_phase_once_before_conditional_sr(
    monkeypatch,
):
    moves = []
    calls = []
    inpainting_model = FakeDDIM("inpainting", moves)
    sr_model = FakeDDIM("sr", moves, super_resolution=True)

    def generation(**_kwargs):
        calls.append("generation")
        return FakeIImage(Image.new("RGB", (512, 512), "green"))

    def super_resolution(**_kwargs):
        calls.append("super_resolution")
        return Image.new("RGB", (2048, 2048), "blue")

    runtime = SimpleNamespace(
        IImage=FakeIImage,
        load_inpainting_model=lambda **_kwargs: inpainting_model,
        load_sr_model=lambda **_kwargs: sr_model,
        sd_run=generation,
        rasg_run=generation,
        sr_run=super_resolution,
        reset_state=Mock(),
    )
    model = build_adapter(
        monkeypatch,
        config={
            "sequential_cpu_offload": True,
            "super_resolution": {
                "enabled": True,
                "minimum_roi_size": 128,
            },
        },
        runtime=runtime,
    )
    moves.clear()

    outcomes = model.reconstruct_many(
        [
            (
                Image.new("RGB", (64, 64), "white"),
                Image.new("L", (64, 64), 255),
                "small object",
            ),
            (
                Image.new("RGB", (256, 256), "white"),
                Image.new("L", (256, 256), 255),
                "large object",
            ),
        ]
    )

    assert calls == ["generation", "generation", "super_resolution"]
    assert all(isinstance(outcome, Image.Image) for outcome in outcomes)
    assert outcomes[0].size == (64, 64)
    assert outcomes[0].getpixel((0, 0)) == (0, 128, 0)
    assert outcomes[1].size == (256, 256)
    assert outcomes[1].getpixel((0, 0)) == (0, 0, 255)
    assert moves.count(("inpainting.unet", "cuda:0")) == 1
    assert moves.count(("sr.unet", "cuda:0")) == 1


def test_reconstruct_many_isolates_generation_failure_and_continues_sr(
    monkeypatch,
):
    runtime, calls, _ = make_runtime()
    generation_count = 0

    def generation(**_kwargs):
        nonlocal generation_count
        generation_count += 1
        if generation_count == 1:
            raise RuntimeError("first generation failed")
        calls.append(("generation_success", {}))
        return FakeIImage(Image.new("RGB", (512, 512), "green"))

    runtime.rasg_run = generation
    model = build_adapter(
        monkeypatch,
        config={"super_resolution": {"enabled": True}},
        runtime=runtime,
    )
    mask = Image.new("L", (64, 64), 255)

    outcomes = model.reconstruct_many(
        [
            (Image.new("RGB", (64, 64)), mask, "first"),
            (Image.new("RGB", (64, 64)), mask, "second"),
        ]
    )

    assert isinstance(outcomes[0], Exception)
    assert outcomes[0].stage == "generation_512"
    assert isinstance(outcomes[1], Image.Image)
    assert [name for name, _ in calls if name == "run_sr"] == ["run_sr"]


def test_offload_to_cpu_preserves_loaded_weights_and_runtime_cache(
    monkeypatch,
):
    moves = []
    inpainting_model = FakeDDIM("inpainting", moves)
    sr_model = FakeDDIM("sr", moves, super_resolution=True)
    runtime = SimpleNamespace(
        IImage=FakeIImage,
        load_inpainting_model=lambda **_kwargs: inpainting_model,
        load_sr_model=lambda **_kwargs: sr_model,
        sd_run=Mock(),
        rasg_run=Mock(),
        sr_run=Mock(),
        reset_state=Mock(),
        clear_model_cache=Mock(),
    )
    model = build_adapter(
        monkeypatch,
        config={"super_resolution": {"enabled": True}},
        runtime=runtime,
    )

    model.offload_to_cpu()

    assert model._inpainting_model is inpainting_model
    assert model._sr_model is sr_model
    assert inpainting_model.unet.device == "cpu"
    assert sr_model.unet.device == "cpu"
    runtime.clear_model_cache.assert_not_called()


def test_unload_clears_hd_painter_runtime_cache_and_owned_weights(monkeypatch):
    from backend.models.object_reconstruction import adapter as module

    runtime, _, reset_calls = make_runtime()
    runtime.clear_model_cache = Mock()
    model = build_adapter(
        monkeypatch,
        config={
            "sequential_cpu_offload": False,
            "super_resolution": {"enabled": False},
        },
        runtime=runtime,
    )
    release_cuda_cache = Mock()
    monkeypatch.setattr(module, "_release_cuda_cache", release_cuda_cache)

    model.unload()

    assert reset_calls == [True]
    runtime.clear_model_cache.assert_called_once_with()
    release_cuda_cache.assert_called_once_with()
    assert model.__dict__ == {}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"method": "unknown"}, "method"),
        ({"model_id": "unknown"}, "model_id"),
        ({"num_steps": 0}, "num_steps"),
        ({"num_steps": 1001}, "num_steps"),
        ({"guidance_scale": 0}, "guidance_scale"),
        ({"super_resolution": {"enabled": True, "use_sam_mask": True}}, "use_sam_mask"),
        ({"super_resolution": {"minimum_roi_size": -1}}, "minimum_roi_size"),
    ],
)
def test_invalid_configuration_is_rejected_before_runtime_import(
    monkeypatch, config, message
):
    from backend.models.object_reconstruction import adapter as module

    imported = False

    def fail_import(**kwargs):
        nonlocal imported
        imported = True
        raise AssertionError("runtime must not be imported")

    monkeypatch.setattr(module, "_load_runtime", fail_import)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    with pytest.raises(module.ObjectReconstructionError, match=message):
        module.HDPainterObjectReconstruction(config=config, device="cuda:0")
    assert imported is False


def test_cuda_is_required_before_runtime_import(monkeypatch):
    from backend.models.object_reconstruction import adapter as module

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module,
        "_load_runtime",
        lambda **kwargs: pytest.fail("runtime must not be imported"),
    )
    with pytest.raises(module.ObjectReconstructionError, match="CUDA"):
        module.HDPainterObjectReconstruction(
            config={"super_resolution": {"enabled": False}},
            device="cuda:0",
        )


def test_global_state_lock_serializes_calls_across_adapter_instances(monkeypatch):
    from backend.models.object_reconstruction import adapter as module

    active = 0
    max_active = 0
    state_lock = threading.Lock()
    runtime, _, _ = make_runtime()

    def slow_run(**kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return FakeIImage(Image.new("RGB", (512, 512), "green"))

    runtime.sd_run = slow_run
    monkeypatch.setattr(module, "_load_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    first = module.HDPainterObjectReconstruction(
        config={"method": "baseline", "super_resolution": {"enabled": False}},
        device="cuda:0",
    )
    second = module.HDPainterObjectReconstruction(
        config={"method": "baseline", "super_resolution": {"enabled": False}},
        device="cuda:0",
    )
    mask = Image.new("L", (32, 32), 255)
    threads = [
        threading.Thread(
            target=model.reconstruct,
            args=(Image.new("RGB", (32, 32)), mask, "object"),
        )
        for model in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 1
