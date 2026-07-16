import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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
    assert sr_kwargs["hr_image"].pil().size == image.size
    assert sr_kwargs["hr_mask"].pil().mode == "RGB"
    assert result.size == image.size
    assert result.getpixel((0, 0)) == (70, 80, 90)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"method": "unknown"}, "method"),
        ({"model_id": "unknown"}, "model_id"),
        ({"num_steps": 0}, "num_steps"),
        ({"num_steps": 1001}, "num_steps"),
        ({"guidance_scale": 0}, "guidance_scale"),
        ({"super_resolution": {"enabled": True, "use_sam_mask": True}}, "use_sam_mask"),
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
