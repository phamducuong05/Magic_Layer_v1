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


class FakeRuntime:
    def __init__(self, calls, **kwargs):
        self.calls = calls
        self.calls.append(("init", kwargs))

    def inpaint(self, image, mask):
        self.calls.append(("inpaint", image.copy(), mask.copy()))
        return Image.new("RGB", image.size, "black")

    def close(self):
        self.calls.append(("close",))


def test_adapter_skips_runtime_inference_for_empty_mask(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "_create_runtime",
        lambda **kwargs: FakeRuntime(calls, **kwargs),
    )
    model = adapter.SmartEraserBackgroundInpaintingModel(_config(), "cpu")
    source = Image.new("RGB", (8, 8), "white")

    result = model.process(source, Image.new("L", source.size, 0))

    assert np.array_equal(np.asarray(result), np.asarray(source))
    assert [call[0] for call in calls] == ["init"]


def test_adapter_preserves_pixels_outside_composition_mask(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "_create_runtime",
        lambda **kwargs: FakeRuntime(calls, **kwargs),
    )
    model = adapter.SmartEraserBackgroundInpaintingModel(_config(), "cpu")
    source = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", source.size, 0)
    mask.putpixel((4, 4), 255)

    result = np.asarray(model.process(source, mask))

    assert np.all(result[0, 0] == 255)
    assert np.all(result[4, 4] == 0)


def test_adapter_forwards_runtime_configuration(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "_create_runtime",
        lambda **kwargs: FakeRuntime(calls, **kwargs),
    )

    adapter.SmartEraserBackgroundInpaintingModel(_config(), "cuda:1")

    assert calls == [
        (
            "init",
            {
                "checkpoint_dir": "checkpoint",
                "clip_dir": "clip",
                "device": "cuda:1",
                "dtype": "float16",
                "num_inference_steps": 2,
                "guidance_scale": 1.5,
                "seed": 42,
                "prompt": "Remove the instance of object",
                "negative_prompt": "",
            },
        )
    ]


def test_adapter_reports_diagnostic_artifacts(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "_create_runtime",
        lambda **kwargs: FakeRuntime(calls, **kwargs),
    )
    model = adapter.SmartEraserBackgroundInpaintingModel(_config(), "cpu")
    source = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", source.size, 0)
    mask.putpixel((4, 4), 255)
    artifacts = []

    result = model.process(
        source,
        mask,
        artifact_callback=lambda stage, image: artifacts.append(
            (stage, image.copy())
        ),
    )

    assert [stage for stage, _ in artifacts] == [
        "after_lama",
        "after_composition_blend",
    ]
    assert np.all(np.asarray(artifacts[0][1]) == 0)
    assert np.array_equal(np.asarray(result), np.asarray(artifacts[1][1]))


def test_adapter_closes_runtime_before_base_cleanup(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "_create_runtime",
        lambda **kwargs: FakeRuntime(calls, **kwargs),
    )
    model = adapter.SmartEraserBackgroundInpaintingModel(_config(), "cpu")

    model.unload()

    assert [call[0] for call in calls] == ["init", "close"]
    assert model.__dict__ == {}


def test_adapter_rejects_non_positive_resolution(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "_create_runtime",
        lambda **kwargs: FakeRuntime(calls, **kwargs),
    )
    invalid_config = _config()
    invalid_config["resolution"] = 0

    try:
        adapter.SmartEraserBackgroundInpaintingModel(invalid_config, "cpu")
    except ValueError as error:
        assert "resolution" in str(error)
    else:
        raise AssertionError("Non-positive resolution was accepted")

    assert calls == []
