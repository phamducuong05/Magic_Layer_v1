import numpy as np
from PIL import Image
import pytest


def _config():
    return {
        "checkpoint_dir": "checkpoint",
        "clip_dir": "clip",
        "clip_model_id": "openai/clip-vit-large-patch14",
        "clip_auto_download": True,
        "resolution": 8,
        "context_scale": 3.0,
        "minimum_context_ratio": 0.25,
        "num_inference_steps": 2,
        "guidance_scale": 1.2,
        "seed": 42,
        "dtype": "float16",
        "prompt": "Remove the instance of",
        "negative_prompt": "objects, text, decorations, artifacts",
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
                "clip_model_id": "openai/clip-vit-large-patch14",
                "clip_auto_download": True,
                "device": "cuda:1",
                "dtype": "float16",
                "num_inference_steps": 2,
                "guidance_scale": 1.2,
                "seed": 42,
                "prompt": "Remove the instance of",
                "negative_prompt": (
                    "objects, text, decorations, artifacts"
                ),
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
        "after_smarteraser",
        "after_composition_blend",
    ]
    assert artifacts[0][1].getpixel((4, 4)) == (0, 0, 0)
    assert artifacts[0][1].getpixel((0, 0)) == (255, 255, 255)
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


def test_adapter_inpaints_distant_mask_groups_independently(monkeypatch):
    from backend.models.background_inpainting.smarteraser import adapter

    runtime_inputs = []
    prepared_sources = []

    class ColorRuntime:
        def __init__(self, **_kwargs):
            pass

        def inpaint(self, image, mask):
            runtime_inputs.append((image.copy(), mask.copy()))
            color = "red" if len(runtime_inputs) == 1 else "blue"
            return Image.new("RGB", image.size, color)

        def close(self):
            pass

    real_prepare_inputs = adapter.prepare_inputs

    def record_prepare_source(image, *args, **kwargs):
        prepared_sources.append(np.asarray(image).copy())
        return real_prepare_inputs(image, *args, **kwargs)

    monkeypatch.setattr(adapter, "_create_runtime", ColorRuntime)
    monkeypatch.setattr(adapter, "prepare_inputs", record_prepare_source)
    config = _config()
    config["minimum_context_ratio"] = 0.1
    model = adapter.SmartEraserBackgroundInpaintingModel(config, "cpu")
    source = Image.new("RGB", (100, 100), "white")
    source.putpixel((50, 50), (10, 20, 30))
    mask = Image.new("L", source.size, 0)
    mask.putpixel((15, 15), 255)
    mask.putpixel((85, 85), 255)

    result = model.process(source, mask)

    assert len(runtime_inputs) == 2
    assert all(image.size == (8, 8) for image, _ in runtime_inputs)
    assert result.getpixel((15, 15)) == (255, 0, 0)
    assert result.getpixel((85, 85)) == (0, 0, 255)
    assert result.getpixel((50, 50)) == (10, 20, 30)
    source_array = np.asarray(source)
    assert all(
        np.array_equal(prepared_source, source_array)
        for prepared_source in prepared_sources
    )


def test_adapter_accumulates_one_diagnostic_sequence_for_all_groups(
    monkeypatch,
):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "_create_runtime",
        lambda **kwargs: FakeRuntime(calls, **kwargs),
    )
    config = _config()
    config["minimum_context_ratio"] = 0.1
    model = adapter.SmartEraserBackgroundInpaintingModel(config, "cpu")
    source = Image.new("RGB", (100, 100), "white")
    mask = Image.new("L", source.size, 0)
    mask.putpixel((15, 15), 255)
    mask.putpixel((85, 85), 255)
    artifacts = []

    result = model.process(
        source,
        mask,
        artifact_callback=lambda stage, image: artifacts.append(
            (stage, image.copy())
        ),
    )

    assert [call[0] for call in calls].count("inpaint") == 2
    assert [stage for stage, _ in artifacts] == [
        "after_smarteraser",
        "after_composition_blend",
    ]
    assert np.array_equal(np.asarray(result), np.asarray(artifacts[1][1]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_scale", 0.5),
        ("minimum_context_ratio", 0.0),
    ],
)
def test_adapter_rejects_invalid_context_before_runtime_load(
    monkeypatch,
    field,
    value,
):
    from backend.models.background_inpainting.smarteraser import adapter

    calls = []
    monkeypatch.setattr(
        adapter,
        "_create_runtime",
        lambda **kwargs: FakeRuntime(calls, **kwargs),
    )
    invalid_config = _config()
    invalid_config[field] = value

    try:
        adapter.SmartEraserBackgroundInpaintingModel(invalid_config, "cpu")
    except ValueError as error:
        assert field in str(error)
    else:
        raise AssertionError(f"Invalid {field} was accepted")

    assert calls == []
