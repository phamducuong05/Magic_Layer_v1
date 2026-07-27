import numpy as np
from PIL import Image


def _config():
    return {
        "source_root": "lama",
        "checkpoint_config_path": "config.yaml",
        "generator_weights_path": "generator_state.pt",
        "pad_out_to_modulo": 8,
        "generation_mask_expansion": 3,
        "composition_mask_expansion": 1,
        "feather_radius": 0.0,
    }


def test_original_lama_adapter_skips_runtime_for_empty_mask(monkeypatch):
    from backend.models.background_inpainting.original_lama import adapter

    calls = []

    class FakeRuntime:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def inpaint(self, image, mask):
            calls.append(("inpaint", image, mask))
            return Image.new("RGB", image.size, "black")

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(adapter, "OriginalLamaRuntime", FakeRuntime)
    model = adapter.OriginalLamaBackgroundInpaintingModel(
        config=_config(),
        device="cpu",
    )
    source = Image.new("RGB", (5, 4), "white")

    result = model.process(source, Image.new("L", source.size, 0))

    assert np.array_equal(np.asarray(result), np.asarray(source))
    assert [call[0] for call in calls] == ["init"]


def test_original_lama_adapter_uses_generation_mask_and_preserves_outside(
    monkeypatch,
):
    from backend.models.background_inpainting.original_lama import adapter

    runtime_masks = []

    class FakeRuntime:
        def __init__(self, **_kwargs):
            pass

        def inpaint(self, image, mask):
            runtime_masks.append(np.asarray(mask))
            return Image.new("RGB", image.size, "black")

        def close(self):
            pass

    monkeypatch.setattr(adapter, "OriginalLamaRuntime", FakeRuntime)
    model = adapter.OriginalLamaBackgroundInpaintingModel(
        config=_config(),
        device="cpu",
    )
    source = Image.new("RGB", (7, 7), "white")
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 3] = 255

    result = model.process(source, Image.fromarray(mask, mode="L"))
    result_array = np.asarray(result)

    assert np.count_nonzero(runtime_masks[0]) == 9
    assert np.all(result_array[3, 3] == 0)
    assert np.all(result_array[0, 0] == 255)


def test_original_lama_adapter_reports_raw_and_composed_artifacts(monkeypatch):
    from backend.models.background_inpainting.original_lama import adapter

    class FakeRuntime:
        def __init__(self, **_kwargs):
            pass

        def inpaint(self, image, mask):
            return Image.new("RGB", image.size, (10, 20, 30))

        def close(self):
            pass

    monkeypatch.setattr(adapter, "OriginalLamaRuntime", FakeRuntime)
    model = adapter.OriginalLamaBackgroundInpaintingModel(
        config=_config(),
        device="cpu",
    )
    source = Image.new("RGB", (7, 7), (200, 210, 220))
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 3] = 255
    artifacts = []

    result = model.process(
        source,
        Image.fromarray(mask, mode="L"),
        artifact_callback=lambda stage, image: artifacts.append(
            (stage, image.copy())
        ),
    )

    assert [stage for stage, _ in artifacts] == [
        "after_lama",
        "after_composition_blend",
    ]
    assert np.all(np.asarray(artifacts[0][1]) == (10, 20, 30))
    assert np.all(np.asarray(artifacts[1][1])[0, 0] == (200, 210, 220))
    assert np.all(np.asarray(artifacts[1][1])[3, 3] == (10, 20, 30))
    assert np.array_equal(np.asarray(result), np.asarray(artifacts[1][1]))


def test_original_lama_adapter_closes_runtime_before_base_cleanup(monkeypatch):
    from backend.models.background_inpainting.original_lama import adapter

    closed = []

    class FakeRuntime:
        def __init__(self, **_kwargs):
            pass

        def inpaint(self, image, mask):
            return image

        def close(self):
            closed.append(True)

    monkeypatch.setattr(adapter, "OriginalLamaRuntime", FakeRuntime)
    model = adapter.OriginalLamaBackgroundInpaintingModel(
        config=_config(),
        device="cpu",
    )

    model.unload()

    assert closed == [True]
    assert model.__dict__ == {}
