"""Tests for the registered SDAmodal completion adapter."""

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package

from backend.models.completion import adapter


def _install_fake_dependencies(monkeypatch, adapter):
    events = {"extract": [], "batch": []}
    loaded_model = object()
    loaded_config = {
        "model": {"use_rgb": True},
        "data": {"input_size": 512, "enlarge_box": 3.0},
    }
    feature_pyramid = object()

    def fake_load(config_path, checkpoint_path, *, device):
        events["load"] = (config_path, checkpoint_path, device)
        return loaded_model, loaded_config

    class FakeExtractor:
        def __init__(self, *, device):
            events["extractor_device"] = device

        def extract(self, image):
            events["extract"].append(image)
            return feature_pyramid

    expected_outputs = [np.ones((4, 5), dtype=np.uint8)]

    def fake_batch(*args, **kwargs):
        events["batch"].append((args, kwargs))
        return expected_outputs

    monkeypatch.setattr(adapter, "load_sdamodal_model", fake_load)
    monkeypatch.setattr(adapter, "DIFTFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(adapter, "complete_masks_from_features", fake_batch)
    return events, loaded_model, loaded_config, feature_pyramid, expected_outputs


def test_adapter_extracts_dift_once_and_completes_the_whole_batch(
    monkeypatch,
    tmp_path,
):
    events, loaded_model, loaded_config, pyramid, expected = (
        _install_fake_dependencies(monkeypatch, adapter)
    )
    config_path = tmp_path / "sdamodal.yaml"
    checkpoint_path = tmp_path / "sdamodal.pth"
    completion = adapter.SDAmodalCompletionModel(
        config={
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
        },
        device="cpu",
    )
    image = Image.new("RGB", (5, 4))
    masks = [
        np.ones((4, 5), dtype=np.uint8),
        np.eye(4, 5, dtype=np.uint8),
    ]
    bboxes = [(0, 0, 5, 4), (0, 0, 4, 4)]

    outputs = completion.complete(image, masks, bboxes)

    assert events["load"] == (config_path, checkpoint_path, "cpu")
    assert events["extractor_device"] == "cpu"
    assert events["extract"] == [image]
    assert outputs is expected
    assert len(events["batch"]) == 1
    args, kwargs = events["batch"][0]
    assert args == (loaded_model, pyramid, masks, bboxes)
    assert kwargs == {
        "image_shape": (4, 5),
        "config": loaded_config,
        "device": "cpu",
    }


def test_adapter_skips_dift_and_batch_completion_for_empty_input(
    monkeypatch,
    tmp_path,
):
    events, *_ = _install_fake_dependencies(monkeypatch, adapter)
    completion = adapter.SDAmodalCompletionModel(
        config={
            "config_path": str(tmp_path / "sdamodal.yaml"),
            "checkpoint_path": str(tmp_path / "sdamodal.pth"),
        },
        device="cpu",
    )

    assert completion.complete(Image.new("RGB", (2, 2)), [], []) == []
    assert events["extract"] == []
    assert events["batch"] == []


def test_adapter_is_registered_and_manager_imports_registration_module():
    from backend.models.registry import ModelRegistry

    assert (
        ModelRegistry.get_class("completion", "sdamodal")
        is adapter.SDAmodalCompletionModel
    )
    manager_source = (MODELS_PATH / "manager.py").read_text(encoding="utf-8")
    assert "from .completion import adapter" in manager_source
