"""Tests for reusable, in-memory DIFT feature extraction."""

import sys
import types
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package

from backend.models.completion.dift import extract_dift_amodal


class FakeFeaturizer:
    def __init__(self):
        self.calls = []

    def forward(self, image_tensor, **kwargs):
        self.calls.append((image_tensor.clone(), kwargs))
        return {
            level: torch.full((1, level + 1, 2, 3), float(level))
            for level in range(4)
        }


def test_extract_runs_featurizer_once_and_returns_cpu_feature_pyramid():
    featurizer = FakeFeaturizer()
    extractor = extract_dift_amodal.DIFTFeatureExtractor(
        featurizer=featurizer,
        image_size=(4, 6),
        prompt="object",
        timestep=181,
        ensemble_size=2,
    )

    features = extractor.extract(Image.new("RGB", (2, 3), color=(255, 0, 127)))

    assert len(featurizer.calls) == 1
    image_tensor, arguments = featurizer.calls[0]
    assert image_tensor.shape == (3, 6, 4)
    assert image_tensor.dtype == torch.float32
    assert image_tensor.min().item() >= -1.0
    assert image_tensor.max().item() <= 1.0
    assert arguments == {
        "prompt": "object",
        "t": 181,
        "up_ft_index": 1,
        "ensemble_size": 2,
    }
    assert set(features) == {0, 1, 2, 3}
    assert [features[level].shape for level in range(4)] == [
        (1, 2, 3),
        (2, 2, 3),
        (3, 2, 3),
        (4, 2, 3),
    ]
    assert all(feature.device.type == "cpu" for feature in features.values())
    assert all(not feature.requires_grad for feature in features.values())


def test_extract_rejects_incomplete_feature_pyramid():
    featurizer = FakeFeaturizer()
    original_forward = featurizer.forward

    def incomplete_forward(*args, **kwargs):
        features = original_forward(*args, **kwargs)
        del features[3]
        return features

    featurizer.forward = incomplete_forward
    extractor = extract_dift_amodal.DIFTFeatureExtractor(featurizer=featurizer)

    try:
        extractor.extract(Image.new("RGB", (2, 2)))
    except ValueError as error:
        assert str(error) == "missing DIFT feature levels: 3"
    else:
        raise AssertionError("an incomplete DIFT feature pyramid was accepted")


def test_production_extractor_has_no_feature_file_io_or_ambiguous_imports():
    source = (
        MODELS_PATH / "completion" / "dift" / "extract_dift_amodal.py"
    ).read_text(encoding="utf-8")

    assert "torch.save" not in source
    assert "torch.load" not in source
    assert "os.listdir" not in source
    assert "from src" not in source
    assert "ipdb" not in source


def test_amodal_featurizer_uses_configured_device_instead_of_cuda_calls():
    source = (
        MODELS_PATH / "completion" / "dift" / "src" / "models" / "dift_sd.py"
    ).read_text(encoding="utf-8")

    assert "self.device = torch.device(device)" in source
    assert ".cuda(" not in source
    assert "device=self.device" in source
