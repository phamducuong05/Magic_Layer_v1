"""Tests for in-memory DIFT feature routing into SDAmodal."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package

from backend.models.completion import inference


def _feature_pyramid():
    return {
        level: torch.full((3, 8, 8), float(level))
        for level in range(4)
    }


def test_multiple_masks_reuse_in_memory_feature_pyramid(monkeypatch):
    pyramid = _feature_pyramid()
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, 0:4, 0:4] = 1
    masks[1, 4:8, 4:8] = 1
    boxes = np.array([[0, 0, 4, 4], [4, 4, 4, 4]])
    calls = []

    def fake_forward(
        model,
        features,
        modal_patch,
        eraser,
        use_rgb,
        threshold,
        **kwargs,
    ):
        calls.append(features)
        return np.full(modal_patch.shape, len(calls), dtype=np.uint8)

    monkeypatch.setattr(inference, "net_forward_aw_sdm", fake_forward)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("feature files must not be read"),
    )
    model = types.SimpleNamespace(device=torch.device("cpu"))

    results = inference.infer_amodal_aw_sdm(
        model,
        pyramid,
        masks,
        np.ones(2, dtype=np.uint8),
        boxes,
        input_size=4,
        device="cpu",
    )

    assert len(calls) == 2
    assert [set(call) for call in calls] == [{0, 1, 2, 3}] * 2
    assert [call[0].shape[:2] for call in calls] == [(24, 24), (24, 24)]
    assert [call[1].shape[:2] for call in calls] == [(48, 48), (48, 48)]
    assert [call[2].shape[:2] for call in calls] == [(96, 96), (96, 96)]
    assert [call[3].shape[:2] for call in calls] == [(96, 96), (96, 96)]
    assert np.all(results[0] == 1)
    assert np.all(results[1] == 2)


def test_missing_feature_level_is_rejected():
    pyramid = _feature_pyramid()
    del pyramid[3]
    model = types.SimpleNamespace(device=torch.device("cpu"))

    with pytest.raises(ValueError, match="missing DIFT feature levels: 3"):
        inference.infer_amodal_aw_sdm(
            model,
            pyramid,
            np.ones((1, 4, 4), dtype=np.uint8),
            np.ones(1, dtype=np.uint8),
            np.array([[0, 0, 4, 4]]),
            device="cpu",
        )


def test_malformed_feature_level_is_rejected():
    pyramid = _feature_pyramid()
    pyramid[2] = torch.zeros((1, 3, 8, 8))
    model = types.SimpleNamespace(device=torch.device("cpu"))

    with pytest.raises(ValueError, match="level 2 must have shape"):
        inference.infer_amodal_aw_sdm(
            model,
            pyramid,
            np.ones((1, 4, 4), dtype=np.uint8),
            np.ones(1, dtype=np.uint8),
            np.array([[0, 0, 4, 4]]),
            device="cpu",
        )


def test_inference_source_has_no_feature_file_loading():
    source = (
        MODELS_PATH / "completion" / "inference.py"
    ).read_text(encoding="utf-8")

    assert "feature/pth" not in source
    assert "torch.load" not in source
