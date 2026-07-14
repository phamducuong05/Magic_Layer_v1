"""Tests for in-memory SDAmodal batch completion orchestration."""

import sys
import types
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package

from backend.models.completion import batch_completion


def test_batch_completion_routes_inputs_and_preserves_visible_masks(monkeypatch):
    model = object()
    feature_pyramid = object()
    first = np.zeros((8, 10), dtype=np.uint8)
    first[2:6, 3:7] = 1
    second = np.zeros((8, 10), dtype=np.uint8)
    second[1:3, 8:10] = 255
    tight_bboxes = [(3, 2, 4, 4), (8, 1, 2, 2)]
    config = {
        "model": {"use_rgb": True},
        "data": {"input_size": 512, "enlarge_box": 4.0},
    }
    patches = [object(), object()]
    captured = {}

    def fake_infer(
        received_model,
        received_features,
        inmodal,
        category,
        bboxes,
        **kwargs,
    ):
        captured["infer"] = (
            received_model,
            received_features,
            inmodal.copy(),
            category.copy(),
            bboxes.copy(),
            kwargs,
        )
        return patches

    def fake_restore(received_patches, bboxes, height, width, interp):
        captured["restore"] = (
            received_patches,
            bboxes.copy(),
            height,
            width,
            interp,
        )
        restored = np.zeros((2, height, width), dtype=np.float32)
        restored[1, 0, 0] = 0.8
        return restored

    monkeypatch.setattr(
        batch_completion.inference,
        "infer_amodal_aw_sdm",
        fake_infer,
    )
    monkeypatch.setattr(
        batch_completion.inference,
        "patch_to_fullimage",
        fake_restore,
    )

    results = batch_completion.complete_masks_from_features(
        model,
        feature_pyramid,
        [first, second],
        tight_bboxes,
        image_shape=(8, 10),
        config=config,
        device="cpu",
    )

    received = captured["infer"]
    assert received[0] is model
    assert received[1] is feature_pyramid
    assert received[2].dtype == np.uint8
    np.testing.assert_array_equal(received[3], np.ones(2, dtype=np.int32))
    np.testing.assert_array_equal(
        received[4],
        np.array([[1, 0, 8, 8], [7, 0, 4, 4]], dtype=np.int32),
    )
    assert received[5] == {
        "use_rgb": True,
        "th": 0.5,
        "input_size": 512,
        "min_input_size": 16,
        "interp": "nearest",
        "args": None,
        "device": "cpu",
    }

    restored = captured["restore"]
    assert restored[0] is patches
    np.testing.assert_array_equal(restored[1], received[4])
    assert restored[2:] == (8, 10, "linear")

    assert len(results) == 2
    assert all(result.dtype == np.uint8 for result in results)
    np.testing.assert_array_equal(results[0], first)
    expected_second = (second > 0).astype(np.uint8)
    expected_second[0, 0] = 1
    np.testing.assert_array_equal(results[1], expected_second)


def test_empty_batch_skips_sdamodal_inference(monkeypatch):
    monkeypatch.setattr(
        batch_completion.inference,
        "infer_amodal_aw_sdm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("SDAmodal must not run for an empty batch")
        ),
    )

    results = batch_completion.complete_masks_from_features(
        object(),
        object(),
        [],
        [],
        image_shape=(8, 10),
        config={},
        device="cpu",
    )

    assert results == []
