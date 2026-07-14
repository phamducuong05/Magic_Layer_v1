"""Tests for the model-agnostic amodal completion architecture."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Avoid importing backend.models.__init__, which eagerly loads model adapters.
models_package = types.ModuleType("backend.models")
models_package.__path__ = [
    str(Path(__file__).resolve().parents[1] / "backend" / "models")
]
sys.modules["backend.models"] = models_package

from backend.models.base import BaseCompletionModel
from backend.models.registry import ModelRegistry


class FakeCompletionModel(BaseCompletionModel):
    def _load_model(self):
        self.loaded = True

    def complete(self, image, modal_masks):
        return modal_masks


def test_completion_model_contract_preserves_mask_order():
    model = FakeCompletionModel(config={}, device="cpu")
    image = Image.new("RGB", (2, 2))
    first = np.zeros((2, 2), dtype=np.uint8)
    second = np.ones((2, 2), dtype=np.uint8)

    result = model.complete(image, [first, second])

    assert model.loaded is True
    assert result[0] is first
    assert result[1] is second


def test_completion_model_requires_complete_method():
    class IncompleteCompletionModel(BaseCompletionModel):
        def _load_model(self):
            pass

    with pytest.raises(TypeError):
        IncompleteCompletionModel(config={}, device="cpu")


def test_completion_adapter_can_register_and_be_retrieved():
    registered = ModelRegistry.register("completion", "fake-step-4")(
        FakeCompletionModel
    )

    assert registered is FakeCompletionModel
    assert (
        ModelRegistry.get_class("completion", "fake-step-4")
        is FakeCompletionModel
    )


def test_existing_registry_categories_remain_available():
    assert {"matting", "inpainting", "segmentation", "completion"}.issubset(
        ModelRegistry._registry
    )


def test_unknown_completion_model_has_clear_error():
    with pytest.raises(ValueError, match="Model 'missing' not found"):
        ModelRegistry.get_class("completion", "missing")
