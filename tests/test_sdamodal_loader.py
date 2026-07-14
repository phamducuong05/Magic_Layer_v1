"""Tests for loading the SDAmodal runtime model."""

import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package

from backend.models.completion.model_loader import load_sdamodal_model


class FakeSDAmodalModel:
    def __init__(self, params, dist_model, device):
        self.params = params
        self.dist_model = dist_model
        self.device = device
        self.loaded_checkpoint = None
        self.phase = None

    def load_state(self, checkpoint_path):
        self.loaded_checkpoint = checkpoint_path

    def switch_to(self, phase):
        self.phase = phase


def _write_config(path, algorithm="AWSDM"):
    path.write_text(
        f"""
model:
  algo: {algorithm}
  use_rgb: true
  backbone_arch: fake
data:
  input_size: 512
  enlarge_box: 3.0
""".strip(),
        encoding="utf-8",
    )


def test_loader_follows_sdamodal_configuration_and_checkpoint_sequence(tmp_path):
    config_path = tmp_path / "sdamodal.yaml"
    checkpoint_path = tmp_path / "sdamodal.pth"
    _write_config(config_path)

    model, config = load_sdamodal_model(
        config_path,
        checkpoint_path,
        device="cpu",
        model_classes={"AWSDM": FakeSDAmodalModel},
    )

    assert model.params is config["model"]
    assert model.dist_model is False
    assert model.device == "cpu"
    assert model.loaded_checkpoint is checkpoint_path
    assert model.phase == "eval"
    assert config["data"] == {"input_size": 512, "enlarge_box": 3.0}


def test_loader_rejects_algorithm_missing_from_completion_models(tmp_path):
    config_path = tmp_path / "sdamodal.yaml"
    _write_config(config_path, algorithm="MissingModel")

    with pytest.raises(ValueError, match="SDAmodal algorithm 'MissingModel'"):
        load_sdamodal_model(
            config_path,
            tmp_path / "sdamodal.pth",
            device="cpu",
            model_classes={"AWSDM": FakeSDAmodalModel},
        )
