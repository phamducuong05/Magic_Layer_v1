"""Tests for explicit SDAmodal device propagation."""

import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package

from backend.models.completion import inference
from backend.models.completion.models.aw_sdm import AWSDM
from backend.models.completion.models.single_stage_model import SingleStageModel
from backend.models.completion.utils import common_utils


class RecordingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen_devices = []

    def forward(self, modal, features):
        self.seen_devices.append(modal.device)
        self.seen_devices.extend(value.device for value in features.values())
        height, width = modal.shape[-2:]
        return torch.zeros((1, 2, height, width), device=modal.device)


def test_net_forward_uses_model_device_for_mask_and_features():
    network = RecordingNetwork()
    model = types.SimpleNamespace(model=network, device=torch.device("cpu"))
    features = {level: np.zeros((2, 2, 3)) for level in range(4)}
    modal = np.ones((4, 4), dtype=np.uint8)

    result = inference.net_forward_aw_sdm(
        model,
        features,
        modal,
        eraser=None,
        use_rgb=True,
        th=0.5,
    )

    assert result.shape == (4, 4)
    assert network.seen_devices == [torch.device("cpu")] * 5


def test_single_stage_model_moves_network_to_requested_device(monkeypatch):
    network = RecordingNetwork()
    monkeypatch.setitem(
        sys.modules[
            "backend.models.completion.models.backbone"
        ].__dict__,
        "tiny-test-backbone",
        lambda **_kwargs: network,
    )
    params = {
        "backbone_arch": "tiny-test-backbone",
        "backbone_param": {},
        "optim": "SGD",
        "lr": 0.1,
        "weight_decay": 0.0,
    }

    model = SingleStageModel(params, device="cpu")

    assert model.device == torch.device("cpu")
    assert next(model.model.parameters()).device == torch.device("cpu")


def test_aw_sdm_set_input_uses_selected_device():
    model = AWSDM.__new__(AWSDM)
    model.device = torch.device("cpu")
    rgb = {0: torch.ones((1, 1, 2, 2))}
    mask = torch.ones((1, 1, 2, 2))
    target = torch.ones((1, 2, 2))

    model.set_input(rgb=rgb, mask=mask, target=target)

    assert model.rgb[0].device == torch.device("cpu")
    assert model.mask.device == torch.device("cpu")
    assert model.target.device == torch.device("cpu")


def test_checkpoint_loading_maps_to_requested_device(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()
    network = nn.Linear(2, 2)
    checkpoint = {
        "state_dict": network.state_dict(),
        "step": 3,
    }
    load_calls = []

    def fake_load(path, map_location):
        load_calls.append((path, map_location))
        return checkpoint

    monkeypatch.setattr(torch, "load", fake_load)

    common_utils.load_state(
        str(checkpoint_path),
        network,
        device=torch.device("cpu"),
    )

    assert load_calls == [(str(checkpoint_path), torch.device("cpu"))]


def test_production_completion_path_has_no_hard_coded_cuda():
    paths = [
        MODELS_PATH / "completion" / "inference.py",
        MODELS_PATH / "completion" / "models" / "aw_sdm.py",
        MODELS_PATH / "completion" / "models" / "single_stage_model.py",
        MODELS_PATH / "completion" / "utils" / "common_utils.py",
    ]

    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert ".cuda(" not in source
    assert "cuda:0" not in source
