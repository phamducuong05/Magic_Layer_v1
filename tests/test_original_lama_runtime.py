import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml
from PIL import Image


def test_vendored_inference_utils_do_not_require_lightning():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from saicinpainting.utils import get_shape; print(get_shape(1))",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "lama")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_generator_config_resolves_local_interpolations():
    from backend.models.background_inpainting.original_lama.runtime import (
        load_generator_config,
    )

    config_path = (
        Path(__file__).resolve().parents[1]
        / "backend"
        / "models"
        / "background_inpainting"
        / "original_lama"
        / "checkpoints"
        / "config.yaml"
    )

    generator = load_generator_config(config_path)

    assert generator["kind"] == "ffc_resnet"
    assert generator["downsample_conv_kwargs"]["ratio_gin"] == 0
    assert generator["downsample_conv_kwargs"]["ratio_gout"] == 0
    assert generator["resnet_conv_kwargs"]["ratio_gin"] == 0.75
    assert generator["resnet_conv_kwargs"]["ratio_gout"] == 0.75


def test_prepare_inputs_binarizes_mask_and_pads_to_modulo():
    from backend.models.background_inpainting.original_lama.runtime import (
        prepare_inputs,
    )

    image = Image.new("RGB", (7, 5), (10, 20, 30))
    mask_array = np.zeros((5, 7), dtype=np.uint8)
    mask_array[1:3, 2:4] = 255

    image_tensor, mask_tensor, original_size = prepare_inputs(
        image,
        Image.fromarray(mask_array, mode="L"),
        modulo=8,
    )

    assert image_tensor.shape == (1, 3, 8, 8)
    assert mask_tensor.shape == (1, 1, 8, 8)
    assert original_size == (5, 7)
    assert set(torch.unique(mask_tensor).tolist()) <= {0.0, 1.0}
    assert torch.all(mask_tensor[0, 0, 1:3, 2:4] == 1)


def test_compose_prediction_preserves_known_pixels_and_unpads():
    from backend.models.background_inpainting.original_lama.runtime import (
        compose_prediction,
    )

    image = torch.ones((1, 3, 8, 8), dtype=torch.float32)
    mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    mask[:, :, 1:3, 2:4] = 1
    prediction = torch.zeros_like(image)

    result = compose_prediction(
        prediction,
        image,
        mask,
        original_size=(5, 7),
    )
    array = np.asarray(result)

    assert result.mode == "RGB"
    assert result.size == (7, 5)
    assert np.all(array[1:3, 2:4] == 0)
    assert np.all(array[0, 0] == 255)


def test_generator_artifact_contains_only_tensor_state():
    artifact = (
        Path(__file__).resolve().parents[1]
        / "backend"
        / "models"
        / "background_inpainting"
        / "original_lama"
        / "checkpoints"
        / "generator_state.pt"
    )

    state = torch.load(artifact, map_location="cpu", weights_only=True)

    assert state
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert not any(key.startswith("generator.") for key in state)


def test_converter_extracts_only_generator_tensors(tmp_path):
    from scripts.convert_lama_checkpoint import convert_checkpoint

    source = tmp_path / "legacy.ckpt"
    destination = tmp_path / "generator_state.pt"
    torch.save(
        {
            "state_dict": {
                "generator.layer.weight": torch.ones(1),
                "discriminator.layer.weight": torch.zeros(1),
            }
        },
        source,
    )

    converted = convert_checkpoint(source, destination)
    stored = torch.load(destination, map_location="cpu", weights_only=True)

    assert converted == 1
    assert list(stored) == ["layer.weight"]
    assert torch.equal(stored["layer.weight"], torch.ones(1))
