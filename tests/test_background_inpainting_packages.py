from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_background_inpainters_are_split_into_strategy_packages():
    from backend.models.background_inpainting.simple_lama.adapter import (
        LamaBackgroundInpaintingModel,
    )
    from backend.models.background_inpainting.sdxl.adapter import (
        SDXLBackgroundInpaintingModel,
    )
    from backend.models.background_inpainting.lama import (
        LamaBackgroundInpaintingModel as LegacyLama,
    )
    from backend.models.background_inpainting.sdxl import (
        SDXLBackgroundInpaintingModel as LegacySDXL,
    )

    assert LegacyLama is LamaBackgroundInpaintingModel
    assert LegacySDXL is SDXLBackgroundInpaintingModel


def test_background_mask_helpers_are_owned_by_common_package():
    from backend.core.helpers import (
        _prepare_inpaint_masks as legacy_prepare,
        _preserve_unmasked_pixels as legacy_preserve,
    )
    from backend.models.background_inpainting.common.masks import (
        prepare_inpaint_masks,
        preserve_unmasked_pixels,
    )

    assert legacy_prepare is prepare_inpaint_masks
    assert legacy_preserve is preserve_unmasked_pixels


def test_original_lama_config_uses_refactored_checkpoint_directory():
    with (ROOT / "backend" / "config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)

    background = config["models"]["background_inpainting"]
    original = background["original_lama"]

    assert background["active"] == "original_lama"
    assert original["source_root"] == "lama"
    assert original["checkpoint_config_path"] == (
        "backend/models/background_inpainting/original_lama/"
        "checkpoints/config.yaml"
    )
    assert original["generator_weights_path"] == (
        "backend/models/background_inpainting/original_lama/"
        "checkpoints/generator_state.pt"
    )
    assert original["legacy_checkpoint_path"] == (
        "backend/models/background_inpainting/original_lama/"
        "checkpoints/best.ckpt"
    )


def test_big_lama_weights_live_with_original_lama_strategy():
    checkpoint_dir = (
        ROOT
        / "backend"
        / "models"
        / "background_inpainting"
        / "original_lama"
        / "checkpoints"
    )

    assert (checkpoint_dir / "config.yaml").is_file()
    assert (checkpoint_dir / "best.ckpt").is_file()
    assert not (ROOT / "backend" / "big-lama").exists()


def test_model_manager_import_registers_smarteraser():
    script = """
import sys
import types

for name in (
    "backend.models.segmentation.sam3",
    "backend.models.matting.birefnet",
    "backend.models.background_inpainting.lama",
    "backend.models.background_inpainting.original_lama",
    "backend.models.background_inpainting.sdxl",
    "backend.models.completion.adapter",
    "backend.models.object_reconstruction.adapter",
):
    sys.modules[name] = types.ModuleType(name)

object_reconstruction = types.ModuleType(
    "backend.models.object_reconstruction"
)
object_reconstruction.adapter = sys.modules[
    "backend.models.object_reconstruction.adapter"
]
sys.modules[
    "backend.models.object_reconstruction"
] = object_reconstruction

from backend.models.manager import ModelManager
from backend.models.registry import ModelRegistry

print(
    ModelRegistry.get_class(
        "background_inpainting",
        "smarteraser",
    ).__name__
)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "SmartEraserBackgroundInpaintingModel" in result.stdout
    )


def test_smarteraser_config_is_local_only_and_lama_stays_default():
    with (ROOT / "backend" / "config.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    background = config["models"]["background_inpainting"]
    smart = background["smarteraser"]

    assert background["active"] == "original_lama"
    assert smart["checkpoint_dir"] == (
        "SmartEraser/Model_framework/ckpts/smarteraser-weights"
    )
    assert smart["clip_dir"] == (
        "SmartEraser/Model_framework/ckpts/clip-vit-large-patch14"
    )
    assert "model_id" not in smart
    assert "download" not in smart
