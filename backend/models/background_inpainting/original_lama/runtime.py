"""Generator-only runtime for the vendored Original LaMa source."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from ....core.logging import get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[4]
logger = get_logger(__name__)


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _lookup(config: Mapping[str, Any], reference: str) -> Any:
    value: Any = config
    for part in reference.split("."):
        value = value[part]
    return value


def _resolve_value(value: Any, config: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return _resolve_value(_lookup(config, value[2:-1]), config)
    if isinstance(value, dict):
        return {
            key: _resolve_value(child, config)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_resolve_value(child, config) for child in value]
    return value


def load_generator_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Original LaMa config not found: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(
        config.get("generator"), dict
    ):
        raise ValueError(
            f"Original LaMa config has no generator mapping: {config_path}"
        )
    return _resolve_value(config["generator"], config)


def prepare_inputs(
    image: Image.Image,
    mask: Image.Image,
    *,
    modulo: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    if modulo < 1:
        raise ValueError("pad_out_to_modulo must be at least 1")

    source = image.convert("RGB")
    binary_mask = mask.convert("L")
    if binary_mask.size != source.size:
        binary_mask = binary_mask.resize(source.size, Image.Resampling.NEAREST)

    image_array = np.asarray(source, dtype=np.float32) / 255.0
    mask_array = (
        np.asarray(binary_mask, dtype=np.uint8) > 127
    ).astype(np.float32)
    height, width = mask_array.shape
    padded_height = ((height + modulo - 1) // modulo) * modulo
    padded_width = ((width + modulo - 1) // modulo) * modulo
    pad_height = padded_height - height
    pad_width = padded_width - width

    image_chw = np.transpose(image_array, (2, 0, 1))
    image_chw = np.pad(
        image_chw,
        ((0, 0), (0, pad_height), (0, pad_width)),
        mode="symmetric",
    )
    mask_chw = np.pad(
        mask_array[None, ...],
        ((0, 0), (0, pad_height), (0, pad_width)),
        mode="symmetric",
    )
    return (
        torch.from_numpy(image_chw.copy()).unsqueeze(0),
        torch.from_numpy(mask_chw.copy()).unsqueeze(0),
        (height, width),
    )


def compose_prediction(
    prediction: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    original_size: tuple[int, int],
) -> Image.Image:
    height, width = original_size
    inpainted = mask * prediction + (1.0 - mask) * image
    array = (
        inpainted[0, :, :height, :width]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .clamp(0.0, 1.0)
        .numpy()
    )
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


class OriginalLamaRuntime:
    def __init__(
        self,
        *,
        source_root: str | Path,
        checkpoint_config_path: str | Path,
        generator_weights_path: str | Path,
        device: str,
        pad_out_to_modulo: int = 8,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Original LaMa was configured for CUDA, but CUDA is unavailable"
            )
        self.pad_out_to_modulo = int(pad_out_to_modulo)
        if self.pad_out_to_modulo < 1:
            raise ValueError("pad_out_to_modulo must be at least 1")

        source_path = resolve_project_path(source_root)
        package_path = source_path / "saicinpainting"
        if not package_path.is_dir():
            raise FileNotFoundError(
                f"Original LaMa source package not found: {package_path}"
            )
        source_string = str(source_path)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)

        generator_config = load_generator_config(checkpoint_config_path)
        kind = generator_config.pop("kind", None)
        if kind != "ffc_resnet":
            raise ValueError(
                f"Unsupported Original LaMa generator kind: {kind!r}"
            )
        try:
            from saicinpainting.training.modules.ffc import FFCResNetGenerator
        except ImportError as exc:
            raise RuntimeError(
                "Original LaMa runtime dependencies are unavailable"
            ) from exc

        weights_path = resolve_project_path(generator_weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Original LaMa generator weights not found: {weights_path}"
            )
        state = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, dict) or not state:
            raise ValueError(
                f"Original LaMa generator state is empty: {weights_path}"
            )

        self.model = FFCResNetGenerator(**generator_config)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.to(self.device)
        logger.info(
            "[OriginalLaMa] Loaded %d generator tensors from %s",
            len(state),
            weights_path,
        )

    @torch.inference_mode()
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image_tensor, mask_tensor, original_size = prepare_inputs(
            image,
            mask,
            modulo=self.pad_out_to_modulo,
        )
        image_tensor = image_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        network_input = torch.cat(
            [image_tensor * (1.0 - mask_tensor), mask_tensor],
            dim=1,
        )
        prediction = self.model(network_input)
        return compose_prediction(
            prediction,
            image_tensor,
            mask_tensor,
            original_size=original_size,
        )

    def inpaint_coarse_to_fine(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        coarse_max_side: int,
        refinement_band_pixels: int,
        coarse_callback: Callable[[Image.Image], None] | None = None,
    ) -> Image.Image:
        """Fill a large hole at low resolution, then refine its inner edge."""
        if coarse_max_side < 64:
            raise ValueError("coarse_max_side must be at least 64")
        if refinement_band_pixels < 1:
            raise ValueError(
                "refinement_band_pixels must be at least 1"
            )

        source = image.convert("RGB")
        binary_mask = mask.convert("L")
        if binary_mask.size != source.size:
            binary_mask = binary_mask.resize(
                source.size, Image.Resampling.NEAREST
            )

        width, height = source.size
        longest_side = max(width, height)
        if longest_side <= coarse_max_side:
            return self.inpaint(source, binary_mask)

        scale = coarse_max_side / longest_side
        coarse_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        coarse_source = source.resize(
            coarse_size, Image.Resampling.LANCZOS
        )
        coarse_mask = binary_mask.resize(
            coarse_size, Image.Resampling.NEAREST
        )
        coarse_result = self.inpaint(coarse_source, coarse_mask)
        coarse_upscaled = coarse_result.resize(
            source.size, Image.Resampling.LANCZOS
        )

        # Keep the original image outside the generation mask. Inside it, the
        # low-resolution result provides globally coherent context for pass 2.
        coarse_seed = Image.composite(
            coarse_upscaled,
            source,
            binary_mask,
        )
        if coarse_callback is not None:
            coarse_callback(coarse_seed)

        mask_array = np.asarray(binary_mask, dtype=np.uint8) > 127
        radius = int(refinement_band_pixels)
        kernel_size = 2 * radius + 1
        eroded = cv2.erode(
            mask_array.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        inner_boundary = mask_array & ~eroded

        # A thin mask may disappear under erosion. Re-running the complete
        # native-resolution hole would defeat coarse-to-fine, so retain the
        # coarse result in that case.
        if not np.any(eroded) or not np.any(inner_boundary):
            return coarse_seed

        refinement_mask = Image.fromarray(
            inner_boundary.astype(np.uint8) * 255,
            mode="L",
        )
        return self.inpaint(coarse_seed, refinement_mask)

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            model.to("cpu")
            del self.model
