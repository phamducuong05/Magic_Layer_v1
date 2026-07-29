"""Project adapter for SmartEraser background inpainting."""

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ....core.logging import get_logger
from ...base import BaseBackgroundInpaintingModel
from ...registry import ModelRegistry
from ..common import prepare_inpaint_masks, preserve_unmasked_pixels
from .geometry import prepare_inputs, restore_output


logger = get_logger(__name__)


class _Runtime(Protocol):
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image: ...

    def close(self) -> None: ...


def _create_runtime(**kwargs: Any) -> _Runtime:
    """Import heavyweight SmartEraser dependencies only when selected."""

    from .runtime import SmartEraserRuntime

    return SmartEraserRuntime(**kwargs)


@ModelRegistry.register("background_inpainting", "smarteraser")
class SmartEraserBackgroundInpaintingModel(BaseBackgroundInpaintingModel):
    """Background inpainter backed by local SmartEraser weights."""

    def _load_model(self) -> None:
        self.resolution = int(self.config.get("resolution", 512))
        if self.resolution <= 0:
            raise ValueError("SmartEraser resolution must be positive")
        self.generation_mask_expansion = int(
            self.config.get("generation_mask_expansion", 1)
        )
        self.composition_mask_expansion = int(
            self.config.get("composition_mask_expansion", 1)
        )
        self.feather_radius = float(self.config.get("feather_radius", 5.0))

        logger.info(
            "[SmartEraser] Loading background model on %s...",
            self.device,
        )
        self.runtime = _create_runtime(
            checkpoint_dir=self.config["checkpoint_dir"],
            clip_dir=self.config["clip_dir"],
            device=self.device,
            dtype=str(self.config.get("dtype", "float16")),
            num_inference_steps=int(
                self.config.get("num_inference_steps", 50)
            ),
            guidance_scale=float(self.config.get("guidance_scale", 1.5)),
            seed=int(self.config.get("seed", 42)),
            prompt=str(
                self.config.get(
                    "prompt",
                    "Remove the instance of object",
                )
            ),
            negative_prompt=str(self.config.get("negative_prompt", "")),
        )
        logger.info("[SmartEraser] Background model loaded successfully.")

    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        *,
        artifact_callback: Callable[[str, Image.Image], None] | None = None,
    ) -> Image.Image:
        """Remove masked content while preserving all uncomposed pixels."""

        del prompt
        source = image.convert("RGB")
        binary_mask = mask.convert("L")
        if binary_mask.size != source.size:
            binary_mask = binary_mask.resize(
                source.size,
                Image.Resampling.NEAREST,
            )
        binary_mask = binary_mask.point(
            lambda value: 255 if value > 127 else 0
        )
        if not np.any(np.asarray(binary_mask, dtype=np.uint8)):
            return source.copy()

        generation_mask, blend_mask = prepare_inpaint_masks(
            binary_mask,
            generation_expansion=self.generation_mask_expansion,
            composition_expansion=self.composition_mask_expansion,
            feather_radius=self.feather_radius,
        )
        prepared = prepare_inputs(
            source,
            generation_mask,
            self.resolution,
        )
        generated_square = self.runtime.inpaint(
            prepared.image,
            prepared.mask,
        )
        generated = restore_output(
            generated_square,
            source,
            prepared.metadata,
        )
        if artifact_callback is not None:
            artifact_callback("after_lama", generated)
        composed = preserve_unmasked_pixels(source, generated, blend_mask)
        if artifact_callback is not None:
            artifact_callback("after_composition_blend", composed)
        return composed

    def unload(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.close()
        super().unload()
