import numpy as np
from PIL import Image

from ....core.logging import get_logger
from ...base import BaseBackgroundInpaintingModel
from ...registry import ModelRegistry
from ..common import prepare_inpaint_masks, preserve_unmasked_pixels
from .runtime import OriginalLamaRuntime

logger = get_logger(__name__)


@ModelRegistry.register("background_inpainting", "original_lama")
class OriginalLamaBackgroundInpaintingModel(BaseBackgroundInpaintingModel):
    """Background inpainter backed by the vendored Original LaMa generator."""

    def _load_model(self):
        self.generation_mask_expansion = int(
            self.config.get("generation_mask_expansion", 17)
        )
        self.composition_mask_expansion = int(
            self.config.get("composition_mask_expansion", 11)
        )
        self.feather_radius = float(self.config.get("feather_radius", 2.0))
        logger.info(
            "[OriginalLaMa] Loading background model on %s...", self.device
        )
        self.runtime = OriginalLamaRuntime(
            source_root=self.config["source_root"],
            checkpoint_config_path=self.config["checkpoint_config_path"],
            generator_weights_path=self.config["generator_weights_path"],
            device=self.device,
            pad_out_to_modulo=int(
                self.config.get("pad_out_to_modulo", 8)
            ),
        )
        logger.info("[OriginalLaMa] Background model loaded successfully.")

    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
    ) -> Image.Image:
        del prompt
        source = image.convert("RGB")
        binary_mask = mask.convert("L")
        if binary_mask.size != source.size:
            binary_mask = binary_mask.resize(
                source.size, Image.Resampling.NEAREST
            )
        if not np.any(np.asarray(binary_mask, dtype=np.uint8) > 127):
            return source.copy()

        generation_mask, blend_mask = prepare_inpaint_masks(
            binary_mask,
            generation_expansion=self.generation_mask_expansion,
            composition_expansion=self.composition_mask_expansion,
            feather_radius=self.feather_radius,
        )
        generated = self.runtime.inpaint(source, generation_mask)
        return preserve_unmasked_pixels(source, generated, blend_mask)

    def unload(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.close()
        super().unload()
