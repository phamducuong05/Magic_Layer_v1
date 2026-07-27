from PIL import Image
from simple_lama_inpainting import SimpleLama

from ....core.logging import get_logger
from ...base import BaseBackgroundInpaintingModel
from ...registry import ModelRegistry
from ..common import prepare_inpaint_masks, preserve_unmasked_pixels

logger = get_logger(__name__)


@ModelRegistry.register("background_inpainting", "lama")
class LamaBackgroundInpaintingModel(BaseBackgroundInpaintingModel):
    """Remove objects and extend surrounding background with SimpleLama."""

    def _load_model(self):
        self.generation_mask_expansion = int(
            self.config.get("generation_mask_expansion", 17)
        )
        self.composition_mask_expansion = int(
            self.config.get("composition_mask_expansion", 11)
        )
        self.feather_radius = float(self.config.get("feather_radius", 2.0))
        logger.info("[SimpleLama] Loading background model on %s...", self.device)
        self.model = SimpleLama(device=self.device)
        logger.info("[SimpleLama] Background model loaded successfully.")

    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
    ) -> Image.Image:
        del prompt
        source = image.convert("RGB")
        generation_mask, blend_mask = prepare_inpaint_masks(
            mask.convert("L"),
            generation_expansion=self.generation_mask_expansion,
            composition_expansion=self.composition_mask_expansion,
            feather_radius=self.feather_radius,
        )
        result = self.model(source, generation_mask)
        return preserve_unmasked_pixels(source, result, blend_mask)
