import logging
from PIL import Image
from simple_lama_inpainting import SimpleLama

from ...core.debug import save_inpaint_debug
from ...core.helpers import _prepare_inpaint_masks, _preserve_unmasked_pixels
from ..base import BaseInpaintingModel
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

@ModelRegistry.register("inpainting", "lama")
class LamaInpaintingModel(BaseInpaintingModel):
    def _load_model(self):
        self.generation_mask_expansion = int(
            self.config.get("generation_mask_expansion", 17)
        )
        self.composition_mask_expansion = int(
            self.config.get("composition_mask_expansion", 11)
        )
        self.feather_radius = float(self.config.get("feather_radius", 2.0))
        self.debug_dir = self.config.get("debug_dir")
        logger.info(f"[LaMa] Loading LaMa model on {self.device}...")
        self.model = SimpleLama(device=self.device)
        logger.info("[LaMa] Model loaded successfully.")

    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        debug_label: str = "",
    ) -> Image.Image:
        source = image.convert("RGB")
        binary_mask = mask.convert("L")
        generation_mask, blend_mask = _prepare_inpaint_masks(
            binary_mask,
            generation_expansion=self.generation_mask_expansion,
            composition_expansion=self.composition_mask_expansion,
            feather_radius=self.feather_radius,
        )
        result = self.model(source, generation_mask)
        composited = _preserve_unmasked_pixels(source, result, blend_mask)
        save_inpaint_debug(
            self.debug_dir,
            debug_label,
            **{
                "01_source": source,
                "02_requested_mask": binary_mask,
                "03_generation_mask": generation_mask,
                "04_raw_model_output": result,
                "05_blend_mask": blend_mask,
                "06_composited_output": composited,
            },
        )
        return composited
