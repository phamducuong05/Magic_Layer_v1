import logging
from PIL import Image
from simple_lama_inpainting import SimpleLama

from ..base import BaseInpaintingModel
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

@ModelRegistry.register("inpainting", "lama")
class LamaInpaintingModel(BaseInpaintingModel):
    def _load_model(self):
        logger.info(f"[LaMa] Loading LaMa model on {self.device}...")
        self.model = SimpleLama(device=self.device)
        logger.info("[LaMa] Model loaded successfully.")

    def process(
        self, image: Image.Image, mask: Image.Image, prompt: str = ""
    ) -> Image.Image:
        # SimpleLama callable takes (image, mask) and returns an Image
        return self.model(image.convert("RGB"), mask.convert("L"))
