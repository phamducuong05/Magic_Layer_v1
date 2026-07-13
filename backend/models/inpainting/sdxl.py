import logging
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting

from ..base import BaseInpaintingModel
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

@ModelRegistry.register("inpainting", "sdxl")
class SDXLInpaintingModel(BaseInpaintingModel):
    def _load_model(self):
        model_id = self.config.get("model_id", "runwayml/stable-diffusion-inpainting")
        use_xformers = self.config.get("use_xformers", True)
        cpu_offload = self.config.get("cpu_offload", True)

        logger.info(f"[SDXL] Loading Inpainting model on {self.device}...")
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        self.model = self.model.to(self.device)

        if self.device == "cuda":
            if use_xformers:
                self.model.enable_xformers_memory_efficient_attention()
            if cpu_offload:
                self.model.enable_model_cpu_offload()

        logger.info("[SDXL] Model loaded successfully.")

    def process(
        self, image: Image.Image, mask: Image.Image, prompt: str = ""
    ) -> Image.Image:
        result = self.model(
            prompt=prompt or "clean natural background, high quality",
            negative_prompt="object, artifact, blur, watermark",
            image=image.convert("RGB"),
            mask_image=mask.convert("L"),
        ).images[0]
        return result
