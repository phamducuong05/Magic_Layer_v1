import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

from ...core.helpers import _prepare_inpaint_masks, _preserve_unmasked_pixels
from ...core.logging import get_logger
from ..base import BaseBackgroundInpaintingModel
from ..registry import ModelRegistry

logger = get_logger(__name__)


@ModelRegistry.register("background_inpainting", "sdxl")
class SDXLBackgroundInpaintingModel(BaseBackgroundInpaintingModel):
    """Remove objects and generate background-only content with SDXL."""

    def _load_model(self):
        model_id = self.config.get(
            "model_id", "runwayml/stable-diffusion-inpainting"
        )
        use_xformers = self.config.get("use_xformers", True)
        cpu_offload = self.config.get("cpu_offload", True)
        self.strength = float(self.config.get("strength", 0.75))
        self.num_inference_steps = int(
            self.config.get("num_inference_steps", 30)
        )
        self.guidance_scale = float(self.config.get("guidance_scale", 5.0))
        self.generation_mask_expansion = int(
            self.config.get("generation_mask_expansion", 17)
        )
        self.composition_mask_expansion = int(
            self.config.get("composition_mask_expansion", 11)
        )
        self.feather_radius = float(self.config.get("feather_radius", 2.0))
        self.default_prompt = self.config.get(
            "prompt",
            "empty seamless continuation of the existing background, matching "
            "the surrounding texture, color, lighting, and perspective",
        )
        self.negative_prompt = self.config.get(
            "negative_prompt",
            "object, person, animal, furniture, decoration, text, watermark, "
            "new detail, artifact, blur, distorted pattern",
        )

        logger.info("[SDXL] Loading background model on %s...", self.device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        if self.device == "cuda":
            if cpu_offload:
                self.model.enable_model_cpu_offload()
            else:
                self.model = self.model.to("cuda")
            if use_xformers:
                self.model.enable_xformers_memory_efficient_attention()
        else:
            self.model = self.model.to("cpu")

        logger.info("[SDXL] Background model loaded successfully.")

    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
    ) -> Image.Image:
        source = image.convert("RGB")
        generation_mask, blend_mask = _prepare_inpaint_masks(
            mask.convert("L"),
            generation_expansion=self.generation_mask_expansion,
            composition_expansion=self.composition_mask_expansion,
            feather_radius=self.feather_radius,
        )
        result = self.model(
            prompt=prompt or self.default_prompt,
            negative_prompt=self.negative_prompt,
            image=source,
            mask_image=generation_mask,
            strength=self.strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images[0]
        return _preserve_unmasked_pixels(source, result, blend_mask)
