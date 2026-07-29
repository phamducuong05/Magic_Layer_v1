"""Local-only SmartEraser model loading and inference."""

from pathlib import Path
from typing import Any

from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPTokenizer

from SmartEraser.Model_framework.modules.clip_visual_token import (
    CLIPVisualPrompt,
)
from SmartEraser.Model_framework.modules.pipeline.pipeline_stable_diffusion_inpaint_region import (
    StableDiffusionInpaintRegionPipeline,
)

from .geometry import build_guidance_crop


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_local_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _REPOSITORY_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"SmartEraser {label} directory was not found: {resolved}"
        )
    return resolved


class SmartEraserRuntime:
    """Own SmartEraser inference resources for one configured device."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        clip_dir: str | Path,
        device: str,
        dtype: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        prompt: str,
        negative_prompt: str,
    ) -> None:
        self.checkpoint_dir = _resolve_local_directory(
            checkpoint_dir,
            "checkpoint",
        )
        self.clip_dir = _resolve_local_directory(clip_dir, "CLIP")
        self.mlp_weight_path = (
            self.checkpoint_dir / "clip_mlp_weight.pth"
        )
        if not self.mlp_weight_path.is_file():
            raise FileNotFoundError(
                "SmartEraser clip_mlp_weight.pth was not found: "
                f"{self.mlp_weight_path}"
            )
        if dtype not in _DTYPES:
            raise ValueError(
                "SmartEraser dtype must be float16, bfloat16, or float32"
            )
        if num_inference_steps <= 0:
            raise ValueError(
                "SmartEraser num_inference_steps must be positive"
            )
        if guidance_scale <= 0:
            raise ValueError("SmartEraser guidance_scale must be positive")

        self.device = device
        self.weight_dtype = (
            torch.float32 if device == "cpu" else _DTYPES[dtype]
        )
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.prompt = str(prompt)
        self.negative_prompt = str(negative_prompt)
        self.pipeline: Any = None
        self.clip_model: Any = None
        self.clip_processor: Any = None
        self.tokenizer: Any = None
        self._load()

    def _load(self) -> None:
        checkpoint_path = str(self.checkpoint_dir)
        clip_path = str(self.clip_dir)
        self.pipeline = (
            StableDiffusionInpaintRegionPipeline.from_pretrained(
                checkpoint_path,
                torch_dtype=self.weight_dtype,
                local_files_only=True,
            ).to(self.device)
        )
        self.clip_model = CLIPVisualPrompt(clip_path)
        self.clip_model.load_mlp_weight(str(self.mlp_weight_path))
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            clip_path,
            local_files_only=True,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            clip_path,
            local_files_only=True,
        )
        for module in (
            self.clip_model.vision_model,
            self.clip_model.text_model,
            self.clip_model.clip_mlp,
        ):
            module.eval().to(
                device=self.device,
                dtype=self.weight_dtype,
            )

    def _tokenize(self, text: str) -> torch.Tensor:
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            max_length=7,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized["input_ids"].to(device=self.device)

    @torch.inference_mode()
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        """Run one SmartEraser inference request."""

        guidance_image = build_guidance_crop(image, mask)
        clip_pixels = self.clip_processor(
            images=guidance_image,
            return_tensors="pt",
        )["pixel_values"].to(
            device=self.device,
            dtype=self.weight_dtype,
        )
        prompt_ids = self._tokenize(self.prompt)
        negative_ids = self._tokenize(self.negative_prompt)
        prompt_embeds, negative_prompt_embeds = (
            self.clip_model.inference_vtoken(
                prompt_ids,
                negative_ids,
                clip_pixels,
                self.pipeline.text_encoder,
            )
        )
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        output = self.pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            mask_image=mask,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        return output.images[0].convert("RGB")

    def close(self) -> None:
        """Release references; ModelManager performs accelerator cleanup."""

        self.pipeline = None
        self.clip_model = None
        self.clip_processor = None
        self.tokenizer = None
