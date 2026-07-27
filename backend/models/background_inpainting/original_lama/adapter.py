from collections.abc import Callable

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
        self.coarse_to_fine_enabled = bool(
            self.config.get("coarse_to_fine_enabled", False)
        )
        self.coarse_to_fine_mask_area_ratio = float(
            self.config.get("coarse_to_fine_mask_area_ratio", 0.2)
        )
        self.coarse_max_side = int(
            self.config.get("coarse_max_side", 1024)
        )
        self.refinement_band_pixels = int(
            self.config.get("refinement_band_pixels", 32)
        )
        if not 0.0 <= self.coarse_to_fine_mask_area_ratio <= 1.0:
            raise ValueError(
                "coarse_to_fine_mask_area_ratio must be in [0, 1]"
            )
        if self.coarse_max_side < 64:
            raise ValueError("coarse_max_side must be at least 64")
        if self.refinement_band_pixels < 1:
            raise ValueError(
                "refinement_band_pixels must be at least 1"
            )
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
        *,
        artifact_callback: Callable[[str, Image.Image], None] | None = None,
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
        generation_array = (
            np.asarray(generation_mask, dtype=np.uint8) > 127
        )
        mask_area_ratio = float(np.mean(generation_array))
        use_coarse_to_fine = (
            self.coarse_to_fine_enabled
            and mask_area_ratio >= self.coarse_to_fine_mask_area_ratio
            and max(source.size) > self.coarse_max_side
        )
        if use_coarse_to_fine:
            logger.info(
                "[OriginalLaMa] Using coarse-to-fine for mask ratio %.3f "
                "(coarse max side=%d, refinement band=%dpx)",
                mask_area_ratio,
                self.coarse_max_side,
                self.refinement_band_pixels,
            )
            generated = self.runtime.inpaint_coarse_to_fine(
                source,
                generation_mask,
                coarse_max_side=self.coarse_max_side,
                refinement_band_pixels=self.refinement_band_pixels,
                coarse_callback=(
                    (
                        lambda coarse: artifact_callback(
                            "after_lama_coarse", coarse
                        )
                    )
                    if artifact_callback is not None
                    else None
                ),
            )
        else:
            generated = self.runtime.inpaint(source, generation_mask)
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
