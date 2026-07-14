"""Reusable, in-memory DIFT feature extraction for amodal completion."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image


REQUIRED_FEATURE_LEVELS = frozenset(range(4))


class DIFTFeatureExtractor:
    """Extract the four DIFT feature levels needed by SDAmodal."""

    def __init__(
        self,
        featurizer: Any = None,
        *,
        model_id: str = "sd2-community/stable-diffusion-2-1",
        device: str | torch.device = "cuda",
        image_size: tuple[int, int] | None = (768, 768),
        prompt: str = "",
        timestep: int = 181,
        up_ft_index: int = 1,
        ensemble_size: int = 2,
    ) -> None:
        self.device = torch.device(device)
        self.image_size = image_size
        self.prompt = prompt
        self.timestep = timestep
        self.up_ft_index = up_ft_index
        self.ensemble_size = ensemble_size
        self.featurizer = (
            featurizer
            if featurizer is not None
            else self._create_featurizer(model_id)
        )

    def _create_featurizer(self, model_id: str) -> Any:
        from .src.models.dift_sd import SDFeaturizer

        return SDFeaturizer(model_id, device=self.device)

    def extract(self, image: Image.Image) -> dict[int, torch.Tensor]:
        """Return levels 0-3 as detached CPU tensors shaped ``(C, H, W)``."""
        prepared_image = image.convert("RGB")
        if self.image_size is not None:
            prepared_image = prepared_image.resize(
                self.image_size,
                Image.Resampling.BILINEAR,
            )

        image_array = np.asarray(prepared_image, dtype=np.float32).copy()
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = image_tensor.div(127.5).sub(1.0)

        with torch.no_grad():
            raw_features = self.featurizer.forward(
                image_tensor,
                prompt=self.prompt,
                t=self.timestep,
                up_ft_index=self.up_ft_index,
                ensemble_size=self.ensemble_size,
            )

        return self._prepare_feature_pyramid(raw_features)

    @staticmethod
    def _prepare_feature_pyramid(
        raw_features: Mapping[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        missing_levels = sorted(REQUIRED_FEATURE_LEVELS - set(raw_features))
        if missing_levels:
            missing = ", ".join(str(level) for level in missing_levels)
            raise ValueError(f"missing DIFT feature levels: {missing}")

        feature_pyramid = {}
        for level in sorted(REQUIRED_FEATURE_LEVELS):
            feature = raw_features[level]
            if not isinstance(feature, torch.Tensor):
                raise ValueError(f"DIFT feature level {level} must be a tensor")
            if feature.ndim != 4 or feature.shape[0] != 1:
                raise ValueError(
                    f"DIFT feature level {level} must have shape (1, C, H, W)"
                )
            feature_pyramid[level] = feature.squeeze(0).detach().cpu()

        return feature_pyramid
