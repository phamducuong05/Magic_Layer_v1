"""Registered application adapter for in-memory SDAmodal completion."""

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from ..base import BaseCompletionModel
from ..registry import ModelRegistry
from .batch_completion import complete_masks_from_features
from .dift.extract_dift_amodal import DIFTFeatureExtractor
from .model_loader import load_sdamodal_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@ModelRegistry.register("completion", "sdamodal")
class SDAmodalCompletionModel(BaseCompletionModel):
    """Complete a batch of modal masks using one shared DIFT extraction."""

    def _load_model(self) -> None:
        config_path = _project_path(
            self.config.get(
                "config_path",
                "backend/models/completion/config_SDAmodal.yaml",
            )
        )
        checkpoint_path = _project_path(
            self.config.get(
                "checkpoint_path",
                "checkpoints/ckpt_SDAmodal.pth",
            )
        )
        self.model, self.sdamodal_config = load_sdamodal_model(
            config_path,
            checkpoint_path,
            device=self.device,
        )
        self.feature_extractor = DIFTFeatureExtractor(device=self.device)

    def complete(
        self,
        image: Image.Image,
        modal_masks: List[np.ndarray],
        bboxes: List[tuple[int, int, int, int]],
    ) -> List[np.ndarray]:
        if not modal_masks:
            return []

        feature_pyramid = self.feature_extractor.extract(image)
        return complete_masks_from_features(
            self.model,
            feature_pyramid,
            modal_masks,
            bboxes,
            image_shape=(image.height, image.width),
            config=self.sdamodal_config,
            device=self.device,
        )
