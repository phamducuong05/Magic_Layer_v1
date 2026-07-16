from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any], device: str):
        self.config = config
        self.device = device
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass


class BaseMattingModel(BaseModel):
    @abstractmethod
    @torch.no_grad()
    def process(self, image: Image.Image) -> torch.Tensor:
        pass


class BaseBackgroundInpaintingModel(BaseModel):
    """Interface for removing objects and filling background RGB only."""

    @abstractmethod
    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
    ) -> Image.Image:
        pass


class BaseObjectReconstructionModel(BaseModel):
    """Interface for reconstructing hidden RGB belonging to one object."""

    @abstractmethod
    def reconstruct(
        self,
        image: Image.Image,
        mask: Image.Image,
        object_context: str = "",
    ) -> Image.Image:
        pass


class BaseSegmentationModel(BaseModel):
    @abstractmethod
    def get_processor(self) -> Any:
        pass


class BaseCompletionModel(BaseModel):
    """Interface for completing multiple modal masks from one source image."""

    @abstractmethod
    def complete(
        self,
        image: Image.Image,
        modal_masks: List[np.ndarray],
        bboxes: List[tuple[int, int, int, int]],
    ) -> List[np.ndarray]:
        """Return amodal masks in input order.

        ``modal_masks[i]`` and ``bboxes[i]`` must describe the same grouped
        object. Bounding boxes use ``(x, y, width, height)`` coordinates.
        """
        pass
