from abc import ABC, abstractmethod
from typing import Any, Dict
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


class BaseInpaintingModel(BaseModel):
    @abstractmethod
    def process(
        self, image: Image.Image, mask: Image.Image, prompt: str = ""
    ) -> Image.Image:
        pass


class BaseSegmentationModel(BaseModel):
    @abstractmethod
    def get_processor(self) -> Any:
        pass
