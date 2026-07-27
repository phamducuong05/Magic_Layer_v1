from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
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

    def unload(self) -> None:
        """Drop model-owned references before clearing accelerator caches.

        Adapters with module-level caches or mutable runtime state should
        override this method, clean those resources, then call ``super()``.
        """
        self.__dict__.clear()


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
        *,
        artifact_callback: Callable[[str, Image.Image], None] | None = None,
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

    def reconstruct_many(
        self,
        requests: Sequence[tuple[Image.Image, Image.Image, str]],
    ) -> list[Image.Image | Exception]:
        """Reconstruct requests in order while isolating individual errors."""
        outcomes: list[Image.Image | Exception] = []
        for image, mask, object_context in requests:
            try:
                outcomes.append(self.reconstruct(image, mask, object_context))
            except Exception as exc:
                outcomes.append(exc)
        return outcomes


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
