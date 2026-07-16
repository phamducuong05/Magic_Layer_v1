import logging
from typing import Optional

from ..config import config
from .registry import ModelRegistry
from .base import (
    BaseBackgroundInpaintingModel,
    BaseCompletionModel,
    BaseMattingModel,
    BaseObjectReconstructionModel,
    BaseSegmentationModel,
)

from .segmentation import sam3
from .matting import birefnet
from .background_inpainting import lama, sdxl
from .completion import adapter
from .object_reconstruction import adapter as object_reconstruction_adapter

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manager quản lý vòng đời của các AI Models.
    Áp dụng Singleton Pattern và Lazy Loading.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._segmentation_model: Optional[BaseSegmentationModel] = None
            self._matting_model: Optional[BaseMattingModel] = None
            self._background_inpainting_model: Optional[
                BaseBackgroundInpaintingModel
            ] = None
            self._object_reconstruction_model: Optional[
                BaseObjectReconstructionModel
            ] = None
            self._completion_model: Optional[BaseCompletionModel] = None
            self._initialized = True

    def _get_model_instance(self, category: str):
        model_config = dict(config.get_model_config(category))
        model_name = model_config.pop("name")
        
        logger.info(f"Instantiating {category} model: {model_name}")
        model_class = ModelRegistry.get_class(category, model_name)
        
        # model_config lúc này chỉ còn lại các parameters
        return model_class(config=model_config, device=config.device)

    def get_segmentation_model(self) -> BaseSegmentationModel:
        if self._segmentation_model is None:
            self._segmentation_model = self._get_model_instance("segmentation")
        return self._segmentation_model

    def get_matting_model(self) -> BaseMattingModel:
        if self._matting_model is None:
            self._matting_model = self._get_model_instance("matting")
        return self._matting_model

    def get_background_inpainting_model(self) -> BaseBackgroundInpaintingModel:
        if self._background_inpainting_model is None:
            self._background_inpainting_model = self._get_model_instance(
                "background_inpainting"
            )
        return self._background_inpainting_model

    def has_object_reconstruction_model(self) -> bool:
        """Report configuration availability without loading model weights."""
        return config.has_active_model("object_reconstruction")

    def get_object_reconstruction_model(
        self,
    ) -> Optional[BaseObjectReconstructionModel]:
        if not self.has_object_reconstruction_model():
            return None
        if self._object_reconstruction_model is None:
            self._object_reconstruction_model = self._get_model_instance(
                "object_reconstruction"
            )
        return self._object_reconstruction_model

    def get_completion_model(self) -> BaseCompletionModel:
        if self._completion_model is None:
            self._completion_model = self._get_model_instance("completion")
        return self._completion_model

    def warmup_all(self):
        """Khởi tạo tất cả các model được cấu hình là active."""
        logger.info("Warming up all active models...")
        self.get_segmentation_model()
        self.get_matting_model()
        self.get_background_inpainting_model()
        logger.info(f"Models are ready on device: {config.device}")

# Export global instance
model_manager = ModelManager()
