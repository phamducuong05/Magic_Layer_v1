import logging
from typing import Any, Dict
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from ..base import BaseSegmentationModel
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

@ModelRegistry.register("segmentation", "sam3")
class Sam3SegmentationModel(BaseSegmentationModel):
    def _load_model(self):
        checkpoint_path = self.config.get("checkpoint_path", "checkpoints/sam3.pt")
        confidence = float(self.config.get("confidence_threshold", 0.5))
        load_hf = self.config.get("load_from_hf", False)

        logger.info(f"[SAM3] Loading model on {self.device}...")
        # Resolve local checkpoints relative to the repository, independent of cwd.
        if checkpoint_path and not load_hf:
            from pathlib import Path

            path = Path(checkpoint_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[3] / path
            checkpoint_path = str(path)

        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=load_hf,
        )
        model = model.to(self.device)
        model.eval()

        self._processor = Sam3Processor(
            model,
            confidence_threshold=confidence,
        )
        logger.info("[SAM3] Model loaded.")

    def get_processor(self) -> Any:
        return self._processor
