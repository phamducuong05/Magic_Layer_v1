import logging
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForImageSegmentation

from ..base import BaseMattingModel
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

@ModelRegistry.register("matting", "birefnet")
class BiRefNetMattingModel(BaseMattingModel):
    def _load_model(self):
        model_id = self.config.get("model_id", "ZhengPeng7/BiRefNet")
        trust_remote = self.config.get("trust_remote_code", True)
        res = self.config.get("resolution", [1024, 1024])

        logger.info(f"[Matting] Loading BiRefNet on {self.device}...")
        model = AutoModelForImageSegmentation.from_pretrained(
            model_id, 
            trust_remote_code=trust_remote
        )
        model.to(self.device)
        model.float()
        model.eval()

        self.model = model
        self.transform = T.Compose([
            T.Resize(tuple(res)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        logger.info("[Matting] Model loaded.")

    @torch.no_grad()
    def process(self, image: Image.Image) -> torch.Tensor:
        orig_size = image.size[::-1] # (H, W)
        
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        input_tensor = input_tensor.to(torch.float32) 
        
        preds = self.model(input_tensor)[-1].sigmoid().cpu()
        alpha = torch.nn.functional.interpolate(preds, size=orig_size, mode='bilinear', align_corners=False)
        return alpha[0, 0]
