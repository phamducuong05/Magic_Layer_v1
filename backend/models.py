import os
import logging

import torch

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import torchvision.transforms as T
# pip install diffusers transformers accelerate
from diffusers import AutoPipelineForInpainting
from dotenv import load_dotenv
from simple_lama_inpainting import SimpleLama
from transformers import AutoModelForImageSegmentation

load_dotenv()

logger = logging.getLogger(__name__)

SAM3_CHECKPOINT = os.getenv("SAM3_CHECKPOINT", "checkpoints/sam3.pt")
SAM3_CONFIDENCE = float(os.getenv("SAM3_CONFIDENCE", "0.5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATTING_MODEL_ID = "ZhengPeng7/BiRefNet"
# SDXL Inpainting model
INPAINT_MODEL_ID = os.getenv(
    "INPAINT_MODEL_ID",
    "runwayml/stable-diffusion-inpainting",
)

_sam3_processor: Sam3Processor | None = None

def get_sam3_processor():
    global _sam3_processor
    if _sam3_processor is None:
        logger.info(f"[SAM3] Loading model on {DEVICE}...")
        model = build_sam3_image_model(
            checkpoint_path=SAM3_CHECKPOINT,
            load_from_HF=False,
        )

        model = model.to(DEVICE)

        model.eval()
        _sam3_processor = Sam3Processor(
            model,
            confidence_threshold=SAM3_CONFIDENCE,
        )
        logger.info("[SAM3] Model loaded.")
    return _sam3_processor

# _inpaint_model: AutoPipelineForInpainting | None = None

# def get_lama_model() -> AutoPipelineForInpainting:
#     global _inpaint_model

#     if _inpaint_model is None:
#         logger.info(f"[Inpainting] Loading model on {DEVICE}...")

#         dtype = (
#             torch.float16
#             if DEVICE == "cuda"
#             else torch.float32
#         )

#         _inpaint_model = AutoPipelineForInpainting.from_pretrained(
#             INPAINT_MODEL_ID,
#             torch_dtype=dtype,
#         )

#         _inpaint_model = _inpaint_model.to(DEVICE)

#         if DEVICE == "cuda":
#             # Kích hoạt xformers để tăng tốc độ gen và giảm thiểu VRAM 
#             _inpaint_model.enable_xformers_memory_efficient_attention()
            
#             # Nếu server RAM yếu, bật tính năng offload giúp chuyển model qua lại giữa CPU và GPU
#             _inpaint_model.enable_model_cpu_offload()

#         logger.info("[Inpainting] Model loaded successfully.")

#     return _inpaint_model


_lama_model = None

def get_lama_model():
    global _lama_model
    if _lama_model is None:
        logger.info(f"[LaMa] Loading LaMa model on {DEVICE}...")
        # SimpleLama tự động tải weight và quản lý device
        _lama_model = SimpleLama(device=DEVICE)
        logger.info("[LaMa] Model loaded successfully.")
    return _lama_model


_matting_model = None

class MattingProcessor:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        # BiRefNet chuẩn hóa theo thông số ImageNet
        self.transform = T.Compose([
            T.Resize((1024, 1024)), # Kích thước chuẩn của BiRefNet
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Trả về alpha mask dạng Tensor [H, W] giá trị 0.0 - 1.0"""
        orig_size = image.size[::-1] # (H, W)
        
        # Tiền xử lý
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        input_tensor = input_tensor.to(torch.float32) 
        # Inference
        # BiRefNet trả về một list các outputs, lấy cái cuối cùng là cái chi tiết nhất
        preds = self.model(input_tensor)[-1].sigmoid().cpu()
        
        # Resize lại về kích thước ảnh gốc
        alpha = torch.nn.functional.interpolate(preds, size=orig_size, mode='bilinear', align_corners=False)
        return alpha[0, 0] # Trả về [H, W]

def get_matting_model():
    global _matting_model
    if _matting_model is None:
        print(f"[Matting] Loading BiRefNet on {DEVICE}...")
        # trust_remote_code=True là bắt buộc vì BiRefNet sử dụng kiến trúc custom trên HF
        model = AutoModelForImageSegmentation.from_pretrained(
            MATTING_MODEL_ID, 
            trust_remote_code=True
        )
        model.to(DEVICE)
        model.float()
        model.eval()
        _matting_model = MattingProcessor(model)
        print("[Matting] Model loaded.")
    return _matting_model

def warmup_models():
    logger.info("Warming up models...")

    get_sam3_processor()
    get_lama_model()
    get_matting_model()

    logger.info(f"Models ready on device: {DEVICE}")