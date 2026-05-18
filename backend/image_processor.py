import io
import base64
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from models import get_sam3_processor, get_lama_model, get_matting_model
from helpers import estimate_fg_color, refine_background, expand_mask

logger = logging.getLogger(__name__)

@dataclass
class ObjectLayer:
    """Một đối tượng đã tách, kèm tọa độ gốc."""
    keyword: str
    png_base64: str          # ảnh RGBA, base64 encoded
    x: int                   # tọa độ góc trên-trái trong ảnh gốc
    y: int
    width: int
    height: int

@dataclass
class ProcessResult:
    """Kết quả trả về cho frontend."""
    background_base64: str       # ảnh nền đã inpaint (JPEG/PNG), base64
    original_width: int
    original_height: int
    layers: List[ObjectLayer] = field(default_factory=list)

def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    """Tính bounding box (x, y, w, h) từ mask nhị phân."""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)

def _extract_rgba_layer(
    image_rgb: Image.Image,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Cắt đối tượng theo bbox, áp mask làm alpha channel.
    Trả về ảnh RGBA với nền trong suốt.
    """
    x, y, w, h = bbox
    # Crop ảnh gốc và mask theo bbox
    crop_rgb = image_rgb.crop((x, y, x + w, y + h))
    mask_crop = Image.fromarray(mask[y:y+h, x:x+w])

    # Chuyển sang RGBA và gán alpha = mask
    rgba = crop_rgb.convert("RGBA")
    r, g, b, _ = rgba.split()
    rgba = Image.merge("RGBA", (r, g, b, mask_crop))
    return rgba


# ──────────────────────────────────────────────
# Pipeline chính
# ──────────────────────────────────────────────

# def process_image(image: Image.Image, keywords: List[str]) -> ProcessResult:
#     orig_w, orig_h = image.size
#     processor = get_sam3_processor()
#     matting_processor = get_matting_model()
#     lama = get_lama_model()

#     all_masks = []
#     object_layers = []

#     device_type = "cuda" if torch.cuda.is_available() else "cpu"
#     mixed_precision_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

#     # Wrap trong no_grad + tắt autocast hoàn toàn
#     with torch.no_grad(), torch.autocast(device_type=device_type, dtype=mixed_precision_dtype):

#         # Đảm bảo ảnh PIL là RGB uint8 sạch
#         image = image.convert("RGB")

#         for keyword in keywords:
#             keyword = keyword.strip()
#             if not keyword:
#                 continue

#             print(f"[SAM3] Keyword: '{keyword}'")
#             processor.reset_all_prompts(inference_state)
#             inference_state = processor.set_text_prompt(
#                 state=inference_state,
#                 prompt=keyword,
#             )

#             masks  = inference_state.get("masks")   # [N, 1, H, W]
#             boxes  = inference_state.get("boxes")   # [N, 4] XYXY
#             scores = inference_state.get("scores")  # [N]

#             if masks is None or len(masks) == 0:
#                 logger.warning(f"[SAM3] Không tìm thấy object cho '{keyword}'")
#                 continue

#             nb_objects = len(masks)
#             print(f"[SAM3] Tìm thấy {nb_objects} object(s) cho '{keyword}'")

#             # Xử lý từng object detect được
#             for i in range(nb_objects):
#                 mask_tensor = masks[i]                          # [1, H, W]
#                 if hasattr(mask_tensor, "cpu"):
#                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
#                 else:
#                     mask_np = np.squeeze(mask_tensor)

#                 mask_binary = (mask_np > 0).astype(np.uint8) * 255

#                 bbox = _bbox_from_mask(mask_binary)
#                 if bbox is None:
#                     continue

#                 score = scores[i].item() if scores is not None else 1.0
#                 label = f"{keyword}_{i}" if nb_objects > 1 else keyword

#                 rgba_layer = _extract_rgba_layer(image, mask_binary, bbox)
#                 all_masks.append(mask_binary)

#                 object_layers.append(ObjectLayer(
#                     keyword=label,
#                     png_base64=_image_to_base64(rgba_layer, fmt="PNG"),
#                     x=bbox[0], y=bbox[1],
#                     width=bbox[2], height=bbox[3],
#                 ))
#                 print(f"[SAM3] Layer '{label}' score={score:.2f} bbox={bbox}")

#     lama = get_lama_model()

#     if all_masks:
#         print(f"[LaMa] Found {len(all_masks)} masks. Starting sequential inpainting...")
        
#         # Đảm bảo ảnh đầu vào là RGB (LaMa không nhận ảnh có kênh Alpha/RGBA trực tiếp)
#         current_background = image.convert("RGB")
#         with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
#             for i, m in enumerate(all_masks):
#                 print(f"[LaMa] Inpainting mask {i+1}/{len(all_masks)}...")
#                 kernel = np.ones((10, 10), np.uint8)
#                 m = cv2.dilate(m, kernel, iterations=1)
                
#                 # 1. Chuyển mask hiện tại từ numpy array sang PIL Image mode "L"
#                 mask_pil = Image.fromarray(m).convert("L")

#                 # 2. Chạy LaMa inpainting
#                 # LaMa (SimpleLama) nhận vào (PIL Image, PIL Mask) và trả về trực tiếp PIL Image
#                 current_background = lama(current_background, mask_pil)
                
#                 # 3. Resize lại để đảm bảo kích thước luôn khớp tuyệt đối với ảnh gốc
#                 if current_background.size != (orig_w, orig_h):
#                     current_background = current_background.resize((orig_w, orig_h), Image.LANCZOS)

#         background = current_background
#         print("[LaMa] All masks processed.")
#     else:
#         logger.warning("Không detect được object nào, dùng ảnh gốc làm background.")
#         background = image.copy()

#     # Giữ nguyên phần log và trả về kết quả
#     print(f"Original image size: {image.size}")           # (W, H)
#     print(f"Background size: {background.size}")          # Khớp với original
    
#     for layer in object_layers:
#         print(f"Layer '{layer.keyword}': x={layer.x}, y={layer.y}, w={layer.width}, h={layer.height}")

#     return ProcessResult(
#         background_base64=_image_to_base64(background, fmt="PNG"),
#         original_width=orig_w,
#         original_height=orig_h,
#         layers=object_layers,
#     )

def process_image(image: Image.Image, keywords: List[str]) -> ProcessResult:
    from models import get_sam3_processor, get_lama_model, get_matting_model

    orig_w, orig_h = image.size
    processor = get_sam3_processor()
    matting_processor = get_matting_model()
    lama = get_lama_model()

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # Lưu trữ thông tin object để xử lý sau
    detected_objects = [] 
    all_inpaint_masks = []

    # --- BƯỚC 1: DETECTION VỚI SAM3 ---
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=mixed_precision_dtype):
        print(f"[SAM3] Processing image ({orig_w}x{orig_h})...")
        inference_state = processor.set_image(image_rgb)

        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword: continue

            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=keyword)

            masks = inference_state.get("masks")
            scores = inference_state.get("scores")

            if masks is None or len(masks) == 0:
                continue

            for i in range(len(masks)):
                mask_tensor = masks[i]
                mask_np = mask_tensor.squeeze(0).cpu().numpy() if hasattr(mask_tensor, "cpu") else np.squeeze(mask_tensor)
                mask_binary = (mask_np > 0).astype(np.uint8) * 255
                
                bbox = _bbox_from_mask(mask_binary)
                if bbox is None: continue

                label = f"{keyword}_{i}" if len(masks) > 1 else keyword
                
                # Lưu mask thô để inpaint và metadata để trích xuất layer sau
                detected_objects.append({
                    "label": label,
                    "sam_mask": mask_binary,
                    "bbox": bbox,
                    "score": scores[i].item() if scores is not None else 1.0
                })
                
                # Tạo mask mở rộng để LaMa xóa sạch chân vật thể (Dilation)
                kernel_size = max(3, int(min(orig_w, orig_h) * 0.015)) # 1.5% kích thước ảnh
                dilated_mask = expand_mask(mask_binary > 0, kernel_size)
                all_inpaint_masks.append(dilated_mask)

    # --- BƯỚC 2: TẠO BACKGROUND (INPAINT + PALETTE REFINE) ---
    if all_inpaint_masks:
        print(f"[LaMa] Sequential inpainting {len(all_inpaint_masks)} masks...")
        current_bg_np = image_np.copy()
        
        # Tắt autocast cho LaMa để tránh lỗi BFloat16
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            for i, m in enumerate(all_inpaint_masks):
                print(f"[LaMa] Processing mask {i+1}...")
                m_pil = Image.fromarray((m * 255).astype(np.uint8)).convert("L")
                bg_pil = Image.fromarray(current_bg_np)
                
                # 1. Inpaint
                inpainted_pil = lama(bg_pil, m_pil)
                if inpainted_pil.size != (orig_w, orig_h):
                    inpainted_pil = inpainted_pil.resize((orig_w, orig_h), Image.LANCZOS)
                
                # 2. Refine Background (Kỹ thuật LayerD)
                # Giúp vùng inpaint sắc nét và đồng nhất màu sắc với xung quanh
                current_bg_np = refine_background(
                    bg=np.array(inpainted_pil),
                    mask=m.astype(bool),
                    n_outer_ratio=0.2,
                    max_num_colors=10
                )
        background_final = Image.fromarray(current_bg_np)
    else:
        background_final = image.copy()

    # --- BƯỚC 3: TRÍCH XUẤT LAYER (MATTING + UNBLENDING) ---
    object_layers = []
    
    if detected_objects:
        print(f"[Matting] Running BiRefNet once for all objects...")
        
        # 1. Chạy Matting 1 lần duy nhất cho cả tấm ảnh (Tắt autocast để tránh lỗi Half/Float)
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            # Ép chạy Float32 hoàn toàn cho Matting
            full_alpha_tensor = matting_processor(image_rgb)
            full_alpha_np = full_alpha_tensor.cpu().numpy().astype(np.float64)

        # 2. Xử lý từng object dựa trên bản đồ Alpha chung đã có
        print(f"[Unblending] Processing {len(detected_objects)} layers...")
        for obj in detected_objects:
            # Chỉ lấy alpha trong vùng SAM3 đã chỉ định cho object này
            object_alpha = full_alpha_np * (obj["sam_mask"] > 0)

            # Unblending sử dụng background sạch đã tạo ở Bước 2
            fg_rgb_np = estimate_fg_color(
                image_np=image_np,
                bg_np=np.array(background_final),
                alpha_np=object_alpha
            )

            # Tạo RGBA Layer
            rgba_np = np.dstack([fg_rgb_np, (object_alpha * 255).astype(np.uint8)])
            rgba_pil = Image.fromarray(rgba_np)
            
            b = obj["bbox"]
            cropped_layer = rgba_pil.crop((b[0], b[1], b[0] + b[2], b[1] + b[3]))

            object_layers.append(ObjectLayer(
                keyword=obj["label"],
                png_base64=_image_to_base64(cropped_layer, fmt="PNG"),
                x=b[0], y=b[1], width=b[2], height=b[3],
            ))
            print(f"[Layer] Extracted: {obj['label']}")

    return ProcessResult(
        background_base64=_image_to_base64(background_final, fmt="PNG"),
        original_width=orig_w,
        original_height=orig_h,
        layers=object_layers,
    )

