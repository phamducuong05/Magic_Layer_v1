import io, base64, logging
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
from models import get_sam3_processor, get_lama_model, get_matting_model
from helpers import (
    estimate_fg_color, estimate_fg_alpha,
    refine_background, expand_mask,
    find_flat_color_region_ccs, shrink_mask_ratio, expand_mask_ratio,
    divide_mask_to_connected_components,
)

logger = logging.getLogger(__name__)

# ── Hyperparams (mirror LayerD) ──────────────────────────────
_TH_ALPHA            = 0.005
_KERNEL_SCALE        = 0.015
_UNBLEND_ALPHA_CLIP  = [0, 0.95]
_PALETTE_PERCENTILE  = 0.99
_FG_REFINE_NUM_COLORS   = 2
_BG_REFINE_NUM_COLORS   = 10
_FG_REFINE_N_INNER_RATIO = 0.1
_BG_REFINE_N_OUTER_RATIO = 0.2


@dataclass
class ObjectLayer:
    keyword: str
    png_base64: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class ProcessResult:
    background_base64: str
    original_width: int
    original_height: int
    layers: List[ObjectLayer] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────

def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int,int,int,int] | None:
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(x0), int(y0), int(x1-x0+1), int(y1-y0+1)


def _calc_kernel_size(image_np: np.ndarray) -> tuple[int,int]:
    h, w = image_np.shape[:2]
    return (round(h * _KERNEL_SCALE), round(w * _KERNEL_SCALE))


def _build_inpaint_mask(
    image_rgb: np.ndarray,
    hard_mask: np.ndarray,           # bool (H,W)
    kernel_size: tuple[int,int],
) -> np.ndarray:
    """
    Tạo inpaint_mask theo chiến lược FG-refine của LayerD:
    shrink CC → expand(shrinked ∪ color_masks).
    Nếu không tìm được flat color region, fallback về expand(hard_mask).
    """
    color_masks, _, ccs = find_flat_color_region_ccs(
        image_rgb, hard_mask,
        max_num_colors=_FG_REFINE_NUM_COLORS,
        percentile=_PALETTE_PERCENTILE,
    )
    if len(ccs) == 0:
        return expand_mask(hard_mask, kernel_size)

    shrinked_ccs = [
        ccs[i] if len(color_masks[i]) == 0
        else shrink_mask_ratio(ccs[i], _FG_REFINE_N_INNER_RATIO)
        for i in range(len(ccs))
    ]
    combined = np.any(shrinked_ccs + sum(color_masks, []), axis=0)
    return expand_mask(combined, kernel_size)


def _refine_alpha_with_colors(
    image_rgb: np.ndarray,
    bg_np:     np.ndarray,
    alpha:     np.ndarray,           # float64 (H,W) in [0,1]
    hard_mask: np.ndarray,           # bool
    kernel_size: tuple[int,int],
) -> tuple[np.ndarray, np.ndarray]:  # refined alpha, refined fg_rgb
    """
    Per-color alpha refinement (port trực tiếp từ LayerD._decompose_step).
    """
    image_uint8 = (image_rgb).astype(np.uint8)
    alpha = alpha.astype(np.float64)
    fg_rgb = estimate_fg_color(image_uint8, bg_np.astype(np.uint8),
                               alpha, _UNBLEND_ALPHA_CLIP)

    color_masks, colors, ccs = find_flat_color_region_ccs(
        image_uint8, hard_mask,
        max_num_colors=_FG_REFINE_NUM_COLORS,
        percentile=_PALETTE_PERCENTILE,
    )

    for colors_cc, color_masks_cc, cc in zip(colors, color_masks, ccs):
        _ref_alpha  = np.zeros_like(alpha)
        _ref_color  = np.zeros_like(fg_rgb)
        _nonzero    = np.zeros_like(alpha)

        for color, cmask in zip(colors_cc, color_masks_cc):
            cmask_exp = expand_mask(cmask, kernel_size)
            ra = estimate_fg_alpha(cmask_exp, color, bg_np.astype(np.uint8), image_uint8)
            if ra is not None:
                _ref_alpha = np.maximum(_ref_alpha, ra)
                _ref_color[ra > 0] = color
                _nonzero += (ra > 0).astype(int)

        boundary = _nonzero > 1
        if _ref_alpha.sum() > 0:
            inner_cc  = (~shrink_mask_ratio(cc, _FG_REFINE_N_INNER_RATIO)) & cc
            target    = ((alpha == 0) | inner_cc) & (~boundary)
            alpha[target]    = np.maximum(alpha[target], _ref_alpha[target])
            upd = target & (_ref_alpha > 0)
            fg_rgb[upd] = _ref_color[upd]

    return alpha, fg_rgb


# ── Overlap helpers ───────────────────────────────────────────

def _build_cumulative_inpaint_mask(
    masks:        List[np.ndarray],  # list of (H,W) uint8 0/255
    target_idx:   int,
    overlap_only: bool = True,
) -> np.ndarray:
    """
    Tạo mask inpaint cho vật thể `target_idx`:
    - mask của chính nó
    - union với các mask của vật thể khác ĐÈ LÊN nó
      (overlap_only=True → chỉ lấy phần giao; False → toàn bộ mask kia)
    Trả về mask uint8 0/255.
    """
    target = masks[target_idx].astype(bool)
    combined = target.copy()

    for i, m in enumerate(masks):
        if i == target_idx:
            continue
        other = m.astype(bool)
        if not np.any(other & target):   # không đè lên target → bỏ qua
            continue
        combined |= (other & target) if overlap_only else other

    return combined.astype(np.uint8) * 255


# ── Pipeline chính ────────────────────────────────────────────

def process_image(image: Image.Image, keywords: List[str]) -> ProcessResult:
    orig_w, orig_h = image.size
    image = image.convert("RGB")
    image_np = np.array(image, dtype=np.uint8)   # (H,W,3) uint8, dùng xuyên suốt

    processor       = get_sam3_processor()
    matting_model   = get_matting_model()
    lama            = get_lama_model()
    kernel_size     = _calc_kernel_size(image_np)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_dtype = (torch.bfloat16
                   if device_type == "cuda" and torch.cuda.is_bf16_supported()
                   else torch.float16)

    # ── 1. SAM3: thu thập tất cả mask ──────────────────────────
    raw_masks: List[np.ndarray] = []   # uint8 0/255, (H,W)
    labels:    List[str]        = []

    with torch.no_grad(), torch.autocast(device_type, dtype=mixed_dtype):
        inference_state = processor.set_image(image)
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue

            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=keyword)

            masks  = inference_state.get("masks")
            scores = inference_state.get("scores")
            if masks is None or len(masks) == 0:
                logger.warning(f"[SAM3] Không tìm thấy object cho '{keyword}'")
                continue

            for i, mt in enumerate(masks):
                mn = mt.squeeze(0)
                if hasattr(mn, "cpu"):
                    mn = mn.cpu().numpy()
                raw_masks.append((mn > 0).astype(np.uint8) * 255)
                lbl = f"{keyword}_{i}" if len(masks) > 1 else keyword
                labels.append(lbl)
                score = scores[i].item() if scores is not None else 1.0
                logger.info(f"[SAM3] '{lbl}' score={score:.2f}")

    if not raw_masks:
        logger.warning("Không detect được object nào.")
        return ProcessResult(
            background_base64=_image_to_base64(image),
            original_width=orig_w, original_height=orig_h,
        )

    # ── 2. Matting: lấy soft alpha cho từng mask ───────────────
    # Matting chạy trên toàn ảnh; SAM3 mask dùng để crop ROI trước
    # để tránh nhầm với các vật thể khác → guided matting
    soft_alphas: List[np.ndarray] = []   # float64 (H,W) in [0,1]

    with torch.no_grad(), torch.autocast(device_type, dtype=mixed_dtype):
        for m_bin in raw_masks:
            # Tạo ảnh crop với vùng ngoài mask bị làm tối
            # → matting tập trung vào đúng vật thể
            guided = image_np.copy()
            guided[m_bin == 0] = 0          # hoặc dùng blur thay vì black
            alpha_f = matting_model(Image.fromarray(guided))   # float64 (H,W)
            hard = alpha_f > _TH_ALPHA
            m_bin_tensor = torch.from_numpy(m_bin).to(alpha_f.device) > 0
            # Giữ lại chỉ phần alpha nằm trong SAM3 mask
            hard = hard & m_bin_tensor
            hard = hard.bool() 
            alpha_f[~hard] = 0.0
            alpha_np = alpha_f.cpu().to(torch.float64).numpy()
            alpha_np = np.clip(alpha_np, 0.0, 1.0)
            soft_alphas.append(alpha_np)

    # ── 3. Inpaint background cho TỪNG vật thể ─────────────────
    # Luôn bắt đầu từ ảnh GỐC; union với mask của vật thể đè lên
    object_layers: List[ObjectLayer] = []

    with torch.no_grad(), torch.autocast(device_type, enabled=False):
        for idx, (alpha, label) in enumerate(zip(soft_alphas, labels)):
            hard_mask = alpha > _TH_ALPHA      # bool (H,W)
            bbox = _bbox_from_mask(hard_mask.astype(np.uint8) * 255)
            if bbox is None:
                continue

            # 3a. Tạo inpaint mask (target + vùng bị đè)
            inpaint_mask_raw = _build_cumulative_inpaint_mask(raw_masks, idx)
            inpaint_mask_fg  = _build_inpaint_mask(image_np, hard_mask, kernel_size)
            # Merge: union của FG-refine mask và overlap mask
            inpaint_mask = np.maximum(inpaint_mask_raw, inpaint_mask_fg)

            # 3b. Inpaint từ ảnh gốc
            mask_pil = Image.fromarray(inpaint_mask).convert("L")
            bg_pil   = lama(image, mask_pil)
            if bg_pil.size != (orig_w, orig_h):
                bg_pil = bg_pil.resize((orig_w, orig_h), Image.LANCZOS)
            bg_np = np.array(bg_pil.convert("RGB"), dtype=np.uint8)

            # 3c. BG refinement (snap màu)
            bg_np = refine_background(
                bg_np, inpaint_mask.astype(bool),
                n_outer_ratio=_BG_REFINE_N_OUTER_RATIO,
                max_num_colors=_BG_REFINE_NUM_COLORS,
            )

            # 3d. Unblend + per-color alpha refinement
            alpha_refined, fg_rgb = _refine_alpha_with_colors(
                image_np, bg_np, alpha.copy(), hard_mask, kernel_size
            )

            # 3e. Tạo RGBA layer crop theo bbox
            x, y, w, h = bbox
            fg_crop    = fg_rgb[y:y+h, x:x+w]
            alpha_crop = (alpha_refined[y:y+h, x:x+w] * 255).astype(np.uint8)
            rgba_arr   = np.dstack([fg_crop, alpha_crop])
            rgba_img   = Image.fromarray(rgba_arr, mode="RGBA")

            object_layers.append(ObjectLayer(
                keyword=label,
                png_base64=_image_to_base64(rgba_img, "PNG"),
                x=x, y=y, width=w, height=h,
            ))
            logger.info(f"[Layer] '{label}' bbox={bbox}")

    # ── 4. Background cuối: inpaint TẤT CẢ mask cùng lúc ──────
    if raw_masks:
        union_mask = np.zeros_like(raw_masks[0])
        for m in raw_masks:
            union_mask = np.maximum(union_mask, m)
        # Dùng inpaint_mask đã qua FG-refine nếu có
        union_mask = expand_mask(union_mask > 0, kernel_size) # Thêm dòng này
        final_mask_pil = Image.fromarray(union_mask).convert("L")
        final_bg = lama(image, final_mask_pil)
        if final_bg.size != (orig_w, orig_h):
            final_bg = final_bg.resize((orig_w, orig_h), Image.LANCZOS)
        final_bg_np = np.array(final_bg.convert("RGB"), dtype=np.uint8)
        final_bg_np = refine_background(
            final_bg_np, union_mask.astype(bool),
            n_outer_ratio=_BG_REFINE_N_OUTER_RATIO,
            max_num_colors=_BG_REFINE_NUM_COLORS,
        )
        final_bg = Image.fromarray(final_bg_np)
    else:
        final_bg = image

    return ProcessResult(
        background_base64=_image_to_base64(final_bg, "PNG"),
        original_width=orig_w,
        original_height=orig_h,
        layers=object_layers,
    )