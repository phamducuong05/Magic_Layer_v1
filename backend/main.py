"""
main.py — FastAPI application cho AI Magic Canvas.

Endpoints:
  POST /api/process-image  — nhận ảnh + keywords, trả layers + background
  GET  /health             — health check
"""

import io
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import numpy as np

try:
    from .core.logging import configure_logging, get_logger
    from .image_processor import ProcessResult, process_image
    from .models import model_manager
except ImportError:  # Legacy: run uvicorn from inside backend/.
    from core.logging import configure_logging, get_logger
    from image_processor import ProcessResult, process_image
    from models import model_manager

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
configure_logging(level="INFO", force=True)
logger = get_logger(__name__)


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="AI Magic Canvas API",
    description="SAM3 segmentation + LaMa inpainting backend",
    version="1.0.0",
)

# CORS — cho phép frontend dev server (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8009", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Startup: load models
# ──────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Server starting — loading AI models...")
    model_manager.warmup_first_stage()
    logger.info("Server ready.")


# ──────────────────────────────────────────────
# Response schema (Pydantic)
# ──────────────────────────────────────────────
class LayerResponse(BaseModel):
    keyword: str
    png_base64: str
    x: int
    y: int
    width: int
    height: int


class ProcessResponse(BaseModel):
    background_base64: str
    original_width: int
    original_height: int
    layers: List[LayerResponse]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/process-image", response_model=ProcessResponse)
async def api_process_image(
    file: UploadFile = File(..., description="Ảnh cần xử lý (JPEG/PNG)"),
    keywords: str = Form(..., description="Danh sách từ khóa cách nhau bởi dấu phẩy, vd: 'dog,cat'"),
):
    """
    Pipeline chính:
    1. Đọc ảnh upload
    2. Parse keywords (tách bởi dấu phẩy)
    3. Chạy SAM3 → lấy masks + tạo RGBA layers
    4. Merge masks → LaMa inpaint → ảnh nền sạch
    5. Trả về JSON với background + danh sách layers
    """
    # ── Validate file ──
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không hỗ trợ: {file.content_type}. Chỉ chấp nhận JPEG/PNG/WEBP.",
        )

    # ── Đọc ảnh ──
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image = Image.fromarray(np.array(image, dtype=np.uint8))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {str(e)}")

    # Giới hạn kích thước để tránh OOM
    MAX_DIM = 2048
    w, h = image.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        logger.info(f"Image resized from {w}x{h} to {image.size}")

    # ── Parse keywords ──
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    if not kw_list:
        raise HTTPException(status_code=400, detail="Cần ít nhất một từ khóa.")
    if len(kw_list) > 10:
        raise HTTPException(status_code=400, detail="Tối đa 10 từ khóa mỗi lần.")

    logger.info(f"Processing image {image.size} with keywords: {kw_list}")

    # ── Chạy pipeline ──
    try:
        result: ProcessResult = process_image(image, kw_list)
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    # ── Build response ──
    return ProcessResponse(
        background_base64=result.background_base64,
        original_width=result.original_width,
        original_height=result.original_height,
        layers=[
            LayerResponse(
                keyword=layer.keyword,
                png_base64=layer.png_base64,
                x=layer.x,
                y=layer.y,
                width=layer.width,
                height=layer.height,
            )
            for layer in result.layers
        ],
    )
