"""
main.py — FastAPI application cho AI Magic Canvas.

Endpoints:
  POST /api/process-image  — nhận ảnh + keywords, trả layers + background
  GET  /health             — health check
"""

import io
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import numpy as np

try:
    from .config import config
    from .core.logging import configure_logging, get_logger
    from .image_processor import ProcessResult, process_image
    from .models import model_manager
    from .services import (
        ClaudeVisionKeywordExtractor,
        InvalidKeywordExtraction,
        KeywordExtractor,
        KeywordExtractorUnavailable,
        normalize_keywords,
    )
except ImportError:  # Legacy: run uvicorn from inside backend/.
    from config import config
    from core.logging import configure_logging, get_logger
    from image_processor import ProcessResult, process_image
    from models import model_manager
    from services import (
        ClaudeVisionKeywordExtractor,
        InvalidKeywordExtraction,
        KeywordExtractor,
        KeywordExtractorUnavailable,
        normalize_keywords,
    )

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
_logging_config = config.get_logging_config()
configure_logging(
    level=(
        "DEBUG"
        if _logging_config["workflow_detail"] == "full"
        else _logging_config["level"]
    ),
    third_party_level=_logging_config["third_party_level"],
    force=True,
)
logger = get_logger(__name__)
_keyword_extractor: Optional[KeywordExtractor] = None


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


def get_keyword_extractor() -> KeywordExtractor:
    """Create the configured VLM adapter only when auto extraction is used."""
    global _keyword_extractor
    if _keyword_extractor is not None:
        return _keyword_extractor

    settings = config.get_vlm_config()
    if settings["name"] != "claude":
        raise KeywordExtractorUnavailable(
            f"Unsupported VLM provider: {settings['name']!r}"
        )
    _keyword_extractor = ClaudeVisionKeywordExtractor(settings)
    return _keyword_extractor


async def resolve_keywords(
    image: Image.Image,
    supplied_keywords: Optional[str],
    *,
    extractor: Optional[KeywordExtractor] = None,
) -> List[str]:
    """Resolve a transitional manual override or invoke the configured VLM."""
    settings = config.get_vlm_config()
    max_keywords = int(settings.get("max_keywords", 10))
    max_length = int(settings.get("max_keyword_length", 80))

    if supplied_keywords is not None:
        return normalize_keywords(
            supplied_keywords.split(","),
            max_keywords=max_keywords,
            max_length=max_length,
        )

    active_extractor = extractor or get_keyword_extractor()
    extracted = await active_extractor.extract_keywords(image)
    return normalize_keywords(
        extracted,
        max_keywords=max_keywords,
        max_length=max_length,
    )


@app.post("/api/process-image", response_model=ProcessResponse)
async def api_process_image(
    file: UploadFile = File(..., description="Ảnh cần xử lý (JPEG/PNG)"),
    keywords: Optional[str] = Form(
        None,
        description=(
            "Manual override tùy chọn; nếu bỏ trống, VLM sẽ tự sinh "
            "keyword cho SAM3"
        ),
    ),
):
    """
    Pipeline chính:
    1. Đọc ảnh upload
    2. Dùng manual keywords hoặc gọi VLM để tự sinh keywords
    3. Chạy SAM3 → lấy masks + tạo RGBA layers
    4. Merge masks → LaMa inpaint → ảnh nền sạch
    5. Trả về JSON với background + danh sách layers
    """
    # ── Validate file ──
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Định dạng file không hỗ trợ: {file.content_type}. "
                "Chỉ chấp nhận JPEG/PNG/WEBP."
            ),
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

    # ── Resolve keywords ──
    try:
        kw_list = await resolve_keywords(image, keywords)
    except KeywordExtractorUnavailable:
        logger.exception("VLM keyword extractor unavailable")
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ phân tích ảnh tạm thời không khả dụng.",
        )
    except InvalidKeywordExtraction:
        if keywords is not None:
            raise HTTPException(
                status_code=400,
                detail="Danh sách từ khóa thủ công không hợp lệ.",
            )
        logger.exception("VLM returned unusable keyword output")
        raise HTTPException(
            status_code=502,
            detail="Dịch vụ phân tích ảnh không trả về từ khóa hợp lệ.",
        )

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
