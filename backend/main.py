"""
main.py — FastAPI application cho AI Magic Canvas.

Endpoints:
  POST /api/process-image  — nhận ảnh + keywords, trả layers + background
  POST /api/process-image/jobs — tạo background processing job
  GET  /api/process-image/jobs/{job_id} — đọc tiến độ hoặc kết quả job
  GET  /health             — health check
"""

import asyncio
from dataclasses import asdict
import io
from typing import List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

import numpy as np

try:
    from .config import config
    from .core.logging import (
        configure_logging,
        get_logger,
        log_event,
        workflow_event,
    )
    from .image_processor import ProcessResult, process_image
    from .models import model_manager
    from .services import (
        InvalidKeywordExtraction,
        InvalidSuppliedKeywords,
        KeywordExtractor,
        KeywordExtractorUnavailable,
        normalize_keywords,
    )
    from .services.process_jobs import ProcessJobSnapshot, ProcessJobStore
except ImportError:  # Legacy: run uvicorn from inside backend/.
    from config import config
    from core.logging import (
        configure_logging,
        get_logger,
        log_event,
        workflow_event,
    )
    from image_processor import ProcessResult, process_image
    from models import model_manager
    from services import (
        InvalidKeywordExtraction,
        InvalidSuppliedKeywords,
        KeywordExtractor,
        KeywordExtractorUnavailable,
        normalize_keywords,
    )
    from services.process_jobs import ProcessJobSnapshot, ProcessJobStore

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
process_job_store = ProcessJobStore()


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
    workflow_event(logger, "server", "Loading AI models")
    model_manager.warmup_first_stage()
    workflow_event(logger, "server", "Ready")


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


class ProcessJobResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: int
    message: str
    result: Optional[ProcessResponse] = None
    error: Optional[str] = None


def _build_process_response(result: ProcessResult) -> ProcessResponse:
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


def _build_job_response(snapshot: ProcessJobSnapshot) -> ProcessJobResponse:
    return ProcessJobResponse(**asdict(snapshot))


async def _read_uploaded_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Định dạng file không hỗ trợ: {file.content_type}. "
                "Chỉ chấp nhận JPEG/PNG/WEBP."
            ),
        )
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image = Image.fromarray(np.array(image, dtype=np.uint8))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Không đọc được ảnh: {str(exc)}",
        ) from exc

    max_dimension = 2048
    width, height = image.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.LANCZOS,
        )
        log_event(
            logger,
            "upload",
            "resized",
            original_size=f"{width}x{height}",
            resized_size=f"{image.width}x{image.height}",
        )
    return image


def _job_error_message(exc: Exception) -> str:
    if isinstance(exc, KeywordExtractorUnavailable):
        return "The image analysis service is temporarily unavailable."
    if isinstance(exc, InvalidSuppliedKeywords):
        return "The supplied keywords are invalid."
    if isinstance(exc, InvalidKeywordExtraction):
        return "The image analysis service returned invalid keywords."
    return "Layer extraction failed. Please try again."


async def _run_process_job(
    job_id: str,
    image: Image.Image,
    supplied_keywords: Optional[str],
) -> None:
    try:
        process_job_store.update(
            job_id,
            "keywords",
            5,
            "Analyzing image keywords",
        )
        keywords = await resolve_keywords(image, supplied_keywords)
        workflow_event(
            logger,
            "keywords",
            "Extracted keywords",
            keywords=keywords,
        )
        process_job_store.update(job_id, "keywords", 10, "Keywords ready")

        def on_progress(stage: str, progress: int, message: str) -> None:
            process_job_store.update(
                job_id,
                stage,
                progress,
                message,
            )

        result = await asyncio.to_thread(
            process_image,
            image,
            keywords,
            progress_callback=on_progress,
        )
        response = _build_process_response(result)
        process_job_store.complete(job_id, response.model_dump())
    except Exception as exc:
        logger.error(
            "[FAILED] Processing job: %s",
            str(exc) or type(exc).__name__,
        )
        logger.debug("Processing job traceback", exc_info=True)
        process_job_store.fail(job_id, _job_error_message(exc))


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/api/process-image/jobs",
    response_model=ProcessJobResponse,
    status_code=202,
)
async def create_process_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    keywords: Optional[str] = Form(None),
):
    image = await _read_uploaded_image(file)
    snapshot = process_job_store.create()
    background_tasks.add_task(
        _run_process_job,
        snapshot.job_id,
        image,
        keywords,
    )
    return _build_job_response(snapshot)


@app.get(
    "/api/process-image/jobs/{job_id}",
    response_model=ProcessJobResponse,
)
async def get_process_job(job_id: str):
    try:
        snapshot = process_job_store.get(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Processing job not found.",
        ) from None
    return _build_job_response(snapshot)


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
    try:
        from .services.claude_vision import ClaudeVisionKeywordExtractor
    except ImportError:  # Legacy: run uvicorn from inside backend/.
        from services.claude_vision import ClaudeVisionKeywordExtractor

    _keyword_extractor = ClaudeVisionKeywordExtractor(settings)
    return _keyword_extractor


async def resolve_keywords(
    image: Image.Image,
    supplied_keywords: Optional[str],
    *,
    extractor: Optional[KeywordExtractor] = None,
) -> List[str]:
    """Generate simple SAM3 targets and their foreground occluders."""
    settings = config.get_vlm_config()
    max_keywords = int(settings.get("max_keywords", 10))
    max_occluders = int(settings.get("max_occluders", 10))
    max_length = int(settings.get("max_keyword_length", 80))

    target_keywords = None
    if supplied_keywords is not None and supplied_keywords.strip():
        try:
            target_keywords = normalize_keywords(
                supplied_keywords.split(","),
                max_keywords=max_keywords,
                max_length=max_length,
            )
        except InvalidKeywordExtraction as exc:
            raise InvalidSuppliedKeywords(str(exc)) from exc

    active_extractor = extractor or get_keyword_extractor()
    extracted = await active_extractor.extract_keywords(
        image,
        target_keywords=target_keywords,
    )
    resolved_targets = normalize_keywords(
        extracted.keywords,
        max_keywords=max(max_keywords, len(extracted.keywords)),
        max_length=max_length,
    )[:max_keywords]
    target_keys = {
        keyword.casefold() for keyword in resolved_targets
    }
    normalized_occluders = (
        normalize_keywords(
            extracted.occluders,
            max_keywords=max(
                max_occluders,
                len(extracted.occluders),
            ),
            max_length=max_length,
        )
        if extracted.occluders
        else []
    )
    resolved_occluders = [
        keyword
        for keyword in normalized_occluders
        if keyword.casefold() not in target_keys
    ][:max_occluders]
    log_event(
        logger,
        "keywords",
        "resolved",
        target_count=len(resolved_targets),
        occluder_count=len(resolved_occluders),
    )
    return [*resolved_targets, *resolved_occluders]


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
    image = await _read_uploaded_image(file)

    # ── Resolve keywords ──
    try:
        kw_list = await resolve_keywords(image, keywords)
    except KeywordExtractorUnavailable:
        logger.exception("VLM keyword extractor unavailable")
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ phân tích ảnh tạm thời không khả dụng.",
        )
    except InvalidSuppliedKeywords:
        raise HTTPException(
            status_code=400,
            detail="Danh sách từ khóa người dùng không hợp lệ.",
        )
    except InvalidKeywordExtraction:
        logger.exception("VLM returned unusable keyword output")
        raise HTTPException(
            status_code=502,
            detail="Dịch vụ phân tích ảnh không trả về từ khóa hợp lệ.",
        )

    workflow_event(
        logger,
        "keywords",
        "Extracted keywords",
        keywords=kw_list,
    )

    # ── Chạy pipeline ──
    try:
        result: ProcessResult = process_image(image, kw_list)
    except Exception as e:
        logger.error("[FAILED] Pipeline: %s", str(e) or type(e).__name__)
        logger.debug("Pipeline traceback", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    # ── Build response ──
    return _build_process_response(result)
