import asyncio
import importlib
import io
import sys
import types
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from backend.services.keyword_extractor import (
    InvalidKeywordExtraction,
    KeywordExtractorUnavailable,
)


class FakeExtractor:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    async def extract_keywords(self, image):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def load_main(monkeypatch):
    fake_image_processor = types.ModuleType("backend.image_processor")
    fake_image_processor.ProcessResult = object
    fake_image_processor.process_image = lambda image, keywords: SimpleNamespace(
        background_base64="background",
        original_width=image.width,
        original_height=image.height,
        layers=[],
    )
    fake_models = types.ModuleType("backend.models")
    fake_models.model_manager = SimpleNamespace(
        warmup_first_stage=lambda: None
    )
    monkeypatch.setitem(
        sys.modules, "backend.image_processor", fake_image_processor
    )
    monkeypatch.setitem(sys.modules, "backend.models", fake_models)
    sys.modules.pop("backend.main", None)
    return importlib.import_module("backend.main")


def image_upload():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "white").save(buffer, format="PNG")
    return ("image.png", buffer.getvalue(), "image/png")


def test_resolve_keywords_uses_manual_override_without_calling_vlm(
    monkeypatch,
):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(result=["ignored"])

    result = asyncio.run(
        main.resolve_keywords(
            Image.new("RGB", (8, 6)),
            " Person, dog,person ",
            extractor=extractor,
        )
    )

    assert result == ["Person", "dog"]
    assert extractor.calls == 0


def test_resolve_keywords_uses_vlm_when_manual_override_is_omitted(
    monkeypatch,
):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(result=["person", "hand"])

    result = asyncio.run(
        main.resolve_keywords(
            Image.new("RGB", (8, 6)),
            None,
            extractor=extractor,
        )
    )

    assert result == ["person", "hand"]
    assert extractor.calls == 1


def test_process_image_returns_503_when_vlm_is_unavailable(monkeypatch):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        error=KeywordExtractorUnavailable("upstream unavailable")
    )
    monkeypatch.setattr(main, "get_keyword_extractor", lambda: extractor)

    response = TestClient(main.app).post(
        "/api/process-image",
        files={"file": image_upload()},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Dịch vụ phân tích ảnh tạm thời không khả dụng."
    }


def test_process_image_returns_502_for_unusable_vlm_output(monkeypatch):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        error=InvalidKeywordExtraction("bad structured output")
    )
    monkeypatch.setattr(main, "get_keyword_extractor", lambda: extractor)

    response = TestClient(main.app).post(
        "/api/process-image",
        files={"file": image_upload()},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Dịch vụ phân tích ảnh không trả về từ khóa hợp lệ."
    }
