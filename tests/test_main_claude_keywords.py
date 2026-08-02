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
        self.target_keywords = None

    async def extract_keywords(self, image, target_keywords=None):
        self.calls += 1
        self.target_keywords = target_keywords
        if self.error is not None:
            raise self.error
        return self.result


def load_main(monkeypatch):
    fake_image_processor = types.ModuleType("backend.image_processor")
    fake_image_processor.ProcessResult = object
    fake_image_processor.process_image = (
        lambda image, keywords, progress_callback=None: SimpleNamespace(
            background_base64="background",
            original_width=image.width,
            original_height=image.height,
            layers=[],
        )
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


def test_resolve_keywords_simplifies_manual_targets_and_adds_occluders(
    monkeypatch,
):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        result=SimpleNamespace(
            keywords=["car"],
            occluders=["person", "Car"],
        )
    )

    result = asyncio.run(
        main.resolve_keywords(
            Image.new("RGB", (8, 6)),
            " red four-door passenger automobile ",
            extractor=extractor,
        )
    )

    assert result == ["car", "person"]
    assert extractor.calls == 1
    assert extractor.target_keywords == ["red four-door passenger automobile"]


def test_resolve_keywords_uses_vlm_when_manual_override_is_omitted(
    monkeypatch,
):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        result=SimpleNamespace(
            keywords=["person", "chair"],
            occluders=[],
        )
    )

    result = asyncio.run(
        main.resolve_keywords(
            Image.new("RGB", (8, 6)),
            None,
            extractor=extractor,
        )
    )

    assert result == ["person", "chair"]
    assert extractor.calls == 1
    assert extractor.target_keywords is None


def test_resolve_keywords_keeps_independent_target_and_occluder_quotas(
    monkeypatch,
):
    main = load_main(monkeypatch)
    targets = [f"target-{index}" for index in range(10)]
    occluders = [f"occluder-{index}" for index in range(10)]
    extractor = FakeExtractor(
        result=SimpleNamespace(
            keywords=targets,
            occluders=occluders,
        )
    )

    result = asyncio.run(
        main.resolve_keywords(
            Image.new("RGB", (8, 6)),
            ",".join(targets),
            extractor=extractor,
        )
    )

    assert result == [*targets, *occluders]


def test_target_duplicate_does_not_consume_an_occluder_slot(monkeypatch):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        result=SimpleNamespace(
            keywords=["car"],
            occluders=[
                "CAR",
                *[f"occluder-{index}" for index in range(10)],
            ],
        )
    )

    result = asyncio.run(
        main.resolve_keywords(
            Image.new("RGB", (8, 6)),
            "car",
            extractor=extractor,
        )
    )

    assert result == [
        "car",
        *[f"occluder-{index}" for index in range(10)],
    ]


def test_resolve_keywords_treats_blank_manual_prompt_as_auto_mode(
    monkeypatch,
):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        result=SimpleNamespace(keywords=["person"], occluders=[])
    )

    result = asyncio.run(
        main.resolve_keywords(
            Image.new("RGB", (8, 6)),
            "   ",
            extractor=extractor,
        )
    )

    assert result == ["person"]
    assert extractor.target_keywords is None


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


def test_process_image_returns_502_for_unusable_vlm_output_in_manual_mode(
    monkeypatch,
):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        error=InvalidKeywordExtraction("bad structured output")
    )
    monkeypatch.setattr(main, "get_keyword_extractor", lambda: extractor)

    response = TestClient(main.app).post(
        "/api/process-image",
        files={"file": image_upload()},
        data={"keywords": "red four-door passenger automobile"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Dịch vụ phân tích ảnh không trả về từ khóa hợp lệ."
    }


def test_process_image_returns_400_for_too_many_user_keywords(monkeypatch):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        result=SimpleNamespace(keywords=["car"], occluders=[])
    )
    monkeypatch.setattr(main, "get_keyword_extractor", lambda: extractor)
    supplied = ",".join(f"target-{index}" for index in range(11))

    response = TestClient(main.app).post(
        "/api/process-image",
        files={"file": image_upload()},
        data={"keywords": supplied},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Danh sách từ khóa người dùng không hợp lệ."
    }
    assert extractor.calls == 0
