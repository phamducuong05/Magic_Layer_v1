import asyncio
import base64
import io
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from PIL import Image

from backend.services.claude_vision import ClaudeVisionKeywordExtractor
from backend.services.keyword_extractor import (
    InvalidKeywordExtraction,
    KeywordExtractorUnavailable,
)


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.last_request = None

    async def create(self, **kwargs):
        self.last_request = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.messages = FakeMessages(response=response, error=error)


def make_response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
    )


def make_settings(**overrides):
    settings = {
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-5",
        "max_tokens": 256,
        "temperature": 0,
        "timeout_seconds": 30,
        "max_retries": 2,
        "max_image_edge": 32,
        "jpeg_quality": 85,
        "max_keywords": 10,
        "max_keyword_length": 80,
    }
    settings.update(overrides)
    return settings


def test_extract_keywords_sends_image_first_and_parses_structured_output():
    client = FakeClient(
        response=make_response(
            '{"keywords": ["person", "hand", "Person"]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    result = asyncio.run(
        extractor.extract_keywords(Image.new("RGB", (100, 50), "white"))
    )

    assert result == ["person", "hand"]
    request = client.messages.last_request
    assert request["model"] == "claude-sonnet-5"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["required"] == [
        "keywords"
    ]
    content = request["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "text"]
    image_source = content[0]["source"]
    assert image_source["media_type"] == "image/jpeg"

    encoded_image = Image.open(
        io.BytesIO(base64.b64decode(image_source["data"]))
    )
    assert encoded_image.size == (32, 16)


def test_extract_keywords_rejects_refusal_and_truncated_responses():
    for stop_reason in ("refusal", "max_tokens"):
        client = FakeClient(
            response=make_response(
                '{"keywords": ["person"]}',
                stop_reason=stop_reason,
            )
        )
        extractor = ClaudeVisionKeywordExtractor(
            make_settings(), client=client
        )

        with pytest.raises(InvalidKeywordExtraction, match=stop_reason):
            asyncio.run(
                extractor.extract_keywords(Image.new("RGB", (8, 8)))
            )


def test_extract_keywords_rejects_malformed_json():
    client = FakeClient(response=make_response("not-json"))
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    with pytest.raises(InvalidKeywordExtraction, match="valid JSON"):
        asyncio.run(extractor.extract_keywords(Image.new("RGB", (8, 8))))


def test_extract_keywords_maps_transient_anthropic_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(error=anthropic.APITimeoutError(request=request))
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    with pytest.raises(
        KeywordExtractorUnavailable,
        match="temporarily unavailable",
    ):
        asyncio.run(extractor.extract_keywords(Image.new("RGB", (8, 8))))


def test_extract_keywords_maps_authentication_error_to_configuration_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request)
    client = FakeClient(
        error=anthropic.AuthenticationError(
            "invalid API key",
            response=response,
            body=None,
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    with pytest.raises(
        KeywordExtractorUnavailable,
        match="not configured correctly",
    ):
        asyncio.run(extractor.extract_keywords(Image.new("RGB", (8, 8))))


def test_extract_keywords_requires_api_key_when_creating_client(
    monkeypatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    extractor = ClaudeVisionKeywordExtractor(make_settings())

    with pytest.raises(KeywordExtractorUnavailable, match="not configured"):
        asyncio.run(extractor.extract_keywords(Image.new("RGB", (8, 8))))
