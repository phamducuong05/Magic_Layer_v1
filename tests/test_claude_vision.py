import asyncio
import base64
import io
import json
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
            '{"visible_person_count": 1, '
            '"keywords": ["person", "chair", "Person"]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    result = asyncio.run(
        extractor.extract_keywords(Image.new("RGB", (100, 50), "white"))
    )

    assert result.keywords == ["person", "chair"]
    assert result.occluders == []
    request = client.messages.last_request
    assert request["model"] == "claude-sonnet-5"
    assert "temperature" not in request
    assert request["output_config"]["format"]["type"] == "json_schema"
    schema = request["output_config"]["format"]["schema"]
    assert schema["required"] == ["visible_person_count", "keywords"]
    assert schema["properties"]["keywords"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert "large, visually important foreground objects" in request["system"]
    content = request["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "text"]
    image_source = content[0]["source"]
    assert image_source["media_type"] == "image/jpeg"

    encoded_image = Image.open(
        io.BytesIO(base64.b64decode(image_source["data"]))
    )
    assert encoded_image.size == (32, 16)


def test_automatic_prompt_requires_every_visible_occluder():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 0, "keywords": ["car"]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    asyncio.run(extractor.extract_keywords(Image.new("RGB", (8, 8))))

    prompt = " ".join(client.messages.last_request["system"].split())
    assert "visibly occludes any part" in prompt
    assert "significantly occludes" not in prompt


def test_extract_keywords_uses_indexed_occluder_mode_for_each_target():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 1, "target_results": ['
            '{"input_index": 0, "refined_keyword": "car", '
            '"occluders": ["person", "tree"]}]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    result = asyncio.run(
        extractor.extract_keywords(
            Image.new("RGB", (100, 50), "white"),
            target_keywords=["red four-door passenger automobile"],
        )
    )

    assert result.keywords == ["car"]
    assert result.occluders == ["person", "tree"]
    assert result.target_results[0].input_index == 0
    assert result.target_results[0].source_keyword == (
        "red four-door passenger automobile"
    )
    request = client.messages.last_request
    schema = request["output_config"]["format"]["schema"]
    assert schema["required"] == [
        "visible_person_count",
        "target_results",
    ]
    item_schema = schema["properties"]["target_results"]["items"]
    assert item_schema["required"] == [
        "input_index",
        "refined_keyword",
        "occluders",
    ]
    assert (
        "exhaustively return independent objects that visibly"
        in request["system"]
    )
    target_data = request["messages"][0]["content"][1]["text"]
    assert '"red four-door passenger automobile"' in target_data


def test_extract_keywords_allows_no_occluders_for_manual_targets():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 0, "target_results": ['
            '{"input_index": 0, "refined_keyword": "car", '
            '"occluders": []}]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    result = asyncio.run(
        extractor.extract_keywords(
            Image.new("RGB", (8, 8)),
            target_keywords=["automobile"],
        )
    )

    assert result.keywords == ["automobile"]
    assert result.occluders == []


def test_both_modes_send_the_same_common_strict_rules():
    cases = [
        (
            None,
            '{"visible_person_count": 1, "keywords": ["person"]}',
        ),
        (
            ["human individual"],
            '{"visible_person_count": 1, "target_results": ['
            '{"input_index": 0, "refined_keyword": "person", '
            '"occluders": []}]}',
        ),
    ]

    for target_keywords, response_text in cases:
        client = FakeClient(response=make_response(response_text))
        extractor = ClaudeVisionKeywordExtractor(
            make_settings(),
            client=client,
        )

        asyncio.run(
            extractor.extract_keywords(
                Image.new("RGB", (8, 8)),
                target_keywords=target_keywords,
            )
        )

        system_prompt = client.messages.last_request["system"]
        normalized_prompt = " ".join(system_prompt.split())
        assert "Common strict rules for every returned keyword:" in system_prompt
        assert "Whole objects only." in system_prompt
        assert "extremely simple, common English nouns" in system_prompt
        assert "Foreground only." in system_prompt
        assert "Exclude clothing, footwear" in system_prompt
        assert "regardless of how tiny" in normalized_prompt
        assert "one to three visible people" in normalized_prompt
        assert (
            "Never return buildings, landmarks, venues, or places"
            in normalized_prompt
        )
        assert (
            "temples, pagodas, churches, monuments, towers, and houses"
            in normalized_prompt
        )


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


def test_extract_keywords_keeps_only_the_first_configured_keywords():
    keywords = [f"object-{index}" for index in range(12)]
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 0, "keywords": ['
            + ", ".join(f'"{keyword}"' for keyword in keywords)
            + "]}"
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    result = asyncio.run(
        extractor.extract_keywords(Image.new("RGB", (8, 8)))
    )

    assert result.keywords == keywords[:10]
    assert result.occluders == []


def test_manual_single_token_target_rejects_model_generalization():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 2, "target_results": ['
            '{"input_index": 0, "refined_keyword": "people", '
            '"occluders": ["chair"]}]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    result = asyncio.run(
        extractor.extract_keywords(
            Image.new("RGB", (8, 8)),
            target_keywords=["man"],
        )
    )

    assert result.keywords == ["man"]
    assert result.occluders == ["chair"]


def test_manual_people_target_is_not_rewritten_by_visible_count():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 2, "target_results": ['
            '{"input_index": 0, "refined_keyword": "person", '
            '"occluders": ["chair"]}]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    result = asyncio.run(
        extractor.extract_keywords(
            Image.new("RGB", (8, 8)),
            target_keywords=["people"],
        )
    )

    assert result.keywords == ["people"]


@pytest.mark.parametrize(
    "target_results",
    [
        [],
        [
            {
                "input_index": 0,
                "refined_keyword": "car",
                "occluders": [],
            },
            {
                "input_index": 0,
                "refined_keyword": "table",
                "occluders": [],
            },
        ],
        [
            {
                "input_index": 2,
                "refined_keyword": "car",
                "occluders": [],
            },
            {
                "input_index": 1,
                "refined_keyword": "table",
                "occluders": [],
            },
        ],
    ],
)
def test_manual_mode_rejects_incomplete_or_invalid_target_indexes(
    target_results,
):
    client = FakeClient(
        response=make_response(
            json.dumps(
                {
                    "visible_person_count": 0,
                    "target_results": target_results,
                }
            )
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    with pytest.raises(InvalidKeywordExtraction, match="target indexes"):
        asyncio.run(
            extractor.extract_keywords(
                Image.new("RGB", (8, 8)),
                target_keywords=["car", "table"],
            )
        )


def test_automatic_mode_rejects_people_for_three_visible_people():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 3, "keywords": ["people"]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    with pytest.raises(InvalidKeywordExtraction, match="people"):
        asyncio.run(extractor.extract_keywords(Image.new("RGB", (8, 8))))


def test_automatic_mode_validates_people_beyond_the_output_quota():
    raw_keywords = [f"object-{index}" for index in range(10)]
    raw_keywords.append("people")
    client = FakeClient(
        response=make_response(
            json.dumps(
                {
                    "visible_person_count": 2,
                    "keywords": raw_keywords,
                }
            )
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    with pytest.raises(InvalidKeywordExtraction, match="people"):
        asyncio.run(extractor.extract_keywords(Image.new("RGB", (8, 8))))


def test_automatic_mode_allows_people_above_three_visible_people():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 4, "keywords": ["people"]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    result = asyncio.run(
        extractor.extract_keywords(Image.new("RGB", (8, 8)))
    )

    assert result.keywords == ["people"]


@pytest.mark.parametrize("child_label", ["hand", "glasses", "hat", "shirt"])
def test_automatic_mode_rejects_non_root_person_labels(child_label):
    client = FakeClient(
        response=make_response(
            json.dumps(
                {
                    "visible_person_count": 1,
                    "keywords": [child_label],
                }
            )
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    with pytest.raises(InvalidKeywordExtraction, match="root person"):
        asyncio.run(extractor.extract_keywords(Image.new("RGB", (8, 8))))


def test_manual_mode_rejects_non_root_person_occluder():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 1, "target_results": ['
            '{"input_index": 0, "refined_keyword": "product", '
            '"occluders": ["hand"]}]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    with pytest.raises(InvalidKeywordExtraction, match="root person"):
        asyncio.run(
            extractor.extract_keywords(
                Image.new("RGB", (8, 8)),
                target_keywords=["product"],
            )
        )


def test_manual_prompt_requires_exhaustive_parent_level_occluders():
    client = FakeClient(
        response=make_response(
            '{"visible_person_count": 1, "target_results": ['
            '{"input_index": 0, "refined_keyword": "product", '
            '"occluders": ["person"]}]}'
        )
    )
    extractor = ClaudeVisionKeywordExtractor(make_settings(), client=client)

    asyncio.run(
        extractor.extract_keywords(
            Image.new("RGB", (8, 8)),
            target_keywords=["product"],
        )
    )

    prompt = " ".join(client.messages.last_request["system"].split())
    assert "Inspect every user target independently" in prompt
    assert "hand, glasses, hat, or clothing" in prompt
    assert "return the person parent" in prompt


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
