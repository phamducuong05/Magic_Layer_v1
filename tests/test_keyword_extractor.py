import pytest

from backend.config import ConfigManager
from backend.services.keyword_extractor import (
    InvalidKeywordExtraction,
    normalize_keywords,
)


def test_normalize_keywords_strips_and_deduplicates_case_insensitively():
    result = normalize_keywords(
        [" Person ", "person", "", "DOG", " dog "],
        max_keywords=10,
        max_length=80,
    )

    assert result == ["Person", "DOG"]


def test_normalize_keywords_rejects_empty_output():
    with pytest.raises(InvalidKeywordExtraction, match="no usable keywords"):
        normalize_keywords(["", "  "], max_keywords=10, max_length=80)


def test_normalize_keywords_rejects_too_many_values():
    with pytest.raises(InvalidKeywordExtraction, match="at most 2"):
        normalize_keywords(
            ["person", "dog", "chair"],
            max_keywords=2,
            max_length=80,
        )


def test_normalize_keywords_rejects_an_overlong_value():
    with pytest.raises(InvalidKeywordExtraction, match="maximum length"):
        normalize_keywords(
            ["wooden chair"],
            max_keywords=10,
            max_length=5,
        )


def test_get_vlm_config_merges_shared_and_provider_settings():
    manager = object.__new__(ConfigManager)
    manager.config_dict = {
        "vlm": {
            "active": "claude",
            "max_keywords": 10,
            "max_occluders": 10,
            "max_keyword_length": 80,
            "claude": {
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-sonnet-5",
            },
        }
    }

    assert manager.get_vlm_config() == {
        "name": "claude",
        "max_keywords": 10,
        "max_occluders": 10,
        "max_keyword_length": 80,
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-5",
    }


def test_get_vlm_config_rejects_a_non_mapping_provider():
    manager = object.__new__(ConfigManager)
    manager.config_dict = {
        "vlm": {
            "active": "claude",
            "claude": "invalid",
        }
    }

    with pytest.raises(ValueError, match="provider 'claude' must be a mapping"):
        manager.get_vlm_config()
