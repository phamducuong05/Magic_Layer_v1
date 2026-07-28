"""Provider-neutral contract for image-to-keyword extraction."""

from __future__ import annotations

from typing import Iterable, Protocol

from PIL import Image


class KeywordExtractionError(RuntimeError):
    """Base error safe for mapping at the HTTP boundary."""


class KeywordExtractorUnavailable(KeywordExtractionError):
    """The configured extractor cannot currently serve requests."""


class InvalidKeywordExtraction(KeywordExtractionError):
    """The extractor returned no usable SAM3 keywords."""


class KeywordExtractor(Protocol):
    async def extract_keywords(self, image: Image.Image) -> list[str]:
        """Return normalized English text prompts for SAM3."""


def normalize_keywords(
    values: Iterable[str],
    *,
    max_keywords: int,
    max_length: int,
) -> list[str]:
    """Normalize keyword values while preserving their original order."""
    normalized: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            raise InvalidKeywordExtraction("Keyword values must be strings.")
        keyword = value.strip()
        if not keyword:
            continue
        if len(keyword) > max_length:
            raise InvalidKeywordExtraction(
                f"Each keyword has a maximum length of {max_length} characters."
            )
        comparison_key = keyword.casefold()
        if comparison_key in seen:
            continue
        seen.add(comparison_key)
        normalized.append(keyword)

    if not normalized:
        raise InvalidKeywordExtraction("The extractor returned no usable keywords.")
    if len(normalized) > max_keywords:
        raise InvalidKeywordExtraction(
            f"The extractor may return at most {max_keywords} keywords."
        )
    return normalized
