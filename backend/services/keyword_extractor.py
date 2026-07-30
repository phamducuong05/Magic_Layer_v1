"""Provider-neutral contract for image-to-keyword extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

from PIL import Image


class KeywordExtractionError(RuntimeError):
    """Base error safe for mapping at the HTTP boundary."""


class KeywordExtractorUnavailable(KeywordExtractionError):
    """The configured extractor cannot currently serve requests."""


class InvalidKeywordExtraction(KeywordExtractionError):
    """The extractor returned no usable SAM3 keywords."""


class InvalidSuppliedKeywords(KeywordExtractionError):
    """The user supplied an invalid keyword list."""


@dataclass(frozen=True)
class TargetKeywordExtraction:
    """Validated target refinement and its ranked raw occluder keywords."""

    input_index: int
    source_keyword: str
    keyword: str
    occluders: tuple[str, ...] = ()


@dataclass(frozen=True)
class KeywordExtractionResult:
    """Structured SAM3 targets and the objects that occlude them."""

    keywords: list[str]
    occluders: list[str] = field(default_factory=list)
    target_results: tuple[TargetKeywordExtraction, ...] = ()
    visible_person_count: int | None = None


class KeywordExtractor(Protocol):
    async def extract_keywords(
        self,
        image: Image.Image,
        target_keywords: Sequence[str] | None = None,
    ) -> KeywordExtractionResult:
        """Return normalized English SAM3 prompts and related occluders."""


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
