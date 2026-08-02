"""External service adapters used by the API boundary."""

from typing import Any

from .keyword_extractor import (
    InvalidKeywordExtraction,
    InvalidSuppliedKeywords,
    KeywordExtractionError,
    KeywordExtractionResult,
    KeywordExtractor,
    KeywordExtractorUnavailable,
    TargetKeywordExtraction,
    normalize_keywords,
)


def __getattr__(name: str) -> Any:
    """Load optional provider SDKs only when their adapter is requested."""
    if name == "ClaudeVisionKeywordExtractor":
        from .claude_vision import ClaudeVisionKeywordExtractor

        return ClaudeVisionKeywordExtractor
    raise AttributeError(name)

__all__ = [
    "ClaudeVisionKeywordExtractor",
    "InvalidKeywordExtraction",
    "InvalidSuppliedKeywords",
    "KeywordExtractionError",
    "KeywordExtractionResult",
    "KeywordExtractor",
    "KeywordExtractorUnavailable",
    "TargetKeywordExtraction",
    "normalize_keywords",
]
