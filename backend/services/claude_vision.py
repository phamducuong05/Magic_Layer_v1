"""Anthropic Claude adapter for foreground SAM3 keyword extraction."""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

import anthropic
from PIL import Image

from ..core.logging import get_logger
from .keyword_extractor import (
    InvalidKeywordExtraction,
    KeywordExtractorUnavailable,
    normalize_keywords,
)

logger = get_logger(__name__)

KEYWORD_PROMPT = """Analyze this image to generate English text prompts for an object
segmentation model.

Include:
1. Visually important foreground objects that occupy a meaningful area or
   play a major role in the scene.
2. Every visible object that significantly occludes an important foreground
   object, even when the occluder itself is small.

Exclude:
1. Background scenery and distant objects.
2. Tiny incidental objects, decorations, textures, shadows, reflections,
   printed images, and object parts that should remain attached to their
   parent object.
3. Objects that are uncertain or not visibly distinguishable.

Use short, concrete English nouns or noun phrases that work as SAM3 text
prompts. Return one keyword per semantic object class. Prefer recall for
significant occluders, but do not list irrelevant background objects."""

KEYWORD_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["keywords"],
    "additionalProperties": False,
}


class ClaudeVisionKeywordExtractor:
    """Extract SAM3 keyword prompts from one image using Claude Vision."""

    def __init__(self, settings: dict[str, Any], client: Any = None):
        self._settings = dict(settings)
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key_env = str(
            self._settings.get("api_key_env", "ANTHROPIC_API_KEY")
        )
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise KeywordExtractorUnavailable(
                "The VLM keyword extractor is not configured."
            )

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=float(self._settings.get("timeout_seconds", 30)),
            max_retries=int(self._settings.get("max_retries", 2)),
        )
        return self._client

    def _encode_image(self, image: Image.Image) -> str:
        prepared = image.convert("RGB").copy()
        max_edge = int(self._settings.get("max_image_edge", 1568))
        prepared.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        prepared.save(
            buffer,
            format="JPEG",
            quality=int(self._settings.get("jpeg_quality", 85)),
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    async def extract_keywords(self, image: Image.Image) -> list[str]:
        client = self._get_client()
        image_data = self._encode_image(image)

        try:
            response = await client.messages.create(
                model=str(self._settings["model"]),
                max_tokens=int(self._settings.get("max_tokens", 256)),
                temperature=float(self._settings.get("temperature", 0)),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": KEYWORD_PROMPT},
                        ],
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": KEYWORD_OUTPUT_SCHEMA,
                    }
                },
            )
        except anthropic.AuthenticationError as exc:
            logger.error("Claude authentication failed")
            raise KeywordExtractorUnavailable(
                "The VLM keyword extractor is not configured correctly."
            ) from exc
        except (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
        ) as exc:
            logger.warning(
                "Claude keyword extraction is unavailable: %s",
                type(exc).__name__,
            )
            raise KeywordExtractorUnavailable(
                "The VLM keyword extractor is temporarily unavailable."
            ) from exc
        except anthropic.APIError as exc:
            logger.warning("Claude keyword extraction failed: %s", type(exc).__name__)
            raise KeywordExtractorUnavailable(
                "The VLM keyword extractor is temporarily unavailable."
            ) from exc

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason in {"refusal", "max_tokens"}:
            raise InvalidKeywordExtraction(
                f"Claude returned an unusable response: {stop_reason}."
            )

        text_blocks = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        if not text_blocks:
            raise InvalidKeywordExtraction(
                "Claude returned no structured keyword output."
            )

        try:
            payload = json.loads("".join(text_blocks))
        except json.JSONDecodeError as exc:
            raise InvalidKeywordExtraction(
                "Claude did not return valid JSON keyword output."
            ) from exc

        if not isinstance(payload, dict) or not isinstance(
            payload.get("keywords"), list
        ):
            raise InvalidKeywordExtraction(
                "Claude returned an invalid keyword structure."
            )

        return normalize_keywords(
            payload["keywords"],
            max_keywords=int(self._settings.get("max_keywords", 10)),
            max_length=int(
                self._settings.get("max_keyword_length", 80)
            ),
        )
