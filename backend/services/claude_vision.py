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
3. Clothing, apparel, wearable items, and accessories belonging or attached
   to a person, animal, or object. This includes shirts, pants, dresses,
   shoes, hats, glasses, jewelry, watches, belts, bags, backpacks, collars,
   leashes, straps, handles, and similar attachments. Treat them as part of
   their parent object instead of separate segmentation keywords.
4. Objects that are uncertain or not visibly distinguishable.

Object hierarchy and grouping rules:
1. Select only root-level, independently meaningful scene objects. Do not
   return components, attachments, contents, or subparts as separate keywords.
2. For vehicles, return only the complete vehicle. Do not return wheels,
   tires, mirrors, doors, windows, lights, license plates, seats, steering
   wheels, handlebars, or other vehicle components.
3. For containers such as shopping bags, baskets, shopping carts, suitcases,
   boxes, bins, trays, shelves, cabinets, and drawers, do not enumerate the
   many small or incidental items inside. Return the container or collection
   as one object when it is visually important.
4. For furniture, electronics, appliances, and buildings, return the complete
   parent object instead of legs, armrests, cushions, handles, screens, keys,
   buttons, ports, cables, doors, windows, roofs, balconies, signs, or other
   attached structural details.
5. For people and animals, return the whole subject. Do not return body parts,
   clothing, footwear, or accessories separately.
6. For plants and food, prefer the complete plant, tree, flower pot, dish, or
   meal instead of individual leaves, branches, flowers, fruits, ingredients,
   toppings, or small pieces.
7. For crowded collections such as racks, shelves, baskets, trays, toy boxes,
   piles, or displays, do not list every visible item.

The small-occluder exception applies only to an independent object. It never
promotes a component, attachment, accessory, body part, clothing item, or one
of many incidental container contents into a separate keyword.

Use short, concrete English nouns or noun phrases that work as SAM3 text
prompts. Return one keyword per semantic object class. Prefer recall for
significant occluders, but do not list irrelevant background objects,
clothing, or accessories. Focus only on scene-defining foreground objects
and objects necessary to preserve important occlusion relationships.

Return at most 10 keywords total. Order them by priority:
1. Objects that significantly occlude another important foreground object.
2. Main foreground objects ordered by visual importance and occupied area.
"""

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

        max_keywords = int(self._settings.get("max_keywords", 10))

        return normalize_keywords(
            payload["keywords"][:max_keywords],
            max_keywords=max_keywords,
            max_length=int(
                self._settings.get("max_keyword_length", 80)
            ),
        )
