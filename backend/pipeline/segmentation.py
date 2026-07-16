"""Text-guided extraction of individual raw SAM3 masks."""

import logging
from typing import Any, Sequence

import torch
from PIL import Image

from ..core.helpers import (
    _bbox_from_mask,
    _inference_context,
    _normalise_mask,
)
from .types import DetectedObject

logger = logging.getLogger(__name__)


def extract_raw_objects(
    image: Image.Image, keywords: Sequence[str], processor: Any
) -> list[DetectedObject]:
    """Return one stable object record per non-empty raw SAM3 mask."""
    objects: list[DetectedObject] = []

    with torch.inference_mode(), _inference_context():
        state = processor.set_image(image)
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue

            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=keyword)
            masks = state.get("masks")
            if masks is None or len(masks) == 0:
                logger.warning("[SAM3] No object found for '%s'", keyword)
                continue

            keyword_masks = [
                _normalise_mask(mask, image.size) for mask in masks
            ]

            for index, mask in enumerate(keyword_masks):
                bbox = _bbox_from_mask(mask)
                if bbox is None:
                    logger.warning(
                        "[SAM3] Ignoring empty raw mask for '%s'", keyword
                    )
                    continue

                display_label = (
                    f"{keyword}_{index}"
                    if len(keyword_masks) > 1
                    else keyword
                )
                objects.append(
                    DetectedObject(
                        object_id=f"object-{len(objects)}",
                        semantic_class=keyword,
                        display_label=display_label,
                        modal_mask=mask,
                        bbox=bbox,
                        segmentation_index=len(objects),
                    )
                )
                logger.info("[SAM3] raw instance '%s'", display_label)

    return objects


# Transitional notebook compatibility. Production orchestration uses the
# explicit raw-object name; the notebook migrates in its later parity step.
extract_objects = extract_raw_objects
