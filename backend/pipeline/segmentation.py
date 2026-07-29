from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from ..core.helpers import (
    _bbox_from_mask,
    _inference_context,
    _normalise_mask,
)
from ..core.logging import get_logger, log_event
from .types import DetectedObject

logger = get_logger(__name__)


def save_object_masks(
    objects: Sequence[DetectedObject],
    diagnostics_directory: str | Path,
) -> None:
    """Save raw SAM3 binary masks immediately after segmentation."""
    directory = Path(diagnostics_directory)
    directory.mkdir(parents=True, exist_ok=True)
    for detected in objects:
        mask_image = Image.fromarray(
            (detected.modal_mask > 0).astype(np.uint8) * 255, mode="L"
        )
        safe_label = detected.display_label.replace(" ", "_")
        filename = f"{detected.object_id}_{safe_label}.png"
        mask_image.save(directory / filename)


def extract_raw_objects(
    image: Image.Image,
    keywords: Sequence[str],
    processor: Any,
    diagnostics_directory: str | Path | None = None,
) -> list[DetectedObject]:
    """Return one stable object record per non-empty raw SAM3 mask."""
    objects: list[DetectedObject] = []

    with torch.inference_mode(), _inference_context():
        state = processor.set_image(image)
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                log_event(
                    logger,
                    "segmentation",
                    "prompt_decision",
                    decision="skip",
                    reason="empty_keyword",
                )
                continue

            log_event(
                logger,
                "segmentation",
                "prompt_decision",
                keyword=keyword,
                decision="run",
            )

            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=keyword)
            masks = state.get("masks")
            if masks is None or len(masks) == 0:
                logger.warning("[SAM3] No object found for '%s'", keyword)
                log_event(
                    logger,
                    "segmentation",
                    "prompt_result",
                    keyword=keyword,
                    decision="no_objects",
                )
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
                    log_event(
                        logger,
                        "segmentation",
                        "mask_decision",
                        keyword=keyword,
                        mask_index=index,
                        decision="reject",
                        reason="empty_mask",
                    )
                    continue

                display_label = (
                    f"{keyword}_{index}"
                    if len(keyword_masks) > 1
                    else keyword
                )
                detected = DetectedObject(
                        object_id=f"object-{len(objects)}",
                        semantic_class=keyword,
                        display_label=display_label,
                        modal_mask=mask,
                        bbox=bbox,
                        segmentation_index=len(objects),
                    )
                objects.append(detected)
                log_event(
                    logger,
                    "segmentation",
                    "object_detected",
                    object_id=detected.object_id,
                    semantic_class=detected.semantic_class,
                    display_label=detected.display_label,
                    bbox=detected.bbox,
                    modal_area=int((detected.modal_mask > 0).sum()),
                )
                logger.debug("[SAM3] raw instance '%s'", display_label)

    if diagnostics_directory is not None and objects:
        save_object_masks(objects, diagnostics_directory)

    return objects


# Transitional notebook compatibility. Production orchestration uses the
# explicit raw-object name; the notebook migrates in its later parity step.
extract_objects = extract_raw_objects
