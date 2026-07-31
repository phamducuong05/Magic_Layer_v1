import math
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


def _duplicate_mask_overlap(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return symmetric overlap requiring near-equality of both masks."""
    first_bool = np.asarray(first) > 0
    second_bool = np.asarray(second) > 0
    if first_bool.shape != second_bool.shape:
        raise ValueError("duplicate mask comparison requires equal shapes")
    first_area = int(np.count_nonzero(first_bool))
    second_area = int(np.count_nonzero(second_bool))
    denominator = max(first_area, second_area)
    if denominator == 0:
        return 0.0
    intersection = int(np.count_nonzero(first_bool & second_bool))
    return intersection / denominator


def _boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    """Return whether two ``(x, y, width, height)`` boxes overlap."""
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return (
        max(first_x, second_x)
        < min(first_x + first_width, second_x + second_width)
        and max(first_y, second_y)
        < min(first_y + first_height, second_y + second_height)
    )


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
    duplicate_mask_overlap_threshold: float = 0.90,
) -> list[DetectedObject]:
    """Return one stable object record per non-empty raw SAM3 mask."""
    if not math.isfinite(duplicate_mask_overlap_threshold) or not (
        0.0 <= duplicate_mask_overlap_threshold <= 1.0
    ):
        raise ValueError(
            "duplicate mask overlap threshold must be finite and in [0, 1]"
        )
    objects: list[DetectedObject] = []
    retained_masks: list[
        tuple[np.ndarray, tuple[int, int, int, int], DetectedObject]
    ] = []

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

                duplicate_of: tuple[DetectedObject, float] | None = None
                for retained_mask, retained_bbox, retained in retained_masks:
                    if not _boxes_overlap(bbox, retained_bbox):
                        continue
                    overlap = _duplicate_mask_overlap(mask, retained_mask)
                    if overlap >= duplicate_mask_overlap_threshold:
                        duplicate_of = (retained, overlap)
                        break
                if duplicate_of is not None:
                    retained, overlap = duplicate_of
                    log_event(
                        logger,
                        "segmentation",
                        "mask_decision",
                        keyword=keyword,
                        mask_index=index,
                        decision="reject",
                        reason="duplicate_mask_overlap",
                        kept_object_id=retained.object_id,
                        kept_keyword=retained.semantic_class,
                        overlap_ratio=overlap,
                        threshold=duplicate_mask_overlap_threshold,
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
                retained_masks.append((mask, bbox, detected))
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
