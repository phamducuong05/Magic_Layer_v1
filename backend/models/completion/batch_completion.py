"""In-memory SDAmodal batch completion from an existing DIFT pyramid."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import inference
from .mask_inputs import prepare_modal_inputs


def complete_masks_from_features(
    model: Any,
    feature_pyramid: Any,
    modal_masks: Sequence[np.ndarray],
    bboxes: Sequence[tuple[int, int, int, int]],
    *,
    image_shape: tuple[int, int],
    config: Mapping[str, Any],
    device: str,
) -> list[np.ndarray]:
    """Return full-image amodal masks while preserving input order."""
    if not modal_masks:
        return []

    inmodal, expanded_bboxes = prepare_modal_inputs(
        modal_masks,
        bboxes,
        image_shape=image_shape,
        enlarge_box=config["data"]["enlarge_box"],
    )
    categories = np.ones(len(modal_masks), dtype=np.int32)

    patches = inference.infer_amodal_aw_sdm(
        model,
        feature_pyramid,
        inmodal,
        categories,
        expanded_bboxes,
        use_rgb=config["model"]["use_rgb"],
        th=0.5,
        input_size=config["data"]["input_size"],
        min_input_size=16,
        interp="nearest",
        args=None,
        device=device,
    )

    height, width = image_shape
    restored_masks = inference.patch_to_fullimage(
        patches,
        expanded_bboxes,
        height,
        width,
        interp="linear",
    )

    return [
        ((restored_mask > 0) | (modal_mask > 0)).astype(np.uint8)
        for restored_mask, modal_mask in zip(restored_masks, inmodal)
    ]
