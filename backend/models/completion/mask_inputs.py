"""Prepare modal masks and bounding boxes for SDAmodal inference."""

from collections.abc import Sequence

import numpy as np


def prepare_modal_inputs(
    modal_masks: Sequence[np.ndarray],
    bboxes: Sequence[tuple[int, int, int, int]],
    *,
    image_shape: tuple[int, int],
    enlarge_box: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize masks and expand their existing ``(x, y, w, h)`` boxes."""
    if enlarge_box <= 0:
        raise ValueError("enlarge_box must be positive")
    if len(bboxes) != len(modal_masks):
        raise ValueError("number of bboxes must match masks")

    binary_masks = []
    expanded_bboxes = []

    for index, (modal_mask, bbox) in enumerate(zip(modal_masks, bboxes)):
        mask = np.asarray(modal_mask)
        if mask.ndim != 2:
            raise ValueError(f"mask {index} must be two-dimensional")
        if mask.shape != image_shape:
            raise ValueError(
                f"mask {index} shape must match image shape {image_shape}"
            )

        binary_mask = (mask > 0).astype(np.uint8)
        if not binary_mask.any():
            raise ValueError(f"mask {index} has no foreground pixels")

        x, y, width, height = bbox

        center_x = x + width / 2.0
        center_y = y + height / 2.0
        side = max(
            np.sqrt(width * height * enlarge_box),
            width * 1.1,
            height * 1.1,
        )
        expanded_bboxes.append(
            [
                int(center_x - side / 2.0),
                int(center_y - side / 2.0),
                int(side),
                int(side),
            ]
        )
        binary_masks.append(binary_mask)

    return (
        np.stack(binary_masks, axis=0),
        np.asarray(expanded_bboxes, dtype=np.int32),
    )
