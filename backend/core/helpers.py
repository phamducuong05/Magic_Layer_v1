"""Small, non-model utilities used by the image pipeline."""

import base64
import io
from contextlib import nullcontext
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

_KERNEL_SCALE = 0.015


def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    rows = np.any(mask > 0, axis=1)
    columns = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(columns)[0][[0, -1]]
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def _calc_kernel_size(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return (
        max(1, round(height * _KERNEL_SCALE)),
        max(1, round(width * _KERNEL_SCALE)),
    )


def _inference_context():
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _normalise_mask(mask, size: tuple[int, int]) -> np.ndarray:
    """Return a binary uint8 SAM mask at the original image resolution."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask).squeeze()
    width, height = size
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return (mask > 0).astype(np.uint8) * 255


def _merge_overlapping_masks(masks: List[np.ndarray]) -> List[np.ndarray]:
    """Union masks whose bounding boxes overlap, including transitive overlaps.

    This function is intended for masks produced by one text keyword. Callers
    should not mix masks from different keywords.
    """
    if len(masks) < 2:
        return masks

    boxes = [_bbox_from_mask(mask) for mask in masks]
    parents = list(range(len(masks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    def boxes_overlap(
        first: Tuple[int, int, int, int] | None,
        second: Tuple[int, int, int, int] | None,
    ) -> bool:
        if first is None or second is None:
            return False
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        return (
            max(first_x, second_x)
            < min(first_x + first_width, second_x + second_width)
            and max(first_y, second_y)
            < min(first_y + first_height, second_y + second_height)
        )

    # Build connected groups. Union-find makes overlap transitive even when the
    # first and last boxes in a chain do not directly overlap each other.
    for first_index in range(len(masks)):
        for second_index in range(first_index + 1, len(masks)):
            if boxes_overlap(boxes[first_index], boxes[second_index]):
                union(first_index, second_index)

    grouped_indices: dict[int, List[int]] = {}
    for index in range(len(masks)):
        grouped_indices.setdefault(find(index), []).append(index)

    merged_masks: List[np.ndarray] = []
    for indices in grouped_indices.values():
        merged = np.zeros_like(masks[indices[0]], dtype=np.uint8)
        for index in indices:
            merged = np.maximum(merged, masks[index].astype(np.uint8))
        merged_masks.append(merged)
    return merged_masks
