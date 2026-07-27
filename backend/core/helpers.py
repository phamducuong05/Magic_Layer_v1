"""Small, non-model utilities used by the image pipeline."""

import base64
import io
from contextlib import nullcontext
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ..models.background_inpainting.common.masks import (
    prepare_inpaint_masks as _prepare_inpaint_masks,
    preserve_unmasked_pixels as _preserve_unmasked_pixels,
)

# Keep a small dilation margin to remove edge residue without regenerating a
# large amount of surrounding background. This was previously 0.015.
_KERNEL_SCALE = 0.0075


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


def _calc_kernel_size(image: np.ndarray, kernel_scale: float = _KERNEL_SCALE) -> tuple[int, int]:
    height, width = image.shape[:2]
    return (
        max(1, round(height * kernel_scale)),
        max(1, round(width * kernel_scale)),
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

    This function return the list of union masks.
    Example:
    masks = [mask1, mask2, mask3, mask4, mask5]
    _merge_overlapping_masks(masks) -> [mask1_union_mask2, mask3_union_mask4_union_mask5]

    NOTICE: This function is only for masks by one text keyword. 
    Don't mix masks from different keywords.
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

    def masks_overlap(first_index: int, second_index: int) -> bool:
        first_box = boxes[first_index]
        second_box = boxes[second_index]
        
        if first_box is None or second_box is None:
            return False
            
        first_x, first_y, first_width, first_height = first_box
        second_x, second_y, second_width, second_height = second_box
        
        # Fast path: check if bounding boxes overlap first
        boxes_do_overlap = (
            max(first_x, second_x)
            < min(first_x + first_width, second_x + second_width)
            and max(first_y, second_y)
            < min(first_y + first_height, second_y + second_height)
        )
        
        if not boxes_do_overlap:
            return False
            
        # Slow path: check actual pixel overlap if bounding boxes intersect
        return bool(np.any((masks[first_index] > 0) & (masks[second_index] > 0)))

    # Build connected groups. Union-find makes overlap transitive even when the
    # first and last boxes in a chain do not directly overlap each other.
    for first_index in range(len(masks)):
        for second_index in range(first_index + 1, len(masks)):
            if masks_overlap(first_index, second_index):
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


def _bbox_area(mask: np.ndarray) -> int:
    """Return the tight positive-pixel bounding-box area of a binary mask."""
    positive_y, positive_x = np.nonzero(mask)
    if positive_x.size == 0:
        return 0
    width = int(positive_x.max() - positive_x.min() + 1)
    height = int(positive_y.max() - positive_y.min() + 1)
    return width * height
