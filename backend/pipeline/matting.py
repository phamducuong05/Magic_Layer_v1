"""Explicit-callable hard-mask to soft-alpha refinement."""

from collections.abc import Callable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from ..core.helpers import _inference_context
from .types import DetectedObject

THRESHOLD_ALPHA = 0.005


def refine_masks(
    image_np: np.ndarray,
    raw_masks: Sequence[np.ndarray],
    matte: Callable[[Image.Image], torch.Tensor],
) -> list[np.ndarray]:
    """Convert hard masks into soft alpha mattes using the supplied callable."""
    soft_alphas: list[np.ndarray] = []

    with torch.inference_mode(), _inference_context():
        for mask in raw_masks:
            guided_image = image_np.copy()
            guided_image[mask == 0] = 0
            alpha = matte(Image.fromarray(guided_image))

            support = cv2.dilate(
                mask, np.ones((5, 5), dtype=np.uint8), iterations=1
            ) > 0
            valid = (alpha > THRESHOLD_ALPHA) & torch.from_numpy(support).to(
                alpha.device
            )
            alpha[~valid] = 0.0
            soft_alphas.append(
                np.clip(alpha.cpu().to(torch.float64).numpy(), 0.0, 1.0)
            )

    return soft_alphas


def refine_objects(
    image_np: np.ndarray,
    objects: Sequence[DetectedObject],
    matte: Callable[[Image.Image], torch.Tensor],
) -> None:
    """Attach one modal soft alpha to each object in input order."""
    soft_alphas = refine_masks(
        image_np, [detected.modal_mask for detected in objects], matte
    )
    for detected, soft_alpha in zip(objects, soft_alphas):
        detected.soft_alpha = soft_alpha
