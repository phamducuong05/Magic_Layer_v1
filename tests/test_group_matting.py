from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image

from backend.pipeline.grouping import group_reconstructed_objects
from backend.pipeline.roi import SquareROI
from backend.pipeline.types import DetectedObject


def _group(
    *,
    shape=(12, 12),
    modal_box=(4, 4, 2, 2),
    reconstructed=False,
):
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, width, height = modal_box
    mask[y : y + height, x : x + width] = 255
    detected = DetectedObject(
        object_id="object-0",
        semantic_class="person",
        display_label="person",
        modal_mask=mask,
        bbox=modal_box,
    )
    if reconstructed:
        detected.reconstruction_canvas = Image.new("RGB", (8, 8), "red")
    return group_reconstructed_objects([detected])[0]


def test_fallback_group_matting_uses_original_expanded_crop_and_modal_support():
    from backend.pipeline.matting import refine_objects

    image_array = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    image = Image.fromarray(image_array, mode="RGB")
    group = _group()
    matte = Mock(return_value=torch.ones((2, 2), dtype=torch.float32))

    refine_objects(
        image,
        [group],
        matte,
        context_ratio=1.5,
        support_dilation_pixels=2,
    )

    supplied_crop = matte.call_args.args[0]
    assert group.matting_source is supplied_crop
    assert group.matting_roi is not None
    assert supplied_crop.size == (8, 8)
    assert np.array_equal(
        np.asarray(supplied_crop), np.asarray(image.crop(group.matting_roi.box))
    )
    assert group.soft_alpha.shape == (12, 12)
    assert group.soft_alpha[4, 4] > 0
    assert group.soft_alpha[group.matting_roi.y, group.matting_roi.x] == 0


def test_reconstructed_group_matting_uses_composed_rgb_and_amodal_support():
    from backend.pipeline.matting import refine_objects

    image = Image.new("RGB", (12, 12), "black")
    group = _group(reconstructed=True)
    group.amodal_mask = group.modal_mask > 0
    group.amodal_mask[8, 8] = True
    group.composed_roi = SquareROI(4, 4, 5, 12, 12)
    group.composed_source = Image.new("RGB", (5, 5), (20, 180, 60))
    matte = Mock(return_value=torch.ones((7, 7), dtype=torch.float32))

    refine_objects(
        image,
        [group],
        matte,
        context_ratio=0.2,
        support_dilation_pixels=1,
    )

    supplied_crop = matte.call_args.args[0]
    assert supplied_crop is group.matting_source
    assert supplied_crop.size == (7, 7)
    assert group.matting_roi.box == (3, 3, 10, 10)
    supplied_array = np.asarray(supplied_crop)
    assert np.all(supplied_array[1:6, 1:6] == (20, 180, 60))
    assert np.all(supplied_array[0, 0] == 0)  # Real original context.
    assert group.soft_alpha.shape == (12, 12)
    assert group.soft_alpha[8, 8] > 0
    assert group.soft_alpha[2, 2] == 0
