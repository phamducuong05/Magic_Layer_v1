"""Tests for preparing modal masks for SDAmodal inference."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package

from backend.models.completion.mask_inputs import prepare_modal_inputs


def test_prepare_modal_inputs_normalizes_masks_and_preserves_order():
    first = np.zeros((8, 10), dtype=bool)
    first[2:6, 3:7] = True
    second = np.zeros((8, 10), dtype=np.uint8)
    second[1:3, 8:10] = 255

    inmodal, bboxes = prepare_modal_inputs(
        [first, second],
        [(3, 2, 4, 4), (8, 1, 2, 2)],
        image_shape=(8, 10),
        enlarge_box=4.0,
    )

    assert inmodal.shape == (2, 8, 10)
    assert inmodal.dtype == np.uint8
    np.testing.assert_array_equal(inmodal[0], first.astype(np.uint8))
    np.testing.assert_array_equal(inmodal[1], (second > 0).astype(np.uint8))
    np.testing.assert_array_equal(
        bboxes,
        np.array([[1, 0, 8, 8], [7, 0, 4, 4]], dtype=np.int32),
    )


def test_prepare_modal_inputs_expands_supplied_bbox_instead_of_recalculating():
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[4, 4] = 1

    _, bboxes = prepare_modal_inputs(
        [mask],
        [(1, 1, 2, 2)],
        image_shape=(8, 10),
        enlarge_box=4.0,
    )

    np.testing.assert_array_equal(
        bboxes,
        np.array([[0, 0, 4, 4]], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.zeros((1, 8, 10), dtype=np.uint8), "mask 0 must be two-dimensional"),
        (np.zeros((7, 10), dtype=np.uint8), "mask 0 shape must match image shape"),
        (np.zeros((8, 10), dtype=np.uint8), "mask 0 has no foreground pixels"),
    ],
)
def test_prepare_modal_inputs_rejects_invalid_masks(mask, message):
    with pytest.raises(ValueError, match=message):
        prepare_modal_inputs(
            [mask],
            [(0, 0, 1, 1)],
            image_shape=(8, 10),
            enlarge_box=3.0,
        )


def test_prepare_modal_inputs_requires_one_bbox_per_mask():
    mask = np.ones((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="number of bboxes must match masks"):
        prepare_modal_inputs(
            [mask],
            [],
            image_shape=(2, 2),
            enlarge_box=3.0,
        )


def test_prepare_modal_inputs_rejects_nonpositive_enlargement():
    mask = np.ones((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="enlarge_box must be positive"):
        prepare_modal_inputs(
            [mask],
            [(0, 0, 2, 2)],
            image_shape=(2, 2),
            enlarge_box=0,
        )
