from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from backend.pipeline.reconstruction import reconstruct_objects
from backend.pipeline.types import DetectedObject


def _object(
    object_id: str,
    *,
    x_offset: int = 0,
) -> DetectedObject:
    modal = np.zeros((12, 16), dtype=np.uint8)
    modal[3:7, x_offset + 2 : x_offset + 5] = 255
    amodal = modal > 0
    amodal[3:7, x_offset + 5 : x_offset + 7] = True
    detected = DetectedObject(
        object_id=object_id,
        semantic_class=object_id,
        display_label=object_id,
        modal_mask=modal,
        bbox=(x_offset + 2, 3, 3, 4),
    )
    detected.amodal_mask = amodal
    detected.completion_hole_mask = amodal & ~(modal > 0)
    detected.completion_hole_area = int(
        np.count_nonzero(detected.completion_hole_mask)
    )
    detected.reconstruction_mask = detected.completion_hole_mask.copy()
    return detected


def _source_image(color=(30, 40, 50)) -> Image.Image:
    return Image.new("RGB", (16, 12), color)


def _valid_result(
    color=(80, 90, 100),
) -> Callable[[Image.Image, Image.Image, str], Image.Image]:
    def reconstruct(
        source: Image.Image, mask: Image.Image, _prompt: str
    ) -> Image.Image:
        result = np.asarray(source).copy()
        result[np.asarray(mask) > 0] = color
        return Image.fromarray(result, mode="RGB")

    return reconstruct


def test_valid_low_texture_reconstruction_is_retained():
    detected = _object("person")

    reconstruct_objects(
        _source_image(),
        [detected],
        _valid_result((70, 70, 70)),
        context_ratio=0.0,
        blend_allowance_ratio=0.0,
    )

    assert detected.reconstruction_canvas is not None
    assert detected.reconstruction_canvas.mode == "RGB"
    assert detected.reconstruction_failure_stage is None
    assert detected.reconstruction_failure_reason is None


@pytest.mark.parametrize(
    ("invalid_result", "expected_stage"),
    [
        (lambda source: [source], "result_type"),
        (lambda source: source.resize((2, 2)), "crop_size_restoration"),
        (lambda source: source.convert("CMYK"), "result_channels"),
    ],
)
def test_invalid_result_contract_uses_modal_fallback(
    invalid_result, expected_stage
):
    detected = _object("person")

    reconstruct_objects(
        _source_image(),
        [detected],
        lambda source, _mask, _prompt: invalid_result(source),
        context_ratio=0.0,
        blend_allowance_ratio=0.0,
    )

    assert detected.reconstruction_canvas is None
    assert detected.reconstruction_roi is None
    assert detected.reconstruction_failure_stage == expected_stage
    assert detected.reconstruction_failure_reason


def test_change_outside_permitted_region_is_restored_to_source():
    detected = _object("person")
    source = _source_image()

    def reconstruct(source, mask, _prompt):
        result = np.asarray(source).copy()
        result[np.asarray(mask) > 0] = (70, 80, 90)
        result[0, 0] = (200, 10, 10)
        return Image.fromarray(result, mode="RGB")

    reconstruct_objects(
        source,
        [detected],
        reconstruct,
        context_ratio=0.0,
        blend_allowance_ratio=0.0,
    )

    assert detected.reconstruction_canvas is not None
    assert detected.reconstruction_failure_stage is None
    assert detected.reconstruction_canvas.getpixel((0, 0)) == source.getpixel(
        (detected.reconstruction_roi.x, detected.reconstruction_roi.y)
    )
    mask_crop = np.asarray(
        detected.reconstruction_mask[
            detected.reconstruction_roi.y : detected.reconstruction_roi.y
            + detected.reconstruction_roi.size,
            detected.reconstruction_roi.x : detected.reconstruction_roi.x
            + detected.reconstruction_roi.size,
        ]
    )
    changed_y, changed_x = np.argwhere(mask_crop)[0]
    assert detected.reconstruction_canvas.getpixel(
        (int(changed_x), int(changed_y))
    ) == (70, 80, 90)


def test_blend_allowance_accepts_adjacent_boundary_changes():
    detected = _object("person")

    def reconstruct(source, mask, _prompt):
        result = np.asarray(source).copy()
        mask_array = np.asarray(mask) > 0
        result[mask_array] = (70, 80, 90)
        first_y, first_x = np.argwhere(mask_array)[0]
        result[first_y, first_x - 1] = (65, 75, 85)
        return Image.fromarray(result, mode="RGB")

    reconstruct_objects(
        _source_image(),
        [detected],
        reconstruct,
        context_ratio=0.0,
        blend_allowance_ratio=0.2,
    )

    assert detected.reconstruction_canvas is not None
    assert detected.reconstruction_failure_stage is None


def test_unusable_blank_hole_is_rejected_but_black_object_is_allowed():
    ordinary = _object("ordinary")
    black_object = _object("black", x_offset=7)
    image = np.full((12, 16, 3), (30, 40, 50), dtype=np.uint8)
    image[black_object.modal_mask > 0] = 0

    def reconstruct(source, mask, _prompt):
        result = np.asarray(source).copy()
        result[np.asarray(mask) > 0] = 0
        return Image.fromarray(result, mode="RGB")

    reconstruct_objects(
        Image.fromarray(image, mode="RGB"),
        [ordinary, black_object],
        reconstruct,
        context_ratio=0.0,
        blend_allowance_ratio=0.0,
    )

    assert ordinary.reconstruction_canvas is None
    assert ordinary.reconstruction_failure_stage == "completion_hole"
    assert black_object.reconstruction_canvas is not None
    assert black_object.reconstruction_failure_stage is None


def test_one_failure_does_not_discard_successful_sibling():
    failed = _object("failed")
    successful = _object("successful", x_offset=7)

    class GenerationFailure(RuntimeError):
        stage = "generation_512"

    def reconstruct(source, mask, prompt):
        if "failed" in prompt:
            raise GenerationFailure("CUDA inference failed")
        return _valid_result()(source, mask, prompt)

    reconstruct_objects(
        _source_image(),
        [failed, successful],
        reconstruct,
        context_ratio=0.0,
        blend_allowance_ratio=0.0,
    )

    assert failed.reconstruction_canvas is None
    assert failed.reconstruction_roi is None
    assert failed.reconstruction_failure_stage == "generation_512"
    assert "CUDA inference failed" in failed.reconstruction_failure_reason
    assert successful.reconstruction_canvas is not None
    assert successful.reconstruction_failure_stage is None


def test_reconstruct_objects_uses_one_batch_call_and_isolates_outcomes():
    failed = _object("failed")
    successful = _object("successful", x_offset=7)
    batch_calls = []

    class GenerationFailure(RuntimeError):
        stage = "generation_512"

    def reconstruct_many(requests):
        batch_calls.append(requests)
        source, mask, _prompt = requests[1]
        return [
            GenerationFailure("synthetic failure"),
            _valid_result()(source, mask, "successful"),
        ]

    reconstruct_objects(
        _source_image(),
        [failed, successful],
        Mock(side_effect=AssertionError("single path must not run")),
        reconstruct_many=reconstruct_many,
        context_ratio=0.0,
    )

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 2
    assert failed.reconstruction_canvas is None
    assert failed.reconstruction_failure_stage == "generation_512"
    assert successful.reconstruction_canvas is not None
    assert successful.reconstruction_failure_stage is None
