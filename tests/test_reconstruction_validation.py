from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from backend.pipeline.reconstruction import reconstruct_objects
from backend.pipeline.reconstruction.validate_reconstruction import (
    reconstruction_color_metrics,
)
from backend.pipeline.roi import SquareROI
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


def test_reconstruction_uses_generation_mask_but_preserves_composition_mask():
    detected = _object("person")
    composition = detected.reconstruction_mask.copy()
    generation = composition.copy()
    generation[3, 8] = True  # Outside tight amodal support.
    detected.reconstruction_generation_mask = generation
    captured = {}

    def reconstruct(source, mask, prompt):
        captured["mask"] = np.asarray(mask) > 0
        captured["prompt"] = prompt
        result = np.asarray(source).copy()
        result[captured["mask"]] = (70, 80, 90)
        return Image.fromarray(result, mode="RGB")

    reconstruct_objects(
        _source_image(),
        [detected],
        reconstruct,
        context_ratio=0.0,
    )

    roi = detected.reconstruction_roi
    # Generation-only support cannot enlarge the target-centric ROI.
    assert not (roi.x <= 8 < roi.x + roi.size)
    expected_generation = generation[
        roi.y : roi.y + roi.size, roi.x : roi.x + roi.size
    ]
    assert np.array_equal(captured["mask"], expected_generation)
    assert np.array_equal(detected.reconstruction_mask, composition)


def test_reconstruction_keeps_full_roi_output_except_protected_modal_pixels():
    detected = _object("person")
    roi = SquareROI(2, 2, 5, 16, 12)
    detected.reconstruction_input_roi = roi
    detected.reconstruction_generation_mask = (
        detected.reconstruction_mask.copy()
    )

    accepted = np.zeros_like(detected.reconstruction_mask, dtype=bool)
    accepted[2:7, 2:7] = True
    accepted[detected.modal_mask > 0] = False
    protected = detected.modal_mask > 0
    protected[2, 2] = True  # Foreign modal outside the target bbox.
    accepted[protected] = False
    detected.reconstruction_accepted_rgb_mask = accepted
    detected.reconstruction_protected_mask = protected

    captured = {}

    def reconstruct(source, mask, _prompt):
        captured["generation"] = np.asarray(mask) > 0
        return Image.new("RGB", source.size, (200, 10, 10))

    source = _source_image()
    reconstruct_objects(
        source,
        [detected],
        reconstruct,
        context_ratio=0.0,
        blend_allowance_ratio=0.0,
    )

    canvas = np.asarray(detected.reconstruction_canvas)
    generation_crop = detected.reconstruction_generation_mask[2:7, 2:7]
    assert np.array_equal(captured["generation"], generation_crop)
    # This point is outside generation but inside accepted full-ROI output.
    assert not generation_crop[0, 1]
    assert tuple(canvas[0, 1]) == (200, 10, 10)
    # Protected foreign modal and visible target modal retain source RGB.
    assert tuple(canvas[0, 0]) == (30, 40, 50)
    assert tuple(canvas[1, 0]) == (30, 40, 50)
    assert np.array_equal(detected.reconstruction_write_mask, accepted)


def test_reconstruction_prompt_names_target_and_occluder_classes():
    detected = _object("person")
    detected.occluder_classes = {"book", "camera"}
    prompts = []

    reconstruct_objects(
        _source_image(),
        [detected],
        lambda source, mask, prompt: (
            prompts.append(prompt) or _valid_result()(source, mask, prompt)
        ),
        context_ratio=0.0,
        prompt_template=(
            "Reconstruct only the hidden continuation of the {target} "
            "behind {occluders}. Do not recreate {occluders}."
        ),
    )

    assert "person" in prompts[0]
    assert "book and camera" in prompts[0]
    assert "Do not recreate" in prompts[0]


def test_reconstruction_prompt_appends_configured_style_hint():
    detected = _object("person")
    prompts = []

    reconstruct_objects(
        _source_image(),
        [detected],
        lambda source, mask, prompt: (
            prompts.append(prompt) or _valid_result()(source, mask, prompt)
        ),
        context_ratio=0.0,
        style_hint="flat vector illustration, crisp edges, solid colors",
    )

    assert prompts[0].endswith(
        "flat vector illustration, crisp edges, solid colors"
    )


def test_validated_reconstruction_rgb_is_not_color_refined():
    detected = _object("person")
    roi = SquareROI(2, 2, 5, 16, 12)
    detected.reconstruction_input_roi = roi
    detected.reconstruction_generation_mask = (
        detected.reconstruction_mask.copy()
    )
    accepted = np.zeros_like(detected.reconstruction_mask, dtype=bool)
    accepted[2:7, 2:7] = True
    accepted[detected.modal_mask > 0] = False
    detected.reconstruction_accepted_rgb_mask = accepted
    source = np.full((12, 16, 3), (250, 250, 250), dtype=np.uint8)
    source[detected.modal_mask > 0] = (220, 20, 20)

    reconstruct_objects(
        Image.fromarray(source, mode="RGB"),
        [detected],
        _valid_result((10, 80, 220)),
        context_ratio=0.0,
    )

    generation_crop = detected.reconstruction_generation_mask[
        roi.y : roi.y + roi.size,
        roi.x : roi.x + roi.size,
    ]
    canvas = np.asarray(detected.reconstruction_canvas)
    assert np.all(canvas[generation_crop] == (10, 80, 220))


def test_soft_color_metrics_include_generated_detail_ratio():
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    source[:, ::2] = 255
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, 4:] = True
    metrics = reconstruction_color_metrics(
        Image.fromarray(source, mode="RGB"),
        source_crop=Image.fromarray(source, mode="RGB"),
        composition_mask=mask,
        modal_mask=~mask,
        occluder_mask=mask,
    )

    assert metrics["generated_detail_ratio"] is not None


def test_reconstruction_can_save_per_object_debug_artifacts(tmp_path):
    detected = _object("person")

    reconstruct_objects(
        _source_image(),
        [detected],
        _valid_result(),
        context_ratio=0.0,
        diagnostics_directory=tmp_path,
        debug_artifacts_provider=lambda: [
            {
                "base_output_512": Image.new("RGB", (512, 512), "red"),
                "sr_output": Image.new("RGB", (2048, 2048), "blue"),
            }
        ],
    )

    object_directory = tmp_path / "person"
    expected_names = [
        "01_initial_modal_mask.png",
        "02_completed_amodal_mask.png",
        "03_completion_hole.png",
        "04_amodal_bbox_mask.png",
        "05_target_bbox_mask.png",
        "06_directional_seed.png",
        "07_filtered_reconstruction_mask.png",
        "08_composition_mask.png",
        "09_source.png",
        "10_roi_real_pixels.png",
        "11_target_modal_protected.png",
        "12_occluder_mask.png",
        "13_full_occluder_in_roi.png",
        "14_generation_before_dilation.png",
        "15_generation_after_dilation.png",
        "16_generation_mask.png",
        "17_foreign_modal_inside_bbox.png",
        "18_foreign_modal_outside_bbox.png",
        "19_protected_source_pixels.png",
        "20_accepted_model_rgb_mask.png",
        "21_base_output_512.png",
        "22_sr_output.png",
        "23_model_output.png",
        "24_validated_output.png",
    ]
    actual_names = sorted(path.name for path in object_directory.iterdir())
    assert expected_names == actual_names
    initial_modal = np.asarray(
        Image.open(object_directory / "01_initial_modal_mask.png")
    ) > 0
    completed_amodal = np.asarray(
        Image.open(object_directory / "02_completed_amodal_mask.png")
    ) > 0
    amodal_bbox = np.asarray(
        Image.open(object_directory / "04_amodal_bbox_mask.png")
    ) > 0
    assert np.all(completed_amodal[initial_modal])
    assert np.any(completed_amodal & ~initial_modal)
    assert np.all(amodal_bbox[completed_amodal])
    assert not (object_directory / "color_refined_output.png").exists()
