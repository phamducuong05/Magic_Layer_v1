"""Unit tests for the image-to-components orchestration pipeline.

These tests use small deterministic fakes. They never load SAM3, BiRefNet, LaMa,
or model weights, so failures point to pipeline wiring rather than model quality.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="the backend requires PyTorch")
pytest.importorskip("cv2", reason="the image pipeline requires OpenCV")

from backend import image_processor as pipeline
from backend.core.helpers import _prepare_inpaint_masks, _preserve_unmasked_pixels
from backend.core.occlusion import PairDecision
from backend.pipeline import background as background_stage
from backend.pipeline import completion as completion_stage
from backend.pipeline import layers as layer_stage
from backend.pipeline import matting as matting_stage
from backend.pipeline import orchestrator as pipeline_orchestrator
from backend.pipeline import reconstruction as reconstruction_stage
from backend.pipeline import roi as roi_stage
from backend.pipeline import segmentation as segmentation_stage


COMPLETION_LIMITS = {
    "max_area_growth_ratio": 4.0,
    "max_bbox_growth_ratio": 9.0,
}


@pytest.fixture
def rgb_image() -> Image.Image:
    """Return a small image with a visible square component."""
    array = np.full((6, 8, 3), 20, dtype=np.uint8)
    array[1:5, 2:6] = (180, 80, 40)
    return Image.fromarray(array, mode="RGB")


class FakeSamProcessor:
    """Minimal stateful replacement for the SAM3 image processor."""

    def __init__(self):
        self.prompts = []
        self.reset_count = 0
        self.image = None

    def set_image(self, image):
        self.image = image
        return {}

    def reset_all_prompts(self, state):
        self.reset_count += 1

    def set_text_prompt(self, state, prompt):
        self.prompts.append(prompt)
        if prompt == "missing":
            state.update(masks=[], scores=[])
            return state

        mask = torch.zeros((1, 6, 8), dtype=torch.float32)
        mask[:, 1:5, 2:6] = 1
        state.update(masks=[mask], scores=torch.tensor([0.91]))
        return state


def _install_matting_fake(monkeypatch, alpha):
    del monkeypatch
    return Mock(return_value=alpha)


def _install_inpainting_fake(monkeypatch, output):
    del monkeypatch
    return Mock(return_value=output)


def _detected_object(object_id, semantic_class, bbox):
    mask = np.zeros((20, 20), dtype=np.uint8)
    x, y, width, height = bbox
    mask[y : y + height, x : x + width] = 255
    return pipeline.DetectedObject(
        object_id=object_id,
        semantic_class=semantic_class,
        display_label=object_id,
        modal_mask=mask,
        bbox=bbox,
    )


def test_square_roi_expands_support_with_configured_context():
    support = np.zeros((10, 12), dtype=bool)
    support[3:9, 4:8] = True

    roi = roi_stage.square_roi_from_support(support, context_ratio=0.25)

    assert (roi.x, roi.y, roi.size) == (1, 1, 9)
    assert roi.image_size == (12, 10)
    assert roi.padding == (0, 0, 0, 0)


def test_square_roi_preserves_border_padding_and_restores_coordinates():
    support = np.zeros((5, 6), dtype=bool)
    support[0:2, 0:2] = True

    roi = roi_stage.square_roi_from_support(support, context_ratio=0.5)
    cropped = np.ones((roi.size, roi.size), dtype=np.uint8)
    restored = roi_stage.restore_array(cropped, roi)

    assert (roi.x, roi.y, roi.size) == (-1, -1, 4)
    assert roi.padding == (1, 1, 0, 0)
    assert roi.inner_box == (1, 1, 4, 4)
    assert restored.shape == support.shape
    assert np.all(restored[0:3, 0:3] == 1)
    assert np.count_nonzero(restored) == 9


def test_roi_crops_image_and_mask_with_identical_padding():
    image_array = np.arange(5 * 6, dtype=np.uint8).reshape(5, 6)
    image = Image.fromarray(image_array, mode="L")
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[0, 0] = 255
    roi = roi_stage.SquareROI(-1, -1, 4, 6, 5)

    image_crop = np.asarray(roi_stage.crop_image(image, roi))
    mask_crop = roi_stage.crop_array(mask, roi)

    assert image_crop.shape == mask_crop.shape == (4, 4)
    assert image_crop[1, 1] == image_array[0, 0]
    assert mask_crop[1, 1] == 255
    assert np.all(image_crop[0, :] == 0)
    assert np.all(mask_crop[:, 0] == 0)


def test_preserve_unmasked_pixels_changes_only_masked_region():
    original = Image.new("RGB", (4, 4), (255, 0, 0))
    generated = Image.new("RGB", (4, 4), (0, 0, 255))
    mask_array = np.zeros((4, 4), dtype=np.uint8)
    mask_array[1:3, 1:3] = 255

    result = _preserve_unmasked_pixels(
        original, generated, Image.fromarray(mask_array, mode="L")
    )
    result_array = np.asarray(result)

    assert np.all(result_array[mask_array == 0] == (255, 0, 0))
    assert np.all(result_array[mask_array > 0] == (0, 0, 255))


def test_prepare_inpaint_masks_generates_beyond_blend_boundary():
    mask_array = np.zeros((31, 31), dtype=np.uint8)
    mask_array[15, 15] = 255

    generation_mask, blend_mask = _prepare_inpaint_masks(
        Image.fromarray(mask_array, mode="L"),
        generation_expansion=17,
        composition_expansion=11,
        feather_radius=2.0,
    )

    generation_array = np.asarray(generation_mask)
    blend_array = np.asarray(blend_mask)
    assert np.count_nonzero(generation_array == 255) > np.count_nonzero(
        blend_array == 255
    )
    assert generation_array[15, 15] == 255
    assert blend_array[15, 15] >= 250


def test_extract_objects_uses_each_nonempty_prompt(monkeypatch, rgb_image):
    processor = FakeSamProcessor()
    del monkeypatch

    objects = segmentation_stage.extract_objects(
        rgb_image, [" component ", "", "missing"], processor
    )

    assert processor.image is rgb_image
    assert processor.prompts == ["component", "missing"]
    assert processor.reset_count == 2
    assert len(objects) == 1
    detected = objects[0]
    assert detected.object_id == "object-0"
    assert detected.semantic_class == "component"
    assert detected.display_label == "component"
    assert detected.bbox == (2, 1, 4, 4)
    assert detected.modal_mask.shape == (6, 8)
    assert detected.modal_mask.dtype == np.uint8
    assert set(np.unique(detected.modal_mask)) == {0, 255}


def test_extract_objects_merges_before_calculating_bbox(monkeypatch, rgb_image):
    processor = FakeSamProcessor()

    def two_masks(state, prompt):
        first = torch.zeros((1, 6, 8), dtype=torch.float32)
        first[:, 1:4, 1:4] = 1
        second = torch.zeros((1, 6, 8), dtype=torch.float32)
        second[:, 2:5, 3:6] = 1
        state.update(masks=[first, second], scores=torch.tensor([0.9, 0.8]))
        return state

    processor.set_text_prompt = two_masks
    del monkeypatch

    objects = segmentation_stage.extract_objects(
        rgb_image, ["button"], processor
    )

    assert len(objects) == 1
    detected = objects[0]
    assert detected.semantic_class == "button"
    assert detected.display_label == "button"
    assert detected.bbox == (1, 1, 5, 4)
    assert np.all(detected.modal_mask[1:4, 1:4] == 255)
    assert np.all(detected.modal_mask[2:5, 3:6] == 255)


def test_extract_objects_keeps_nonoverlapping_same_keyword(
    monkeypatch, rgb_image
):
    processor = FakeSamProcessor()

    def two_masks(state, prompt):
        first = torch.zeros((1, 6, 8), dtype=torch.float32)
        first[:, 0:2, 0:2] = 1
        second = torch.zeros((1, 6, 8), dtype=torch.float32)
        second[:, 4:6, 6:8] = 1
        state.update(masks=[first, second], scores=torch.tensor([0.9, 0.8]))
        return state

    processor.set_text_prompt = two_masks
    del monkeypatch

    objects = segmentation_stage.extract_objects(
        rgb_image, ["button"], processor
    )

    assert [detected.object_id for detected in objects] == [
        "object-0",
        "object-1",
    ]
    assert [detected.semantic_class for detected in objects] == [
        "button",
        "button",
    ]
    assert [detected.display_label for detected in objects] == [
        "button_0",
        "button_1",
    ]
    assert [detected.bbox for detected in objects] == [
        (0, 0, 2, 2),
        (6, 4, 2, 2),
    ]


def test_link_overlap_partners_records_cross_class_relationship():
    person = _detected_object("person-1", "person", (0, 0, 10, 10))
    chair = _detected_object("chair-1", "chair", (5, 5, 10, 10))

    pairs = completion_stage.link_overlap_partners([person, chair])

    assert pairs == [("person-1", "chair-1")]
    assert person.overlap_partner_ids == {"chair-1"}
    assert chair.overlap_partner_ids == {"person-1"}


def test_link_overlap_partners_ignores_same_class_and_separate_objects():
    first_person = _detected_object("person-1", "person", (0, 0, 10, 10))
    second_person = _detected_object("person-2", "person", (5, 5, 10, 10))
    chair = _detected_object("chair-1", "chair", (15, 15, 2, 2))

    pairs = completion_stage.link_overlap_partners(
        [first_person, second_person, chair]
    )

    assert pairs == []
    assert first_person.overlap_partner_ids == set()
    assert second_person.overlap_partner_ids == set()
    assert chair.overlap_partner_ids == set()


def test_link_overlap_partners_keeps_multiple_partners_unique():
    person = _detected_object("person-1", "person", (0, 0, 10, 10))
    chair = _detected_object("chair-1", "chair", (1, 1, 5, 5))
    table = _detected_object("table-1", "table", (2, 2, 5, 5))

    pairs = completion_stage.link_overlap_partners([person, chair, table])

    assert pairs == [
        ("person-1", "chair-1"),
        ("person-1", "table-1"),
        ("chair-1", "table-1"),
    ]
    assert person.overlap_partner_ids == {"chair-1", "table-1"}
    assert chair.overlap_partner_ids == {"person-1", "table-1"}
    assert table.overlap_partner_ids == {"person-1", "chair-1"}


def test_complete_overlapping_objects_skips_model_without_overlap(
    monkeypatch, rgb_image
):
    del monkeypatch
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    completion_model = Mock()
    candidates = completion_stage.get_completion_candidates([person])

    completion_stage.complete_objects(
        rgb_image, candidates, completion_model, **COMPLETION_LIMITS
    )

    completion_model.complete.assert_not_called()
    assert person.amodal_mask is None
    assert person.completion_hole_area is None


def test_complete_overlapping_objects_batches_and_maps_results(
    monkeypatch, rgb_image
):
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    table = _detected_object("table-1", "table", (6, 0, 2, 2))
    person.overlap_partner_ids.add(chair.object_id)
    chair.overlap_partner_ids.add(person.object_id)

    person_amodal = person.modal_mask.copy()
    person_amodal[0, 4] = 255
    chair_amodal = chair.modal_mask.copy()
    chair_amodal[2, 6] = 2
    completion_model = Mock()
    completion_model.complete.return_value = [person_amodal, chair_amodal]
    del monkeypatch

    completion_stage.complete_objects(
        rgb_image,
        completion_stage.get_completion_candidates([person, chair, table]),
        completion_model,
        **COMPLETION_LIMITS,
    )

    completion_model.complete.assert_called_once()
    completed_image, modal_masks, bboxes = completion_model.complete.call_args.args
    assert completed_image is rgb_image
    assert modal_masks == [person.modal_mask, chair.modal_mask]
    assert bboxes == [person.bbox, chair.bbox]
    assert np.array_equal(person.amodal_mask, person_amodal > 0)
    assert np.array_equal(chair.amodal_mask, chair_amodal > 0)
    assert person.amodal_mask.dtype == bool
    assert chair.amodal_mask.dtype == bool
    assert table.amodal_mask is None


def test_complete_overlapping_objects_counts_only_newly_completed_pixels(
    monkeypatch, rgb_image
):
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    person.overlap_partner_ids.add(chair.object_id)
    chair.overlap_partner_ids.add(person.object_id)

    person_amodal = person.modal_mask.astype(bool)
    person_amodal[10, 10] = True
    chair.modal_mask = (chair.modal_mask > 0).astype(np.uint8)
    chair_amodal = chair.modal_mask.copy()
    chair_amodal[10, 10] = 1
    chair_amodal[10, 11] = 1

    completion_model = Mock()
    completion_model.complete.return_value = [person_amodal, chair_amodal]
    del monkeypatch

    completion_stage.complete_objects(
        rgb_image,
        [person, chair],
        completion_model,
        **COMPLETION_LIMITS,
    )

    assert np.array_equal(
        person.completion_hole_mask,
        person_amodal & (person.modal_mask == 0),
    )
    assert np.array_equal(
        chair.completion_hole_mask,
        (chair_amodal > 0) & (chair.modal_mask == 0),
    )
    assert person.completion_hole_area == 1
    assert chair.completion_hole_area == 2


@pytest.mark.parametrize(
    "invalid_kind",
    [
        "wrong_shape",
        "nonnumeric",
        "nonfinite",
        "empty",
        "missing_modal_pixel",
    ],
)
def test_completion_rejects_malformed_object_outputs(
    rgb_image, invalid_kind
):
    detected = _detected_object("person-1", "person", (0, 0, 4, 4))
    output = detected.modal_mask.copy()
    if invalid_kind == "wrong_shape":
        output = output[:-1]
    elif invalid_kind == "nonnumeric":
        output = output.astype(str)
    elif invalid_kind == "nonfinite":
        output = output.astype(np.float32)
        output[0, 0] = np.nan
    elif invalid_kind == "empty":
        output.fill(0)
    elif invalid_kind == "missing_modal_pixel":
        output[0, 0] = 0

    completion_model = Mock()
    completion_model.complete.return_value = [output]

    completion_stage.complete_objects(
        rgb_image,
        [detected],
        completion_model,
        **COMPLETION_LIMITS,
    )

    expected_modal = detected.modal_mask > 0
    assert np.array_equal(detected.amodal_mask, expected_modal)
    assert detected.amodal_mask.dtype == bool
    assert not np.any(detected.completion_hole_mask)
    assert detected.completion_hole_area == 0


def test_completion_rejects_excessive_area_growth(rgb_image):
    detected = _detected_object("person-1", "person", (0, 0, 4, 4))
    oversized = np.zeros_like(detected.modal_mask)
    oversized[0:8, 0:8] = 1
    completion_model = Mock()
    completion_model.complete.return_value = [oversized]

    completion_stage.complete_objects(
        rgb_image,
        [detected],
        completion_model,
        max_area_growth_ratio=3.0,
        max_bbox_growth_ratio=9.0,
    )

    assert np.array_equal(detected.amodal_mask, detected.modal_mask > 0)
    assert detected.completion_hole_area == 0


def test_completion_rejects_excessive_bbox_growth(rgb_image):
    detected = _detected_object("person-1", "person", (0, 0, 4, 4))
    scattered = detected.modal_mask.copy()
    scattered[15, 15] = 1
    completion_model = Mock()
    completion_model.complete.return_value = [scattered]

    completion_stage.complete_objects(
        rgb_image,
        [detected],
        completion_model,
        max_area_growth_ratio=4.0,
        max_bbox_growth_ratio=9.0,
    )

    assert np.array_equal(detected.amodal_mask, detected.modal_mask > 0)
    assert detected.completion_hole_area == 0


def test_completion_preserves_valid_objects_when_another_output_is_invalid(
    rgb_image,
):
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    valid_person = person.modal_mask.copy()
    valid_person[0, 4] = 1
    invalid_chair = chair.modal_mask[:-1]
    completion_model = Mock()
    completion_model.complete.return_value = [valid_person, invalid_chair]

    completion_stage.complete_objects(
        rgb_image,
        [person, chair],
        completion_model,
        **COMPLETION_LIMITS,
    )

    assert np.array_equal(person.amodal_mask, valid_person > 0)
    assert person.completion_hole_area == 1
    assert np.array_equal(chair.amodal_mask, chair.modal_mask > 0)
    assert chair.completion_hole_area == 0


def test_completion_batch_failure_falls_back_all_candidates(rgb_image):
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    completion_model = Mock()
    completion_model.complete.side_effect = RuntimeError("DIFT failed")

    completion_stage.complete_objects(
        rgb_image,
        [person, chair],
        completion_model,
        **COMPLETION_LIMITS,
    )

    for detected in (person, chair):
        assert np.array_equal(
            detected.amodal_mask, detected.modal_mask > 0
        )
        assert not np.any(detected.completion_hole_mask)
        assert detected.completion_hole_area == 0


def test_completion_output_count_mismatch_falls_back_all_candidates(rgb_image):
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    partial = person.modal_mask.copy()
    partial[0, 4] = 1
    completion_model = Mock()
    completion_model.complete.return_value = [partial]

    completion_stage.complete_objects(
        rgb_image,
        [person, chair],
        completion_model,
        **COMPLETION_LIMITS,
    )

    for detected in (person, chair):
        assert np.array_equal(
            detected.amodal_mask, detected.modal_mask > 0
        )
        assert not np.any(detected.completion_hole_mask)
        assert detected.completion_hole_area == 0


def test_completion_validation_config_is_explicit():
    from backend.config import config

    assert config.get_pipeline_config("completion") == COMPLETION_LIMITS


def test_apply_pair_decisions_records_unique_occluders_on_hidden_object():
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    table = _detected_object("table-1", "table", (1, 1, 4, 4))
    decisions = [
        PairDecision(
            "person-1", "chair-1", "person-1", "chair-1"
        ),
        PairDecision(
            "person-1", "table-1", "person-1", "table-1"
        ),
    ]

    reconstruction_stage.apply_pair_decisions(
        [person, chair, table], decisions
    )

    assert person.occluder_ids == {"chair-1", "table-1"}
    assert chair.occluder_ids == set()
    assert table.occluder_ids == set()


def test_apply_pair_decisions_ignores_ambiguous_pair():
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    ambiguous = PairDecision("person-1", "chair-1", None, None)

    reconstruction_stage.apply_pair_decisions(
        [person, chair], [ambiguous]
    )

    assert person.occluder_ids == set()
    assert chair.occluder_ids == set()


def test_build_reconstruction_masks_constrains_assigned_occluders():
    hidden = _detected_object("hidden", "person", (8, 8, 4, 4))
    first_occluder = _detected_object("first", "chair", (0, 0, 1, 1))
    second_occluder = _detected_object("second", "table", (0, 0, 1, 1))
    distant_occluder = _detected_object("distant", "lamp", (0, 0, 1, 1))

    hidden.amodal_mask = hidden.modal_mask > 0
    hidden.amodal_mask[8:12, 12:14] = True
    hidden.completion_hole_mask = hidden.amodal_mask & (
        hidden.modal_mask == 0
    )
    hidden.occluder_ids = {"first", "second", "distant"}

    first_occluder.modal_mask.fill(0)
    first_occluder.modal_mask[7, 13] = 255
    first_occluder.modal_mask[9, 10] = 255
    second_occluder.modal_mask.fill(0)
    second_occluder.modal_mask[12, 8] = 255
    distant_occluder.modal_mask.fill(0)
    distant_occluder.modal_mask[0, 0] = 255

    reconstruction_stage.build_reconstruction_masks(
        [hidden, first_occluder, second_occluder, distant_occluder],
        (3, 3),
    )

    reconstruction = hidden.reconstruction_mask
    assert reconstruction.dtype == bool
    assert np.all(reconstruction[hidden.completion_hole_mask])
    assert reconstruction[7, 13]
    assert reconstruction[12, 8]
    assert not reconstruction[9, 10]
    assert not reconstruction[0, 0]
    assert first_occluder.reconstruction_mask is None
    assert second_occluder.reconstruction_mask is None
    assert distant_occluder.reconstruction_mask is None


def test_reconstruct_objects_inpaints_only_nonempty_masks(rgb_image):
    hidden = _detected_object("hidden", "person", (0, 0, 4, 4))
    empty = _detected_object("empty", "chair", (4, 0, 4, 4))
    ordinary = _detected_object("ordinary", "table", (0, 4, 4, 4))
    hidden.reconstruction_mask = np.zeros((6, 8), dtype=bool)
    hidden.reconstruction_mask[2:4, 3:5] = True
    hidden.amodal_mask = np.zeros((6, 8), dtype=bool)
    hidden.amodal_mask[1:5, 2:6] = True
    empty.reconstruction_mask = np.zeros((6, 8), dtype=bool)
    empty.amodal_mask = empty.reconstruction_mask.copy()

    reconstructed = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
    inpaint = Mock(return_value=reconstructed)

    reconstruction_stage.reconstruct_objects(
        rgb_image,
        [hidden, empty, ordinary],
        inpaint,
        context_ratio=0.25,
    )

    inpaint.assert_called_once()
    source, mask, prompt = inpaint.call_args.args
    assert source.mode == "RGB"
    assert source.size == (6, 6)
    assert mask.mode == "L"
    assert mask.size == source.size
    assert set(np.unique(np.asarray(mask))) == {0, 255}
    assert "person" in prompt
    assert hidden.reconstruction_canvas.mode == "RGB"
    assert hidden.reconstruction_canvas.size == source.size
    assert hidden.reconstruction_roi is not None
    restored_mask = roi_stage.restore_array(
        np.asarray(mask) > 0, hidden.reconstruction_roi
    )
    assert np.array_equal(restored_mask, hidden.reconstruction_mask)
    assert empty.reconstruction_canvas is None
    assert empty.reconstruction_roi is None
    assert ordinary.reconstruction_canvas is None
    assert ordinary.reconstruction_roi is None


def test_reconstruction_context_ratio_is_configured():
    from backend.config import config

    assert config.get_pipeline_config("reconstruction") == {
        "context_ratio": 0.25
    }


def test_refine_masks_guides_matting_and_limits_alpha(monkeypatch, rgb_image):
    raw_mask = np.zeros((6, 8), dtype=np.uint8)
    raw_mask[2:4, 3:5] = 255
    predicted_alpha = torch.ones((6, 8), dtype=torch.float32)
    matting = _install_matting_fake(monkeypatch, predicted_alpha)

    result = matting_stage.refine_masks(
        np.asarray(rgb_image), [raw_mask], matting
    )

    assert len(result) == 1
    assert result[0].shape == (6, 8)
    assert result[0].dtype == np.float64
    assert np.all((0 <= result[0]) & (result[0] <= 1))

    guided_image = np.asarray(matting.call_args.args[0])
    assert np.all(guided_image[raw_mask == 0] == 0)
    assert np.array_equal(
        guided_image[raw_mask > 0], np.asarray(rgb_image)[raw_mask > 0]
    )


def test_extract_object_layers_builds_rgba_crop(monkeypatch, rgb_image):
    image_np = np.asarray(rgb_image)
    alpha = np.zeros((6, 8), dtype=np.float64)
    alpha[1:5, 2:6] = 0.75
    kernel_size = (1, 1)

    inpaint = _install_inpainting_fake(
        monkeypatch, Image.new("RGB", rgb_image.size, (10, 10, 10))
    )
    inpaint_mask = np.zeros((6, 8), dtype=bool)
    inpaint_mask[1:5, 2:6] = True
    build_mask = Mock(return_value=inpaint_mask)
    monkeypatch.setattr(layer_stage, "build_inpaint_mask", build_mask)
    monkeypatch.setattr(
        layer_stage,
        "refine_background",
        Mock(side_effect=lambda background, *_args, **_kwargs: background),
    )
    foreground = np.full_like(image_np, 200)
    refine_alpha = Mock(return_value=(alpha, foreground))
    monkeypatch.setattr(
        layer_stage, "refine_alpha_with_colors", refine_alpha
    )

    layers = layer_stage.extract_layers(
        rgb_image,
        image_np,
        [alpha],
        ["component"],
        kernel_size,
        inpaint,
    )

    assert len(layers) == 1
    layer = layers[0]
    assert (layer.x, layer.y, layer.width, layer.height) == (2, 1, 4, 4)
    assert layer.keyword == "component"
    assert layer.png_base64
    build_mask.assert_called_once()
    inpaint.assert_called_once()
    assert inpaint.call_args.args[0] is rgb_image
    assert inpaint.call_args.args[1].mode == "L"
    refine_alpha.assert_called_once()


def test_generate_final_background_unions_masks(monkeypatch, rgb_image):
    first = np.zeros((6, 8), dtype=np.uint8)
    first[1:3, 1:3] = 255
    second = np.zeros((6, 8), dtype=np.uint8)
    second[3:5, 5:7] = 255
    soft_alpha = np.zeros((6, 8), dtype=np.float64)
    soft_alpha[0, 7] = 0.5

    inpainted = Image.new("RGB", rgb_image.size, (30, 40, 50))
    inpaint = _install_inpainting_fake(monkeypatch, inpainted)
    expand = Mock(side_effect=lambda mask, _kernel: mask)
    monkeypatch.setattr(background_stage, "expand_mask", expand)
    refine = Mock(side_effect=lambda background, *_args, **_kwargs: background)
    monkeypatch.setattr(background_stage, "refine_background", refine)

    result = background_stage.generate_background_from_masks(
        rgb_image, [first, second], [soft_alpha], (1, 1), inpaint
    )

    assert result.size == rgb_image.size
    expand.assert_called_once()
    union = expand.call_args.args[0]
    expected_union = (first > 0) | (second > 0) | (soft_alpha > 0.005)
    assert np.array_equal(union, expected_union)
    inpaint.assert_called_once()
    passed_mask = np.asarray(inpaint.call_args.args[1])
    assert np.array_equal(passed_mask > 0, union)
    refine.assert_called_once()


def test_process_image_coordinates_all_pipeline_stages(monkeypatch, rgb_image):
    raw_mask = np.zeros((6, 8), dtype=np.uint8)
    raw_mask[1:5, 2:6] = 255
    alpha = raw_mask.astype(np.float64) / 255
    expected_layer = pipeline.ObjectLayer("component", "layer-data", 2, 1, 4, 4)
    expected_background = Image.new("RGB", rgb_image.size, (1, 2, 3))

    detected = pipeline.DetectedObject(
        object_id="object-0",
        semantic_class="component",
        display_label="component",
        modal_mask=raw_mask,
        bbox=(2, 1, 4, 4),
    )
    manager = Mock()
    processor = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        processor
    )
    matte = manager.get_matting_model.return_value.process
    inpaint = manager.get_inpainting_model.return_value.process
    extract_objects = Mock(return_value=[detected])
    link_overlaps = Mock(return_value=[])
    apply_decisions = Mock()
    build_reconstruction = Mock()
    events = []
    reconstruct_objects = Mock(
        side_effect=lambda *_args, **_kwargs: events.append("reconstruct")
    )

    def attach_alpha(_image_np, objects, supplied_matte):
        events.append("matte")
        assert supplied_matte is matte
        objects[0].soft_alpha = alpha

    refine_objects = Mock(side_effect=attach_alpha)
    extract_layers = Mock(return_value=[expected_layer])
    generate_background = Mock(return_value=expected_background)
    encode = Mock(side_effect=lambda image, fmt="PNG": f"encoded-{image.size}")
    monkeypatch.setattr(
        pipeline_orchestrator, "extract_objects", extract_objects
    )
    monkeypatch.setattr(
        pipeline_orchestrator, "link_overlap_partners", link_overlaps
    )
    monkeypatch.setattr(
        pipeline_orchestrator, "apply_pair_decisions", apply_decisions
    )
    monkeypatch.setattr(
        pipeline_orchestrator,
        "build_reconstruction_masks",
        build_reconstruction,
    )
    monkeypatch.setattr(
        pipeline_orchestrator,
        "reconstruct_objects",
        reconstruct_objects,
    )
    monkeypatch.setattr(
        pipeline_orchestrator, "refine_objects", refine_objects
    )
    monkeypatch.setattr(
        pipeline_orchestrator, "extract_object_layers", extract_layers
    )
    monkeypatch.setattr(
        pipeline_orchestrator,
        "generate_final_background",
        generate_background,
    )
    monkeypatch.setattr(pipeline_orchestrator, "_image_to_base64", encode)

    result = pipeline_orchestrator.process_image(
        rgb_image, ["component"], manager=manager
    )

    assert result.original_width == 8
    assert result.original_height == 6
    assert result.background_base64 == "encoded-(8, 6)"
    assert result.layers == [expected_layer]
    extract_objects.assert_called_once()
    prepared_image, prepared_keywords, supplied_processor = (
        extract_objects.call_args.args
    )
    assert prepared_image.mode == "RGB"
    assert prepared_image.size == rgb_image.size
    assert prepared_keywords == ["component"]
    assert supplied_processor is processor
    link_overlaps.assert_called_once_with([detected])
    manager.get_completion_model.assert_not_called()
    apply_decisions.assert_called_once_with([detected], [])
    build_reconstruction.assert_called_once()
    reconstruct_objects.assert_called_once_with(
        prepared_image,
        [detected],
        inpaint,
        context_ratio=0.25,
    )
    assert events[:2] == ["reconstruct", "matte"]
    refine_objects.assert_called_once()
    extract_layers.assert_called_once()
    generate_background.assert_called_once()
    assert extract_layers.call_args.args[-1] is inpaint
    assert generate_background.call_args.args[-1] is inpaint


def test_process_image_returns_original_when_nothing_detected(
    monkeypatch, rgb_image
):
    manager = Mock()
    extract_objects = Mock(return_value=[])
    monkeypatch.setattr(
        pipeline_orchestrator, "extract_objects", extract_objects
    )
    monkeypatch.setattr(
        pipeline_orchestrator,
        "_image_to_base64",
        Mock(return_value="original-image"),
    )

    result = pipeline_orchestrator.process_image(
        rgb_image, ["missing"], manager=manager
    )

    assert result.background_base64 == "original-image"
    assert result.layers == []
    manager.get_matting_model.assert_not_called()


def test_process_masks_returns_completed_and_bypass_masks_in_object_order(
    monkeypatch, rgb_image
):
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    lamp = _detected_object("lamp-1", "lamp", (6, 0, 2, 2))
    for detected in (person, chair, lamp):
        detected.modal_mask = np.zeros((6, 8), dtype=np.uint8)
    person.modal_mask[0:4, 0:4] = 255
    chair.modal_mask[2:6, 2:6] = 255
    lamp.modal_mask[0:2, 6:8] = 255
    person.overlap_partner_ids.add(chair.object_id)
    chair.overlap_partner_ids.add(person.object_id)

    person_amodal = (person.modal_mask > 0).astype(np.uint8)
    chair_amodal = (chair.modal_mask > 0).astype(np.uint8)
    person_amodal[4, 1] = 1
    chair_amodal[1, 4] = 1
    completion_model = Mock()
    completion_model.complete.return_value = [person_amodal, chair_amodal]

    monkeypatch.setattr(
        pipeline_orchestrator,
        "extract_objects",
        Mock(return_value=[person, chair, lamp]),
    )
    monkeypatch.setattr(
        pipeline_orchestrator,
        "link_overlap_partners",
        Mock(return_value=[("person-1", "chair-1")]),
    )
    manager = Mock()
    manager.get_completion_model.return_value = completion_model
    matting_getter = Mock()
    inpainting_getter = Mock()
    manager.get_matting_model = matting_getter
    manager.get_inpainting_model = inpainting_getter

    masks = pipeline_orchestrator.process_masks(
        rgb_image, ["person", "chair", "lamp"], manager=manager
    )

    assert len(masks) == 3
    assert np.array_equal(masks[0], person_amodal > 0)
    assert np.array_equal(masks[1], chair_amodal > 0)
    assert np.array_equal(masks[2], lamp.modal_mask)
    assert all(mask.shape == (6, 8) for mask in masks)
    completion_model.complete.assert_called_once()
    matting_getter.assert_not_called()
    inpainting_getter.assert_not_called()


def test_process_masks_skips_completion_model_without_overlap(
    monkeypatch, rgb_image
):
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    person.modal_mask = np.zeros((6, 8), dtype=np.uint8)
    person.modal_mask[0:4, 0:4] = 255
    monkeypatch.setattr(
        pipeline_orchestrator,
        "extract_objects",
        Mock(return_value=[person]),
    )
    monkeypatch.setattr(
        pipeline_orchestrator, "link_overlap_partners", Mock(return_value=[])
    )
    completion_getter = Mock()
    manager = Mock()
    manager.get_completion_model = completion_getter

    masks = pipeline_orchestrator.process_masks(
        rgb_image, ["person"], manager=manager
    )

    assert masks == [person.modal_mask]
    completion_getter.assert_not_called()
