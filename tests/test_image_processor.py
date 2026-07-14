"""Unit tests for the image-to-components orchestration pipeline.

These tests use small deterministic fakes. They never load SAM3, BiRefNet, LaMa,
or model weights, so failures point to pipeline wiring rather than model quality.
"""

import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="the backend requires PyTorch")
pytest.importorskip("cv2", reason="the image pipeline requires OpenCV")

# Importing the real registry would import every heavyweight model adapter and
# its optional dependencies. image_processor only needs the manager interface,
# which each test configures with deterministic fakes.
models_module = types.ModuleType("backend.models")
models_module.model_manager = Mock()
sys.modules["backend.models"] = models_module

from backend import image_processor as pipeline
from backend.core.helpers import _prepare_inpaint_masks, _preserve_unmasked_pixels
from backend.core.occlusion import PairDecision


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


def _install_segmentation_fake(monkeypatch, processor):
    segmentation_model = Mock()
    segmentation_model.get_processor.return_value = processor
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_segmentation_model",
        Mock(return_value=segmentation_model),
    )


def _install_matting_fake(monkeypatch, alpha):
    matting_model = Mock()
    matting_model.process.return_value = alpha
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_matting_model",
        Mock(return_value=matting_model),
    )
    return matting_model.process


def _install_inpainting_fake(monkeypatch, output):
    inpainting_model = Mock()
    inpainting_model.process.return_value = output
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_inpainting_model",
        Mock(return_value=inpainting_model),
    )
    return inpainting_model.process


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
    _install_segmentation_fake(monkeypatch, processor)

    objects = pipeline._extract_objects(rgb_image, [" component ", "", "missing"])

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
    _install_segmentation_fake(monkeypatch, processor)

    objects = pipeline._extract_objects(rgb_image, ["button"])

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
    _install_segmentation_fake(monkeypatch, processor)

    objects = pipeline._extract_objects(rgb_image, ["button"])

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

    pairs = pipeline._link_overlap_partners([person, chair])

    assert pairs == [("person-1", "chair-1")]
    assert person.overlap_partner_ids == {"chair-1"}
    assert chair.overlap_partner_ids == {"person-1"}


def test_link_overlap_partners_ignores_same_class_and_separate_objects():
    first_person = _detected_object("person-1", "person", (0, 0, 10, 10))
    second_person = _detected_object("person-2", "person", (5, 5, 10, 10))
    chair = _detected_object("chair-1", "chair", (15, 15, 2, 2))

    pairs = pipeline._link_overlap_partners(
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

    pairs = pipeline._link_overlap_partners([person, chair, table])

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
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    get_completion_model = Mock()
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_completion_model",
        get_completion_model,
    )

    pipeline._complete_overlapping_objects(rgb_image, [person])

    get_completion_model.assert_not_called()
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

    person_amodal = np.ones_like(person.modal_mask)
    chair_amodal = np.full_like(chair.modal_mask, 2)
    completion_model = Mock()
    completion_model.complete.return_value = [person_amodal, chair_amodal]
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_completion_model",
        Mock(return_value=completion_model),
    )

    pipeline._complete_overlapping_objects(
        rgb_image, [person, chair, table]
    )

    completion_model.complete.assert_called_once()
    completed_image, modal_masks, bboxes = completion_model.complete.call_args.args
    assert completed_image is rgb_image
    assert modal_masks == [person.modal_mask, chair.modal_mask]
    assert bboxes == [person.bbox, chair.bbox]
    assert person.amodal_mask is person_amodal
    assert chair.amodal_mask is chair_amodal
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
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_completion_model",
        Mock(return_value=completion_model),
    )

    pipeline._complete_overlapping_objects(rgb_image, [person, chair])

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

    pipeline._apply_pair_decisions([person, chair, table], decisions)

    assert person.occluder_ids == {"chair-1", "table-1"}
    assert chair.occluder_ids == set()
    assert table.occluder_ids == set()


def test_apply_pair_decisions_ignores_ambiguous_pair():
    person = _detected_object("person-1", "person", (0, 0, 4, 4))
    chair = _detected_object("chair-1", "chair", (2, 2, 4, 4))
    ambiguous = PairDecision("person-1", "chair-1", None, None)

    pipeline._apply_pair_decisions([person, chair], [ambiguous])

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

    pipeline._build_reconstruction_masks(
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


def test_refine_masks_guides_matting_and_limits_alpha(monkeypatch, rgb_image):
    raw_mask = np.zeros((6, 8), dtype=np.uint8)
    raw_mask[2:4, 3:5] = 255
    predicted_alpha = torch.ones((6, 8), dtype=torch.float32)
    matting = _install_matting_fake(monkeypatch, predicted_alpha)

    result = pipeline._refine_masks(np.asarray(rgb_image), [raw_mask])

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
    monkeypatch.setattr(pipeline, "build_inpaint_mask", build_mask)
    monkeypatch.setattr(
        pipeline,
        "refine_background",
        Mock(side_effect=lambda background, *_args, **_kwargs: background),
    )
    foreground = np.full_like(image_np, 200)
    refine_alpha = Mock(return_value=(alpha, foreground))
    monkeypatch.setattr(pipeline, "refine_alpha_with_colors", refine_alpha)

    layers = pipeline._extract_object_layers(
        rgb_image, image_np, [alpha], ["component"], kernel_size
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
    monkeypatch.setattr(pipeline, "expand_mask", expand)
    refine = Mock(side_effect=lambda background, *_args, **_kwargs: background)
    monkeypatch.setattr(pipeline, "refine_background", refine)

    result = pipeline._generate_final_background(
        rgb_image, [first, second], [soft_alpha], (1, 1)
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
    extract_objects = Mock(return_value=[detected])
    link_overlaps = Mock(return_value=[])
    complete_overlaps = Mock()
    refine_masks = Mock(return_value=[alpha])
    extract_layers = Mock(return_value=[expected_layer])
    generate_background = Mock(return_value=expected_background)
    encode = Mock(side_effect=lambda image, fmt="PNG": f"encoded-{image.size}")
    monkeypatch.setattr(pipeline, "_extract_objects", extract_objects)
    monkeypatch.setattr(pipeline, "_link_overlap_partners", link_overlaps)
    monkeypatch.setattr(
        pipeline, "_complete_overlapping_objects", complete_overlaps
    )
    monkeypatch.setattr(pipeline, "_refine_masks", refine_masks)
    monkeypatch.setattr(pipeline, "_extract_object_layers", extract_layers)
    monkeypatch.setattr(
        pipeline, "_generate_final_background", generate_background
    )
    monkeypatch.setattr(pipeline, "_image_to_base64", encode)

    result = pipeline.process_image(rgb_image, ["component"])

    assert result.original_width == 8
    assert result.original_height == 6
    assert result.background_base64 == "encoded-(8, 6)"
    assert result.layers == [expected_layer]
    extract_objects.assert_called_once()
    prepared_image, prepared_keywords = extract_objects.call_args.args
    assert prepared_image.mode == "RGB"
    assert prepared_image.size == rgb_image.size
    assert prepared_keywords == ["component"]
    link_overlaps.assert_called_once_with([detected])
    complete_overlaps.assert_called_once_with(prepared_image, [detected])
    refine_masks.assert_called_once()
    extract_layers.assert_called_once()
    generate_background.assert_called_once()


def test_process_image_returns_original_when_nothing_detected(
    monkeypatch, rgb_image
):
    monkeypatch.setattr(pipeline, "_extract_objects", Mock(return_value=[]))
    refine_masks = Mock()
    monkeypatch.setattr(pipeline, "_refine_masks", refine_masks)
    monkeypatch.setattr(
        pipeline, "_image_to_base64", Mock(return_value="original-image")
    )

    result = pipeline.process_image(rgb_image, ["missing"])

    assert result.background_base64 == "original-image"
    assert result.layers == []
    refine_masks.assert_not_called()


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
        pipeline, "_extract_objects", Mock(return_value=[person, chair, lamp])
    )
    monkeypatch.setattr(
        pipeline,
        "_link_overlap_partners",
        Mock(return_value=[("person-1", "chair-1")]),
    )
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_completion_model",
        Mock(return_value=completion_model),
    )
    matting_getter = Mock()
    inpainting_getter = Mock()
    monkeypatch.setattr(
        pipeline.model_manager, "get_matting_model", matting_getter
    )
    monkeypatch.setattr(
        pipeline.model_manager, "get_inpainting_model", inpainting_getter
    )

    masks = pipeline.process_masks(rgb_image, ["person", "chair", "lamp"])

    assert masks == [person_amodal, chair_amodal, lamp.modal_mask]
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
        pipeline, "_extract_objects", Mock(return_value=[person])
    )
    monkeypatch.setattr(
        pipeline, "_link_overlap_partners", Mock(return_value=[])
    )
    completion_getter = Mock()
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_completion_model",
        completion_getter,
    )

    masks = pipeline.process_masks(rgb_image, ["person"])

    assert masks == [person.modal_mask]
    completion_getter.assert_not_called()
