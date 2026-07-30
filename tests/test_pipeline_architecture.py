"""Tests for explicit-dependency image pipeline modules."""

import inspect
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image


def test_pipeline_types_preserve_contract_and_add_per_object_alpha():
    from backend.pipeline.types import DetectedObject, ObjectLayer, ProcessResult

    mask = np.ones((3, 4), dtype=np.uint8)
    detected = DetectedObject(
        object_id="object-0",
        semantic_class="person",
        display_label="person",
        modal_mask=mask,
        bbox=(0, 0, 4, 3),
    )

    assert detected.soft_alpha is None
    assert detected.original_modal_bbox == (0, 0, 4, 3)
    assert ObjectLayer("person", "png", 0, 0, 4, 3).keyword == "person"
    assert ProcessResult("background", 4, 3).layers == []


def test_segmentation_uses_the_supplied_processor():
    from backend.pipeline.segmentation import extract_raw_objects

    processor = Mock()
    processor.set_image.return_value = {}
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1:, 1:3] = 1
    processor.set_text_prompt.return_value = {"masks": [mask]}

    objects = extract_raw_objects(
        Image.new("RGB", (4, 3)), [" person "], processor
    )

    assert len(objects) == 1
    assert objects[0].semantic_class == "person"
    assert objects[0].bbox == (1, 1, 2, 2)
    processor.set_image.assert_called_once()
    processor.set_text_prompt.assert_called_once()


def test_segmentation_does_not_group_during_raw_extraction():
    from backend.pipeline import segmentation

    source = inspect.getsource(segmentation.extract_raw_objects)

    assert "_merge_overlapping_masks" not in source


def test_completion_candidates_and_model_are_explicit_and_ordered():
    from backend.pipeline.completion import (
        complete_objects,
        get_completion_candidates,
    )
    from backend.pipeline.types import DetectedObject

    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    first = DetectedObject("first", "person", "person", mask, (1, 1, 2, 2))
    second = DetectedObject("second", "chair", "chair", mask, (1, 1, 2, 2))
    third = DetectedObject("third", "lamp", "lamp", mask, (1, 1, 2, 2))
    first.overlap_partner_ids.add("second")
    second.overlap_partner_ids.add("first")

    candidates = get_completion_candidates([first, second, third])
    amodal_first = mask.copy()
    amodal_first[0, 1] = 1
    amodal_second = mask.copy()
    model = Mock()
    model.complete.return_value = [amodal_first, amodal_second]
    image = Image.new("RGB", (5, 4))

    complete_objects(
        image,
        candidates,
        model,
        max_area_growth_ratio=4.0,
        max_bbox_growth_ratio=9.0,
    )

    assert candidates == [first, second]
    model.complete.assert_called_once_with(
        image, [first.modal_mask, second.modal_mask], [first.bbox, second.bbox]
    )
    assert first.completion_hole_area == 1
    assert second.completion_hole_area == 0


def test_completion_linking_uses_explicit_external_pairs_only():
    from backend.pipeline.completion import (
        get_completion_candidates,
        link_overlap_partners,
    )
    from backend.pipeline.types import DetectedObject

    mask = np.ones((4, 5), dtype=np.uint8)
    first = DetectedObject(
        "a", "class-a", "class-a", mask.copy(), (0, 0, 5, 4)
    )
    second = DetectedObject(
        "b", "class-b", "class-b", mask.copy(), (0, 0, 5, 4)
    )
    third = DetectedObject(
        "c", "class-c", "class-c", mask.copy(), (0, 0, 5, 4)
    )

    pairs = link_overlap_partners(
        [first, second, third],
        pairs=[("b", "c")],
    )

    assert pairs == [("b", "c")]
    assert first.overlap_partner_ids == set()
    assert second.overlap_partner_ids == {"c"}
    assert third.overlap_partner_ids == {"b"}
    assert [
        item.object_id
        for item in get_completion_candidates([first, second, third])
    ] == ["b", "c"]


def test_detected_object_keeps_raw_and_effective_hole_areas_separate():
    from backend.pipeline.types import DetectedObject

    mask = np.ones((3, 4), dtype=np.uint8)
    detected = DetectedObject(
        "object-0", "person", "person", mask, (0, 0, 4, 3)
    )
    detected.completion_hole_area = 7
    detected.effective_completion_hole_area = 0

    assert detected.completion_hole_area == 7
    assert detected.effective_completion_hole_area == 0


def test_extracted_geometry_stages_do_not_reference_global_model_manager():
    from backend.pipeline import completion, reconstruction, segmentation

    for module in (segmentation, completion, reconstruction):
        assert "model_manager" not in inspect.getsource(module)


def test_matting_attaches_alpha_using_the_supplied_callable():
    from backend.pipeline.grouping import group_reconstructed_objects
    from backend.pipeline.matting import refine_objects
    from backend.pipeline.types import DetectedObject

    mask = np.ones((4, 4), dtype=np.uint8)
    detected = DetectedObject(
        "object-0", "person", "person", mask, (0, 0, 4, 4)
    )
    group = group_reconstructed_objects([detected])[0]
    matte = Mock(return_value=torch.full((4, 4), 0.75))

    refine_objects(
        Image.new("RGB", (4, 4), (127, 127, 127)),
        [group],
        matte,
        context_ratio=0.0,
        support_dilation_pixels=0,
    )

    matte.assert_called_once()
    assert np.allclose(group.soft_alpha, 0.75)


def test_reconstruction_and_background_use_distinct_callable_contracts():
    from backend.pipeline import background, layers, matting, reconstruction

    assert "background_inpaint" in inspect.signature(
        layers.extract_object_layers
    ).parameters
    assert "background_inpaint" in inspect.signature(
        background.generate_final_background
    ).parameters
    assert "reconstruct" in inspect.signature(
        reconstruction.reconstruct_objects
    ).parameters
    for module in (matting, layers, background):
        assert "model_manager" not in inspect.getsource(module)


def test_process_masks_does_not_resolve_completion_without_candidates(
    monkeypatch,
):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    mask = np.ones((3, 4), dtype=np.uint8)
    detected = DetectedObject(
        "object-0", "person", "person", mask, (0, 0, 4, 3)
    )
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    monkeypatch.setattr(
        orchestrator, "extract_raw_objects", Mock(return_value=[detected])
    )
    monkeypatch.setattr(
        orchestrator, "link_overlap_partners", Mock(return_value=[])
    )

    masks = orchestrator.process_masks(
        Image.new("RGB", (4, 3)), ["person"], manager=manager
    )

    assert masks == [detected.modal_mask]
    manager.get_completion_model.assert_not_called()


def test_process_masks_skips_internal_containment_completion(monkeypatch):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    large_mask = np.zeros((20, 20), dtype=np.uint8)
    large_mask[0:12, 0:12] = 255
    small_mask = np.zeros((20, 20), dtype=np.uint8)
    small_mask[3:7, 3:7] = 255
    large = DetectedObject(
        "large", "table", "table", large_mask, (0, 0, 12, 12)
    )
    small = DetectedObject(
        "small", "product", "product", small_mask, (3, 3, 4, 4)
    )
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[large, small]),
    )
    complete = Mock()
    monkeypatch.setattr(orchestrator, "_complete_candidates", complete)

    masks = orchestrator.process_masks(
        Image.new("RGB", (20, 20)),
        ["table", "product"],
        manager,
    )

    assert len(masks) == 2
    complete.assert_not_called()


def test_merge_intent_does_not_expand_external_completion_target(monkeypatch):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    def detected(
        object_id: str,
        semantic_class: str,
        bbox: tuple[int, int, int, int],
    ) -> DetectedObject:
        mask = np.zeros((40, 40), dtype=np.uint8)
        x, y, width, height = bbox
        mask[y : y + height, x : x + width] = 255
        return DetectedObject(
            object_id,
            semantic_class,
            semantic_class,
            mask,
            bbox,
        )

    # A contains exactly 70% of B. C overlaps only B's exposed strip.
    a = detected("a", "table", (0, 0, 20, 20))
    b = detected("b", "product", (13, 5, 10, 10))
    c = detected("c", "hand", (21, 5, 6, 10))
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[a, b, c]),
    )
    captured_completion_ids: list[str] = []

    def capture_completion(_image, _objects, _manager, candidates):
        captured_completion_ids.extend(
            item.object_id for item in candidates
        )

    monkeypatch.setattr(
        orchestrator,
        "_complete_candidates",
        Mock(side_effect=capture_completion),
    )

    masks = orchestrator.process_masks(
        Image.new("RGB", (40, 40)),
        ["table", "product", "hand"],
        manager=manager,
    )

    assert len(masks) == 3
    assert captured_completion_ids == ["b", "c"]
    assert "a" not in captured_completion_ids


def test_process_image_completes_overlap_without_reconstruction_model(
    monkeypatch,
):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    mask = np.ones((3, 4), dtype=np.uint8)
    detected = DetectedObject(
        "object-0", "person", "person", mask, (0, 0, 4, 3)
    )
    occluder = DetectedObject(
        "object-1", "chair", "chair", mask, (0, 0, 4, 3)
    )
    detected.soft_alpha = mask.astype(np.float64)
    occluder.soft_alpha = mask.astype(np.float64)
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    manager.has_object_reconstruction_model.return_value = False
    completion_model = manager.get_completion_model.return_value
    completion_model.complete.return_value = [mask, mask]
    background_inpaint = Mock()
    manager.get_background_inpainting_model.return_value.process = (
        background_inpaint
    )
    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[detected, occluder]),
    )
    filter_pairs = Mock(return_value=[])
    monkeypatch.setattr(
        orchestrator, "filter_pairs_by_amodal_overlap", filter_pairs
    )
    prepare_reconstruction = Mock()
    monkeypatch.setattr(
        orchestrator,
        "prepare_raw_reconstruction_masks",
        prepare_reconstruction,
    )
    monkeypatch.setattr(orchestrator, "refine_objects", Mock())
    extract_layers = Mock(return_value=[])
    monkeypatch.setattr(orchestrator, "extract_object_layers", extract_layers)
    expected_background = Image.new("RGB", (4, 3), "white")
    generate_background = Mock(return_value=expected_background)
    monkeypatch.setattr(
        orchestrator, "generate_final_background", generate_background
    )

    result = orchestrator.process_image(
        Image.new("RGB", (4, 3)), ["person"], manager=manager
    )

    assert result.original_width == 4
    assert result.original_height == 3
    completion_model.complete.assert_called_once()
    filter_pairs.assert_called_once_with(
        [detected, occluder], [("object-0", "object-1")]
    )
    prepare_reconstruction.assert_not_called()
    manager.get_object_reconstruction_model.assert_not_called()
    manager.get_background_inpainting_model.assert_called_once()
    assert (
        extract_layers.call_args.args[-1].__wrapped__ is background_inpaint
    )
    assert (
        generate_background.call_args.args[-1].__wrapped__
        is background_inpaint
    )
