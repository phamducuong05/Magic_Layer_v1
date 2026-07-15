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
    assert ObjectLayer("person", "png", 0, 0, 4, 3).keyword == "person"
    assert ProcessResult("background", 4, 3).layers == []


def test_segmentation_uses_the_supplied_processor():
    from backend.pipeline.segmentation import extract_objects

    processor = Mock()
    processor.set_image.return_value = {}
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1:, 1:3] = 1
    processor.set_text_prompt.return_value = {"masks": [mask]}

    objects = extract_objects(
        Image.new("RGB", (4, 3)), [" person "], processor
    )

    assert len(objects) == 1
    assert objects[0].semantic_class == "person"
    assert objects[0].bbox == (1, 1, 2, 2)
    processor.set_image.assert_called_once()
    processor.set_text_prompt.assert_called_once()


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


def test_extracted_geometry_stages_do_not_reference_global_model_manager():
    from backend.pipeline import completion, reconstruction, segmentation

    for module in (segmentation, completion, reconstruction):
        assert "model_manager" not in inspect.getsource(module)


def test_matting_attaches_alpha_using_the_supplied_callable():
    from backend.pipeline.matting import refine_objects
    from backend.pipeline.types import DetectedObject

    mask = np.ones((3, 4), dtype=np.uint8)
    detected = DetectedObject(
        "object-0", "person", "person", mask, (0, 0, 4, 3)
    )
    matte = Mock(return_value=torch.full((3, 4), 0.75))

    refine_objects(
        np.full((3, 4, 3), 127, dtype=np.uint8), [detected], matte
    )

    matte.assert_called_once()
    assert np.allclose(detected.soft_alpha, 0.75)


def test_rendering_stages_require_explicit_inpainting_callable():
    from backend.pipeline import background, layers, matting

    assert "inpaint" in inspect.signature(
        layers.extract_object_layers
    ).parameters
    assert "inpaint" in inspect.signature(
        background.generate_final_background
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
        orchestrator, "extract_objects", Mock(return_value=[detected])
    )
    monkeypatch.setattr(
        orchestrator, "link_overlap_partners", Mock(return_value=[])
    )

    masks = orchestrator.process_masks(
        Image.new("RGB", (4, 3)), ["person"], manager=manager
    )

    assert masks == [detected.modal_mask]
    manager.get_completion_model.assert_not_called()


def test_process_image_reuses_one_explicit_inpainting_callable(monkeypatch):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    mask = np.ones((3, 4), dtype=np.uint8)
    detected = DetectedObject(
        "object-0", "person", "person", mask, (0, 0, 4, 3)
    )
    detected.soft_alpha = mask.astype(np.float64)
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    inpaint = Mock()
    manager.get_inpainting_model.return_value.process = inpaint
    monkeypatch.setattr(
        orchestrator, "extract_objects", Mock(return_value=[detected])
    )
    monkeypatch.setattr(
        orchestrator, "link_overlap_partners", Mock(return_value=[])
    )
    monkeypatch.setattr(orchestrator, "apply_pair_decisions", Mock())
    monkeypatch.setattr(orchestrator, "build_reconstruction_masks", Mock())
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
    manager.get_completion_model.assert_not_called()
    manager.get_inpainting_model.assert_called_once()
    assert extract_layers.call_args.args[-1] is inpaint
    assert generate_background.call_args.args[-1] is inpaint
