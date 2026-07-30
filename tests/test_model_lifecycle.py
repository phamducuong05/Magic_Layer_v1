"""Tests for stage-scoped GPU model residency."""

from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pytest
from PIL import Image


def test_server_startup_does_not_warm_every_model():
    main_path = Path(__file__).resolve().parents[1] / "backend" / "main.py"
    source = main_path.read_text(encoding="utf-8")

    assert "model_manager.warmup_all()" not in source
    assert "model_manager.warmup_first_stage()" in source


def test_process_image_releases_models_at_stage_boundaries(monkeypatch):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    image = Image.new("RGB", (4, 4), "white")
    mask = np.ones((4, 4), dtype=np.uint8) * 255
    detected = DetectedObject(
        "object-0", "person", "person", mask, (0, 0, 4, 4)
    )
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    manager.has_object_reconstruction_model.return_value = False
    manager.get_background_inpainting_model.return_value.process = Mock()

    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[detected]),
    )
    monkeypatch.setattr(
        orchestrator, "link_overlap_partners", Mock(return_value=[])
    )
    monkeypatch.setattr(orchestrator, "refine_objects", Mock())
    monkeypatch.setattr(
        orchestrator, "extract_object_layers", Mock(return_value=[])
    )
    monkeypatch.setattr(
        orchestrator,
        "generate_final_background",
        Mock(return_value=image),
    )

    orchestrator.process_image(image, ["person"], manager=manager)

    assert manager.release_model.call_args_list == [
        call("segmentation"),
        call("matting"),
        call("background_inpainting"),
    ]


def test_process_masks_releases_segmentation_and_completion(monkeypatch):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    image = Image.new("RGB", (4, 4), "white")
    mask = np.ones((4, 4), dtype=np.uint8) * 255
    first = DetectedObject("first", "person", "person", mask, (0, 0, 4, 4))
    second = DetectedObject("second", "chair", "chair", mask, (0, 0, 4, 4))
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    manager.get_completion_model.return_value.complete.return_value = [
        mask,
        mask,
    ]
    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[first, second]),
    )

    orchestrator.process_masks(
        image, ["person", "chair"], manager=manager
    )

    assert manager.release_model.call_args_list == [
        call("segmentation"),
        call("completion"),
    ]


@pytest.mark.parametrize(
    ("failed_stage", "expected_releases"),
    [
        ("matting", [call("segmentation"), call("matting")]),
        (
            "background_inpainting",
            [
                call("segmentation"),
                call("matting"),
                call("background_inpainting"),
            ],
        ),
    ],
)
def test_stage_load_failure_still_releases_cuda_stage(
    monkeypatch, failed_stage, expected_releases
):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    image = Image.new("RGB", (4, 4), "white")
    mask = np.ones((4, 4), dtype=np.uint8) * 255
    detected = DetectedObject(
        "object-0", "person", "person", mask, (0, 0, 4, 4)
    )
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    if failed_stage == "matting":
        manager.get_matting_model.side_effect = RuntimeError("load failed")
    else:
        manager.get_background_inpainting_model.side_effect = RuntimeError(
            "load failed"
        )
    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[detected]),
    )
    monkeypatch.setattr(
        orchestrator, "link_overlap_partners", Mock(return_value=[])
    )
    monkeypatch.setattr(orchestrator, "refine_objects", Mock())

    with pytest.raises(RuntimeError, match="load failed"):
        orchestrator.process_image(image, ["person"], manager=manager)

    assert manager.release_model.call_args_list == expected_releases


def test_hd_painter_load_failure_releases_object_reconstruction_stage(
    monkeypatch,
):
    from backend.pipeline import orchestrator
    from backend.pipeline.types import DetectedObject

    image = Image.new("RGB", (4, 4), "white")
    mask = np.ones((4, 4), dtype=np.uint8) * 255
    first = DetectedObject("first", "person", "person", mask, (0, 0, 4, 4))
    second = DetectedObject("second", "chair", "chair", mask, (0, 0, 4, 4))

    def link_partners(objects, pairs):
        assert tuple(pairs) == (("first", "second"),)
        objects[0].overlap_partner_ids.add(objects[1].object_id)
        objects[1].overlap_partner_ids.add(objects[0].object_id)
        return list(pairs)

    def prepare(objects, *_args, **_kwargs):
        objects[0].reconstruction_mask = np.ones((4, 4), dtype=bool)
        return []

    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    manager.has_object_reconstruction_model.return_value = True
    manager.get_object_reconstruction_model.side_effect = RuntimeError(
        "load failed"
    )
    monkeypatch.setattr(
        orchestrator,
        "extract_raw_objects",
        Mock(return_value=[first, second]),
    )
    monkeypatch.setattr(
        orchestrator, "link_overlap_partners", link_partners
    )
    monkeypatch.setattr(orchestrator, "_complete_candidates", Mock())
    monkeypatch.setattr(
        orchestrator,
        "filter_pairs_by_amodal_overlap",
        Mock(return_value=[("first", "second")]),
    )
    monkeypatch.setattr(
        orchestrator, "prepare_raw_reconstruction_masks", prepare
    )

    with pytest.raises(RuntimeError, match="load failed"):
        orchestrator.process_image(image, ["person"], manager=manager)

    assert manager.release_model.call_args_list == [
        call("segmentation"),
        call("completion"),
        call("object_reconstruction"),
    ]
