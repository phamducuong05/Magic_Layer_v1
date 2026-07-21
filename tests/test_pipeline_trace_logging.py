"""Tests for terminal-visible pipeline stage and decision tracing."""

import logging
from unittest.mock import Mock

import numpy as np
from PIL import Image


def test_simple_request_logs_every_pipeline_stage(monkeypatch, caplog):
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

    with caplog.at_level(logging.INFO, logger=orchestrator.__name__):
        orchestrator.process_image(image, ["person"], manager=manager)

    expected_markers = [
        "[PIPELINE] START",
        "[SEGMENTATION] START",
        "[SEGMENTATION] COMPLETE",
        "[OVERLAP_DETECTION] START",
        "[OVERLAP_DETECTION] COMPLETE",
        "[COMPLETION] SKIP",
        "[AMODAL_OVERLAP_VALIDATION] SKIP",
        "[DEPTH_ORDERING] SKIP",
        "[OBJECT_RECONSTRUCTION] SKIP",
        "[GROUPING] START",
        "[GROUPING] COMPLETE",
        "[MATTING] START",
        "[MATTING] COMPLETE",
        "[LAYER_EXTRACTION] START",
        "[LAYER_EXTRACTION] COMPLETE",
        "[BACKGROUND_INPAINTING] START",
        "[BACKGROUND_INPAINTING] COMPLETE",
        "[PIPELINE] COMPLETE",
    ]
    for marker in expected_markers:
        assert marker in caplog.text
