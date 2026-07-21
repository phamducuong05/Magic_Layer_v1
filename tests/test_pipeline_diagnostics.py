from unittest.mock import Mock

import numpy as np
from PIL import Image

from backend.core.occlusion import PairDecision
from backend.pipeline.grouping import group_reconstructed_objects
from backend.pipeline.types import DetectedObject


def _object(object_id: str, semantic_class: str, offset: int):
    modal = np.zeros((8, 8), dtype=np.uint8)
    modal[1:3, offset : offset + 2] = 255
    detected = DetectedObject(
        object_id=object_id,
        semantic_class=semantic_class,
        display_label=semantic_class,
        modal_mask=modal,
        bbox=(offset, 1, 2, 2),
        segmentation_index=offset,
    )
    detected.amodal_mask = modal > 0
    detected.amodal_mask[3, offset] = True
    detected.completion_hole_mask = detected.amodal_mask & ~(modal > 0)
    detected.completion_hole_area = 1
    detected.effective_completion_hole_area = 1
    return detected


def test_build_pipeline_diagnostics_records_metadata_without_pixel_data():
    from backend.pipeline.diagnostics import build_pipeline_diagnostics

    first = _object("object-0", "person", 1)
    second = _object("object-1", "chair", 3)
    first.completion_failure_stage = "validation"
    first.completion_failure_reason = "shape mismatch"
    second.reconstruction_failure_stage = "inference"
    second.reconstruction_failure_reason = "out of memory"
    group = group_reconstructed_objects([first])[0]
    group.reconstruction_conflicts = (("object-0", "object-2"),)
    decision = PairDecision(
        "object-0", "object-1", "object-0", "object-1"
    )

    diagnostics = build_pipeline_diagnostics(
        raw_objects=[first, second],
        final_groups=[group],
        potential_pairs=[("object-0", "object-1")],
        retained_pairs=[("object-0", "object-1")],
        completion_candidate_count=2,
        pair_decisions=[decision],
        stage_timings_ms={
            "completion": 1.0,
            "object_reconstruction": 2.0,
            "group_composition": 3.0,
            "matting": 4.0,
            "background_inpainting": 5.0,
        },
        peak_gpu_memory_bytes=1234,
    )

    assert diagnostics.raw_object_count == 2
    assert diagnostics.final_group_count == 1
    assert diagnostics.potential_overlap_pair_count == 1
    assert diagnostics.retained_amodal_pair_count == 1
    assert diagnostics.completion_candidate_count == 2
    assert diagnostics.objects[0].modal_area == 4
    assert diagnostics.objects[0].amodal_area == 5
    assert diagnostics.objects[0].raw_completion_hole_area == 1
    assert diagnostics.objects[0].effective_completion_hole_area == 1
    assert diagnostics.pair_decisions[0].occluded_id == "object-0"
    assert diagnostics.pair_decisions[0].ambiguous_reason is None
    assert diagnostics.groups[0].member_ids == ("object-0",)
    assert diagnostics.groups[0].reconstruction_conflicts == (
        ("object-0", "object-2"),
    )
    assert {fallback.stage for fallback in diagnostics.fallbacks} == {
        "completion.validation",
        "reconstruction.inference",
    }
    assert diagnostics.peak_gpu_memory_bytes == 1234
    assert "modal_mask" not in repr(diagnostics)
    assert "soft_alpha" not in repr(diagnostics)


def test_ambiguous_pair_diagnostic_records_reason():
    from backend.pipeline.diagnostics import build_pipeline_diagnostics

    first = _object("object-0", "person", 1)
    second = _object("object-1", "chair", 3)
    diagnostics = build_pipeline_diagnostics(
        raw_objects=[first, second],
        final_groups=[],
        potential_pairs=[("object-0", "object-1")],
        retained_pairs=[("object-0", "object-1")],
        completion_candidate_count=2,
        pair_decisions=[
            PairDecision("object-0", "object-1", None, None)
        ],
        stage_timings_ms={},
        peak_gpu_memory_bytes=None,
    )

    assert diagnostics.pair_decisions[0].ambiguous_reason == (
        "completion-hole areas are within tie tolerance"
    )


def test_completion_fallback_records_stage_and_reason():
    from backend.pipeline.completion import complete_objects

    detected = _object("object-0", "person", 1)
    invalid_shape = np.ones((2, 2), dtype=np.uint8)
    model = Mock()
    model.complete.return_value = [invalid_shape]

    complete_objects(
        Image.new("RGB", (8, 8)),
        [detected],
        model,
        max_area_growth_ratio=4.0,
        max_bbox_growth_ratio=9.0,
    )

    assert detected.completion_failure_stage == "validation"
    assert "shape mismatch" in detected.completion_failure_reason


def test_process_image_returns_step34_diagnostics(monkeypatch):
    from backend.pipeline import orchestrator

    detected = _object("object-0", "person", 1)
    detected.soft_alpha = (detected.modal_mask > 0).astype(np.float64)
    manager = Mock()
    manager.get_segmentation_model.return_value.get_processor.return_value = (
        Mock()
    )
    background_process = Mock(
        side_effect=lambda image, _mask: Image.new("RGB", image.size)
    )
    manager.get_background_inpainting_model.return_value.process = (
        background_process
    )

    monkeypatch.setattr(
        orchestrator, "extract_raw_objects", Mock(return_value=[detected])
    )
    monkeypatch.setattr(
        orchestrator, "link_overlap_partners", Mock(return_value=[])
    )
    monkeypatch.setattr(orchestrator, "refine_objects", Mock())

    def extract_layers(groups, _kernel, inpaint):
        inpaint(Image.new("RGB", (2, 2)), Image.new("L", (2, 2)))
        return []

    def final_background(image, _groups, _kernel, inpaint):
        return inpaint(image, Image.new("L", image.size))

    monkeypatch.setattr(orchestrator, "extract_object_layers", extract_layers)
    monkeypatch.setattr(
        orchestrator, "generate_final_background", final_background
    )
    monkeypatch.setattr(
        orchestrator.config,
        "get_pipeline_config",
        Mock(
            side_effect=lambda stage: {
                "matting": {
                    "context_ratio": 0.25,
                    "support_dilation_pixels": 2,
                },
                "diagnostics": {
                    "enabled": True,
                    "log_summary": True,
                    "track_peak_gpu_memory": False,
                },
            }[stage]
        ),
    )

    result = orchestrator.process_image(
        Image.new("RGB", (8, 8)), ["person"], manager=manager
    )

    assert result.diagnostics is not None
    assert result.diagnostics.raw_object_count == 1
    assert result.diagnostics.final_group_count == 1
    assert set(result.diagnostics.stage_timings_ms) == {
        "completion",
        "object_reconstruction",
        "group_composition",
        "matting",
        "background_inpainting",
    }
    assert all(
        elapsed >= 0
        for elapsed in result.diagnostics.stage_timings_ms.values()
    )
    assert background_process.call_count == 2


def test_gpu_peak_tracking_is_optional(monkeypatch):
    from backend.pipeline import diagnostics as diagnostics_stage

    cuda = Mock()
    cuda.is_available.return_value = True
    cuda.max_memory_allocated.return_value = 4096
    monkeypatch.setattr(diagnostics_stage.torch, "cuda", cuda)

    assert diagnostics_stage.start_gpu_memory_tracking(False) is False
    cuda.reset_peak_memory_stats.assert_not_called()
    assert diagnostics_stage.read_peak_gpu_memory(False) is None

    assert diagnostics_stage.start_gpu_memory_tracking(True) is True
    cuda.reset_peak_memory_stats.assert_called_once()
    assert diagnostics_stage.read_peak_gpu_memory(True) == 4096


def test_diagnostics_contract_is_available_from_public_pipeline_api():
    from backend.pipeline import PipelineDiagnostics
    from backend.image_processor import PipelineDiagnostics as PublicDiagnostics

    assert PipelineDiagnostics.__name__ == "PipelineDiagnostics"
    assert PublicDiagnostics is PipelineDiagnostics


def test_missing_reconstruction_model_records_only_eligible_fallbacks():
    from backend.pipeline.orchestrator import (
        _mark_missing_reconstruction_model,
    )

    eligible = _object("object-0", "person", 1)
    eligible.reconstruction_mask = np.zeros((8, 8), dtype=bool)
    eligible.reconstruction_mask[3, 1] = True
    untouched = _object("object-1", "chair", 3)

    _mark_missing_reconstruction_model([eligible, untouched])

    assert eligible.reconstruction_failure_stage == "model_resolution"
    assert "no object-reconstruction model" in (
        eligible.reconstruction_failure_reason
    )
    assert untouched.reconstruction_failure_stage is None
