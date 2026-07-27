from unittest.mock import Mock

import numpy as np
from PIL import Image

from backend.pipeline.roi import SquareROI
from backend.pipeline.types import DetectedObject


def _detected(
    object_id: str,
    semantic_class: str,
    bbox: tuple[int, int, int, int],
    *,
    segmentation_index: int,
    shape: tuple[int, int] = (8, 10),
) -> DetectedObject:
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, width, height = bbox
    mask[y, x] = 255
    mask[y, x + width - 1] = 255
    mask[y + height - 1, x] = 255
    mask[y + height - 1, x + width - 1] = 255
    return DetectedObject(
        object_id=object_id,
        semantic_class=semantic_class,
        display_label=object_id,
        modal_mask=mask,
        bbox=bbox,
        segmentation_index=segmentation_index,
    )


def test_same_class_bbox_overlap_groups_pixel_disjoint_masks():
    from backend.pipeline.grouping import group_reconstructed_objects

    first = _detected("first", "person", (0, 0, 4, 4), segmentation_index=0)
    second = _detected("second", "person", (2, 2, 4, 4), segmentation_index=1)
    assert not np.any((first.modal_mask > 0) & (second.modal_mask > 0))
    first.amodal_mask = first.modal_mask > 0
    first.amodal_mask[1, 1] = True
    second.amodal_mask = second.modal_mask > 0
    second.amodal_mask[4, 4] = True
    second.reconstruction_canvas = Image.new("RGB", (3, 3), "red")

    groups = group_reconstructed_objects([first, second])

    assert len(groups) == 1
    group = groups[0]
    assert group.group_id == "group-first"
    assert group.semantic_class == "person"
    assert group.member_ids == ("first", "second")
    assert group.members == (first, second)
    assert group.bbox == (0, 0, 6, 6)
    assert np.array_equal(
        group.modal_mask > 0,
        (first.modal_mask > 0) | (second.modal_mask > 0),
    )
    assert np.array_equal(
        group.amodal_mask,
        first.amodal_mask | second.amodal_mask,
    )
    assert np.array_equal(
        group.effective_support_mask,
        (first.modal_mask > 0) | second.amodal_mask,
    )
    assert not group.effective_support_mask[1, 1]
    assert group.effective_support_mask[4, 4]
    assert group.has_reconstruction is True


def test_grouping_is_same_class_only_and_requires_positive_bbox_area():
    from backend.pipeline.grouping import group_reconstructed_objects

    first = _detected("first", "person", (0, 0, 3, 3), segmentation_index=0)
    touching = _detected(
        "touching", "person", (3, 0, 3, 3), segmentation_index=1
    )
    overlapping_class = _detected(
        "chair", "chair", (1, 1, 3, 3), segmentation_index=2
    )

    groups = group_reconstructed_objects(
        [first, touching, overlapping_class]
    )

    assert [group.member_ids for group in groups] == [
        ("first",),
        ("touching",),
        ("chair",),
    ]


def test_grouping_is_transitive_and_uses_stable_segmentation_order():
    from backend.pipeline.grouping import group_reconstructed_objects

    first = _detected("first", "rope", (0, 1, 3, 3), segmentation_index=0)
    second = _detected("second", "rope", (2, 1, 3, 3), segmentation_index=1)
    third = _detected("third", "rope", (4, 1, 3, 3), segmentation_index=2)
    assert first.original_modal_bbox[0] + 3 <= third.original_modal_bbox[0]

    groups = group_reconstructed_objects([third, first, second])

    assert len(groups) == 1
    assert groups[0].member_ids == ("first", "second", "third")
    assert groups[0].segmentation_index == 0


def test_grouped_amodal_union_uses_modal_fallback_per_member():
    from backend.pipeline.grouping import group_reconstructed_objects

    completed = _detected(
        "completed", "person", (0, 0, 4, 4), segmentation_index=0
    )
    fallback = _detected(
        "fallback", "person", (2, 2, 4, 4), segmentation_index=1
    )
    completed.amodal_mask = completed.modal_mask > 0
    completed.amodal_mask[1, 1] = True
    completed.reconstruction_canvas = Image.new("RGB", (4, 4), "red")
    fallback.amodal_mask = fallback.modal_mask > 0
    fallback.amodal_mask[4, 4] = True
    fallback.reconstruction_failure_stage = "generation_512"
    fallback.reconstruction_failure_reason = "test failure"

    group = group_reconstructed_objects([completed, fallback])[0]

    assert np.array_equal(
        group.amodal_mask,
        completed.amodal_mask | fallback.amodal_mask,
    )
    expected = completed.amodal_mask | (fallback.modal_mask > 0)
    assert np.array_equal(group.effective_support_mask, expected)
    assert not group.effective_support_mask[4, 4]
    assert group.has_reconstruction is True
    assert group.members[1].reconstruction_failure_stage == "generation_512"


def test_group_rgb_composition_maps_rois_and_applies_conflict_priority():
    from backend.pipeline.grouping import (
        compose_group_sources,
        group_reconstructed_objects,
    )

    image = Image.new("RGB", (6, 6), (10, 20, 30))
    first = _detected(
        "first",
        "person",
        (0, 0, 4, 4),
        segmentation_index=0,
        shape=(6, 6),
    )
    second = _detected(
        "second",
        "person",
        (2, 0, 4, 4),
        segmentation_index=1,
        shape=(6, 6),
    )
    first.reconstruction_roi = SquareROI(0, 0, 4, 6, 6)
    first.reconstruction_canvas = Image.new("RGB", (4, 4), "red")
    first.reconstruction_mask = np.zeros((6, 6), dtype=bool)
    first.reconstruction_mask[0, 2] = True  # Protected by second's modal mask.
    first.reconstruction_mask[1, 2] = True  # Wins reconstruction overlap.
    second.reconstruction_roi = SquareROI(2, 0, 4, 6, 6)
    second.reconstruction_canvas = Image.new("RGB", (4, 4), "blue")
    second.reconstruction_mask = np.zeros((6, 6), dtype=bool)
    second.reconstruction_mask[1, 2] = True
    second.reconstruction_mask[1, 4] = True
    group = group_reconstructed_objects([second, first])[0]

    compose_group_sources(image, [group])

    assert group.composed_source is not None
    assert group.composed_roi is not None

    def pixel(x, y):
        return group.composed_source.getpixel(
            (x - group.composed_roi.x, y - group.composed_roi.y)
        )

    assert pixel(2, 0) == (10, 20, 30)  # Original modal RGB wins.
    assert pixel(2, 1) == (255, 0, 0)  # First reconstruction wins.
    assert pixel(4, 1) == (0, 0, 255)  # Later non-conflicting RGB is used.
    assert pixel(1, 1) == (10, 20, 30)  # Untouched source is preserved.
    assert group.reconstruction_conflicts == (("first", "second"),)


def test_group_composition_uses_evidence_guided_write_and_support_masks():
    from backend.pipeline.grouping import (
        compose_group_sources,
        group_reconstructed_objects,
    )

    image = Image.new("RGB", (6, 6), (10, 20, 30))
    detected = _detected(
        "person",
        "person",
        (1, 1, 2, 2),
        segmentation_index=0,
        shape=(6, 6),
    )
    detected.amodal_mask = detected.modal_mask > 0
    detected.reconstruction_roi = SquareROI(0, 0, 6, 6, 6)
    detected.reconstruction_canvas = Image.new("RGB", (6, 6), "red")
    detected.reconstruction_mask = np.zeros((6, 6), dtype=bool)
    detected.reconstruction_mask[2, 2] = True
    detected.reconstruction_write_mask = (
        detected.reconstruction_mask.copy()
    )
    detected.reconstruction_write_mask[2, 3] = True
    detected.reconstruction_support_mask = (
        detected.amodal_mask.copy()
    )
    detected.reconstruction_support_mask[2, 3] = True

    group = group_reconstructed_objects([detected])[0]
    compose_group_sources(image, [group])

    assert group.effective_support_mask[2, 3]
    assert group.composed_source is not None
    assert group.composed_roi is not None
    local_x = 3 - group.composed_roi.x
    local_y = 2 - group.composed_roi.y
    assert group.composed_source.getpixel((local_x, local_y)) == (255, 0, 0)


def test_group_composition_blends_reconstruction_with_soft_write_alpha():
    from backend.pipeline.grouping import (
        compose_group_sources,
        group_reconstructed_objects,
    )

    image = Image.new("RGB", (6, 6), (10, 20, 30))
    detected = _detected(
        "person",
        "person",
        (1, 1, 2, 2),
        segmentation_index=0,
        shape=(6, 6),
    )
    detected.amodal_mask = detected.modal_mask > 0
    detected.reconstruction_roi = SquareROI(0, 0, 6, 6, 6)
    detected.reconstruction_canvas = Image.new("RGB", (6, 6), (210, 20, 30))
    detected.reconstruction_mask = np.zeros((6, 6), dtype=bool)
    detected.reconstruction_mask[2, 3] = True
    detected.reconstruction_write_mask = (
        detected.reconstruction_mask.copy()
    )
    detected.reconstruction_write_alpha = np.zeros(
        (6, 6), dtype=np.float64
    )
    detected.reconstruction_write_alpha[2, 3] = 0.5
    detected.reconstruction_support_mask = (
        detected.amodal_mask.copy()
    )
    detected.reconstruction_support_mask[2, 3] = True

    group = group_reconstructed_objects([detected])[0]
    compose_group_sources(image, [group])

    local_x = 3 - group.composed_roi.x
    local_y = 2 - group.composed_roi.y
    assert group.composed_source.getpixel((local_x, local_y)) == (
        110,
        20,
        30,
    )


def test_group_rgb_composition_initializes_fallback_group_from_original():
    from backend.pipeline.grouping import (
        compose_group_sources,
        group_reconstructed_objects,
    )

    image_array = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    image = Image.fromarray(image_array, mode="RGB")
    detected = _detected(
        "ordinary",
        "chair",
        (2, 1, 3, 3),
        segmentation_index=0,
        shape=(6, 8),
    )
    group = group_reconstructed_objects([detected])[0]

    compose_group_sources(image, [group])

    assert group.has_reconstruction is False
    assert group.composed_source is not None
    assert group.composed_roi is not None
    assert np.array_equal(
        np.asarray(group.composed_source),
        np.asarray(image.crop(group.composed_roi.box)),
    )


def test_orchestrator_groups_after_reconstruction_before_downstream(
    monkeypatch,
):
    from backend.pipeline import orchestrator

    first = _detected("first", "person", (0, 0, 3, 3), segmentation_index=0)
    second = _detected("second", "chair", (1, 1, 3, 3), segmentation_index=1)
    objects = [first, second]
    final_groups = [Mock(name="final-group")]
    events: list[str] = []

    manager = Mock()
    manager.has_object_reconstruction_model.return_value = True
    reconstruction_model = manager.get_object_reconstruction_model.return_value
    reconstruction_model.reconstruct = Mock()
    reconstruction_model.reconstruct_many = Mock()
    manager.get_matting_model.return_value.process = Mock()
    manager.get_background_inpainting_model.return_value.process = Mock()

    monkeypatch.setattr(orchestrator, "_segment", Mock(return_value=objects))
    monkeypatch.setattr(
        orchestrator,
        "link_overlap_partners",
        Mock(return_value=[("first", "second")]),
    )
    monkeypatch.setattr(orchestrator, "_complete_candidates", Mock())
    monkeypatch.setattr(
        orchestrator,
        "filter_pairs_by_amodal_overlap",
        Mock(return_value=[("first", "second")]),
    )

    def prepare(*_args, **_kwargs):
        first.reconstruction_mask = np.zeros_like(first.modal_mask, dtype=bool)
        first.reconstruction_mask[1, 1] = True

    monkeypatch.setattr(orchestrator, "prepare_raw_reconstruction_masks", prepare)
    def reconstruct(*_args, **_kwargs):
        first.reconstruction_canvas = Image.new("RGB", (3, 3), "red")
        first.reconstruction_roi = SquareROI(0, 0, 3, 10, 8)
        events.append("reconstruct")

    reconstruct_stage = Mock(side_effect=reconstruct)
    monkeypatch.setattr(
        orchestrator,
        "reconstruct_objects",
        reconstruct_stage,
    )

    def refine_support(_image, supplied, _matte, **kwargs):
        assert supplied == objects
        assert kwargs == {
            "alpha_low_threshold": 0.45,
            "alpha_high_threshold": 0.7,
            "change_threshold": 8.0,
            "connection_margin_pixels": 4,
            "max_extension_area_ratio": 2.0,
            "min_component_area_pixels": 8,
            "generation_evidence_margin_pixels": 2,
            "require_generation_evidence": True,
            "alpha_write_epsilon": 0.01,
            "alpha_feather_pixels": 0,
            "fallback_to_validated_output": True,
            "diagnostics_directory": "outputs/reconstruction_debug",
        }
        events.append("support")

    monkeypatch.setattr(
        orchestrator,
        "refine_reconstruction_supports",
        Mock(side_effect=refine_support),
        raising=False,
    )

    def group(supplied, decisions):
        assert supplied == objects
        assert decisions == []
        events.append("group")
        return final_groups

    monkeypatch.setattr(
        orchestrator,
        "group_reconstructed_objects",
        Mock(side_effect=group),
        raising=False,
    )

    def compose(_image, supplied):
        assert supplied == final_groups
        events.append("compose")

    monkeypatch.setattr(
        orchestrator,
        "compose_group_sources",
        Mock(side_effect=compose),
        raising=False,
    )

    def matte(_image, supplied, _matte, **kwargs):
        assert supplied == final_groups
        assert kwargs == {
            "context_ratio": 0.025,
            "support_dilation_pixels": 2,
        }
        events.append("matte")

    monkeypatch.setattr(
        orchestrator,
        "refine_objects",
        Mock(side_effect=matte),
    )

    def layers(supplied, _kernel_size, _background_inpaint, **kwargs):
        assert supplied == final_groups
        assert kwargs == {
            "final_alpha_threshold": 0.03,
            "final_min_component_area_pixels": 4,
        }
        events.append("layers")
        return []

    monkeypatch.setattr(
        orchestrator,
        "extract_object_layers",
        Mock(side_effect=layers),
    )
    monkeypatch.setattr(
        orchestrator,
        "generate_final_background",
        Mock(return_value=Image.new("RGB", (10, 8))),
    )
    monkeypatch.setattr(orchestrator, "_image_to_base64", Mock(return_value="bg"))

    orchestrator.process_image(
        Image.new("RGB", (10, 8)), ["person", "chair"], manager=manager
    )

    assert events == [
        "reconstruct",
        "support",
        "group",
        "compose",
        "matte",
        "layers",
    ]
    assert (
        reconstruct_stage.call_args.kwargs["reconstruct_many"]
        is reconstruction_model.reconstruct_many
    )
    manager.offload_model.assert_called_once_with("object_reconstruction")


def test_final_groups_are_ordered_back_to_front_from_pair_decisions():
    from backend.core.occlusion import PairDecision
    from backend.pipeline.grouping import group_reconstructed_objects

    hidden = _detected("person", "person", (0, 0, 3, 3), segmentation_index=9)
    front = _detected("camera", "camera", (1, 1, 3, 3), segmentation_index=0)
    decision = PairDecision(
        "person",
        "camera",
        "person",
        "camera",
        reconstruction_directions=(("person", "camera"),),
    )

    groups = group_reconstructed_objects([front, hidden], [decision])

    assert [group.member_ids for group in groups] == [
        ("person",),
        ("camera",),
    ]


def test_group_depth_order_ignores_internal_edges_after_same_class_merge():
    from backend.core.occlusion import PairDecision
    from backend.pipeline.grouping import group_reconstructed_objects

    first = _detected("first", "person", (0, 0, 3, 3), segmentation_index=1)
    second = _detected("second", "person", (2, 1, 3, 3), segmentation_index=0)
    decision = PairDecision(
        "first", "second", "first", "second",
        reconstruction_directions=(("first", "second"),),
    )

    groups = group_reconstructed_objects([first, second], [decision])

    assert len(groups) == 1
    assert groups[0].member_ids == ("second", "first")
