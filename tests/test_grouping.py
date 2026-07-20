from unittest.mock import Mock

import numpy as np
from PIL import Image

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
    fallback.reconstruction_failure_stage = "generation_512"
    fallback.reconstruction_failure_reason = "test failure"

    group = group_reconstructed_objects([completed, fallback])[0]

    expected = completed.amodal_mask | (fallback.modal_mask > 0)
    assert np.array_equal(group.amodal_mask, expected)
    assert group.has_reconstruction is False
    assert group.members[1].reconstruction_failure_stage == "generation_512"


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
    manager.get_object_reconstruction_model.return_value.reconstruct = Mock()
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
    monkeypatch.setattr(
        orchestrator,
        "reconstruct_objects",
        Mock(side_effect=lambda *_args, **_kwargs: events.append("reconstruct")),
    )

    def group(supplied):
        assert supplied == objects
        events.append("group")
        return final_groups

    monkeypatch.setattr(
        orchestrator,
        "group_reconstructed_objects",
        Mock(side_effect=group),
        raising=False,
    )

    def matte(_image, supplied, _matte):
        assert supplied == final_groups
        events.append("matte")

    monkeypatch.setattr(
        orchestrator,
        "refine_objects",
        Mock(side_effect=matte),
    )

    def layers(_image, _array, supplied, *_args):
        assert supplied == final_groups
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

    assert events == ["reconstruct", "group", "matte", "layers"]
