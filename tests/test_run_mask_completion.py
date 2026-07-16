"""Tests for the real-model mask-completion command-line output helper."""

from unittest.mock import Mock

import numpy as np
from PIL import Image

from backend.pipeline.types import DetectedObject
from backend.run_mask_completion import (
    build_diagnostic_groups,
    run_diagnostics,
    save_masks,
    save_workflow_visualizations,
)


def test_save_masks_writes_binary_full_size_pngs(tmp_path):
    first = np.zeros((3, 4), dtype=bool)
    first[1, 2] = True
    second = np.zeros((3, 4), dtype=np.uint8)
    second[0, 0] = 255

    paths = save_masks([first, second], tmp_path / "masks")

    assert [path.name for path in paths] == ["mask_0.png", "mask_1.png"]
    saved_first = np.asarray(Image.open(paths[0]))
    saved_second = np.asarray(Image.open(paths[1]))
    assert saved_first.shape == (3, 4)
    assert set(np.unique(saved_first)) == {0, 255}
    assert set(np.unique(saved_second)) == {0, 255}


def _object(object_id, semantic_class, mask, bbox):
    return DetectedObject(
        object_id=object_id,
        semantic_class=semantic_class,
        display_label=object_id,
        modal_mask=mask.astype(np.uint8) * 255,
        bbox=bbox,
    )


def test_diagnostic_groups_use_same_class_original_bbox_and_validated_masks():
    first_mask = np.zeros((6, 8), dtype=bool)
    first_mask[0:3, 0:3] = True
    second_mask = np.zeros((6, 8), dtype=bool)
    second_mask[2:5, 2:5] = True
    other_class_mask = np.zeros((6, 8), dtype=bool)
    other_class_mask[1:4, 1:4] = True
    first = _object("first", "person", first_mask, (0, 0, 3, 3))
    second = _object("second", "person", second_mask, (2, 2, 3, 3))
    chair = _object("chair", "chair", other_class_mask, (1, 1, 3, 3))
    first.amodal_mask = first_mask.copy()
    first.amodal_mask[0, 3] = True
    second.amodal_mask = second_mask.copy()
    chair.amodal_mask = other_class_mask.copy()

    groups = build_diagnostic_groups([first, second, chair])

    assert [group.member_ids for group in groups] == [
        ["first", "second"],
        ["chair"],
    ]
    assert groups[0].semantic_class == "person"
    assert np.array_equal(
        groups[0].mask, first.amodal_mask | second.amodal_mask
    )


def test_workflow_visualizations_save_each_requested_stage(tmp_path):
    first_mask = np.zeros((6, 8), dtype=bool)
    first_mask[0:4, 0:4] = True
    second_mask = np.zeros((6, 8), dtype=bool)
    second_mask[2:6, 2:6] = True
    first = _object("first", "person", first_mask, (0, 0, 4, 4))
    second = _object("second", "chair", second_mask, (2, 2, 4, 4))
    first.amodal_mask = first_mask.copy()
    first.amodal_mask[4, 1] = True
    second.amodal_mask = second_mask.copy()
    first.completion_hole_mask = first.amodal_mask & ~first_mask
    second.completion_hole_mask = np.zeros_like(second_mask)
    first.completion_hole_area = 1
    second.completion_hole_area = 0
    groups = build_diagnostic_groups([first, second])
    output_dir = tmp_path / "diagnostics"

    summary_path = save_workflow_visualizations(
        Image.new("RGB", (8, 6), "gray"),
        [first, second],
        [("first", "second")],
        [("first", "second")],
        {"first": first.amodal_mask, "second": second.amodal_mask},
        groups,
        output_dir,
    )

    expected = [
        output_dir / "00_original.png",
        output_dir / "01_raw_masks" / "overview.png",
        output_dir / "02_cross_class_bbox" / "candidate_pairs.png",
        output_dir / "03_raw_completion" / "first_predicted.png",
        output_dir / "04_validated_completion" / "first_validated.png",
        output_dir / "04_validated_completion" / "first_completion_hole.png",
        output_dir / "05_amodal_pair_filter" / "retained_pairs.png",
        output_dir / "06_diagnostic_groups" / "group-0_mask.png",
        output_dir / "07_final_overview.png",
        output_dir / "summary.json",
    ]
    assert summary_path == output_dir / "summary.json"
    assert all(path.is_file() for path in expected)


def test_diagnostics_do_not_load_completion_model_without_bbox_candidates(
    tmp_path,
):
    class EmptyProcessor:
        def set_image(self, image):
            del image
            return {}

        def reset_all_prompts(self, state):
            del state

        def set_text_prompt(self, state, prompt):
            del prompt
            state["masks"] = []
            return state

    get_completion_model = Mock()

    summary_path = run_diagnostics(
        Image.new("RGB", (8, 6)),
        ["missing"],
        tmp_path / "diagnostics",
        processor=EmptyProcessor(),
        get_completion_model=get_completion_model,
        completion_config={
            "max_area_growth_ratio": 4.0,
            "max_bbox_growth_ratio": 9.0,
        },
    )

    assert summary_path.is_file()
    get_completion_model.assert_not_called()
