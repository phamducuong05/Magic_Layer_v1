"""Tests for object-, pair-, and group-level INFO decisions."""

import logging

import numpy as np
import torch
from PIL import Image

from backend.pipeline.types import DetectedObject


def _object(object_id, semantic_class, bbox):
    mask = np.zeros((6, 6), dtype=np.uint8)
    x, y, width, height = bbox
    mask[y : y + height, x : x + width] = 255
    return DetectedObject(
        object_id,
        semantic_class,
        object_id,
        mask,
        bbox,
    )


def test_overlap_and_amodal_validation_log_each_pair(caplog):
    from backend.pipeline.completion import (
        filter_pairs_by_amodal_overlap,
        link_overlap_partners,
    )

    person = _object("person", "person", (0, 0, 4, 4))
    chair = _object("chair", "chair", (2, 2, 4, 4))
    person.amodal_mask = person.modal_mask > 0
    chair.amodal_mask = chair.modal_mask > 0

    with caplog.at_level(logging.DEBUG):
        pairs = link_overlap_partners([person, chair])
        retained = filter_pairs_by_amodal_overlap(
            [person, chair], pairs
        )

    assert retained == [("person", "chair")]
    assert "[OVERLAP_DETECTION] PAIR_DECISION" in caplog.text
    assert "decision=completion_candidate" in caplog.text
    assert "[AMODAL_OVERLAP_VALIDATION] PAIR_DECISION" in caplog.text
    assert "decision=retain" in caplog.text


def test_depth_and_reconstruction_mask_log_each_object_and_pair(caplog):
    from backend.pipeline.reconstruction import prepare_raw_reconstruction_masks

    hidden = _object("hidden", "person", (0, 0, 4, 4))
    occluder = _object("occluder", "chair", (2, 2, 4, 4))
    for detected, hole_area in ((hidden, 4), (occluder, 1)):
        detected.amodal_mask = detected.modal_mask > 0
        detected.completion_hole_mask = np.zeros((6, 6), dtype=bool)
        detected.completion_hole_area = hole_area
    hidden.completion_hole_mask[4, 1:5] = True
    hidden.amodal_mask |= hidden.completion_hole_mask
    occluder.completion_hole_mask[1, 4] = True
    occluder.amodal_mask |= occluder.completion_hole_mask

    with caplog.at_level(logging.DEBUG):
        decisions = prepare_raw_reconstruction_masks(
            [hidden, occluder],
            [("hidden", "occluder")],
            (3, 3),
            minimum_hole_area_pixels=0,
            minimum_hole_area_ratio=0.0,
            tie_tolerance_ratio=0.0,
        )

    assert decisions[0].occluded_id == "hidden"
    assert "[DEPTH_ORDERING] OBJECT_HOLE" in caplog.text
    assert "[DEPTH_ORDERING] PAIR_DECISION" in caplog.text
    assert "occluded_id=hidden" in caplog.text
    assert "[RECONSTRUCTION_MASK] DECISION" in caplog.text
    assert "decision=created" in caplog.text


def test_grouping_logs_pair_decisions_and_final_members(caplog):
    from backend.pipeline.grouping import group_reconstructed_objects

    first = _object("first", "person", (0, 0, 3, 3))
    second = _object("second", "person", (2, 2, 3, 3))

    with caplog.at_level(logging.DEBUG):
        groups = group_reconstructed_objects([first, second])

    assert len(groups) == 1
    assert "[GROUPING] PAIR_DECISION" in caplog.text
    assert "decision=merge" in caplog.text
    assert "[GROUPING] GROUP_CREATED" in caplog.text
    assert "member_ids=first,second" in caplog.text


def test_reconstruction_logs_run_success_and_bypass_decisions(caplog):
    from backend.pipeline.reconstruction import reconstruct_objects

    hidden = _object("hidden", "person", (0, 0, 3, 3))
    hidden.amodal_mask = hidden.modal_mask > 0
    hidden.completion_hole_mask = np.zeros((6, 6), dtype=bool)
    hidden.completion_hole_mask[3, 1] = True
    hidden.amodal_mask |= hidden.completion_hole_mask
    hidden.reconstruction_mask = hidden.completion_hole_mask.copy()
    ordinary = _object("ordinary", "chair", (3, 3, 2, 2))

    with caplog.at_level(logging.DEBUG):
        reconstruct_objects(
            Image.new("RGB", (6, 6), "white"),
            [hidden, ordinary],
            lambda image, _mask, _prompt: image,
            context_ratio=0.0,
        )

    assert "[OBJECT_RECONSTRUCTION] OBJECT_DECISION" in caplog.text
    assert "decision=run" in caplog.text
    assert "decision=accepted" in caplog.text
    assert "decision=bypass" in caplog.text


def test_matting_logs_source_selection_and_alpha_result(caplog):
    from backend.pipeline.grouping import group_reconstructed_objects
    from backend.pipeline.matting import refine_objects

    detected = _object("person", "person", (1, 1, 3, 3))
    group = group_reconstructed_objects([detected])[0]

    with caplog.at_level(logging.DEBUG):
        refine_objects(
            Image.new("RGB", (6, 6), "white"),
            [group],
            lambda crop: torch.ones(crop.height, crop.width),
            context_ratio=0.0,
            support_dilation_pixels=0,
        )

    assert "[MATTING] GROUP_DECISION" in caplog.text
    assert "source=original_modal_rgb" in caplog.text
    assert "[MATTING] GROUP_RESULT" in caplog.text
