"""Occlusion role assignment and reconstruction mask construction."""

from collections.abc import Sequence

import numpy as np

from ...core.layerd_refine import expand_mask
from ...core.occlusion import (
    OverlapPair,
    PairDecision,
    assign_pair_roles,
    effective_hole_area,
)
from ..types import DetectedObject


def apply_pair_decisions(
    objects: Sequence[DetectedObject], decisions: Sequence[PairDecision]
) -> None:
    """Record each decisive pair's occluder on its occluded object."""
    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.occluder_ids.clear()

    for decision in decisions:
        if decision.ambiguous:
            continue
        objects_by_id[decision.occluded_id].occluder_ids.add(
            decision.occluder_id
        )


def build_reconstruction_masks(
    objects: Sequence[DetectedObject], kernel_size: tuple[int, int]
) -> None:
    """Build constrained masks for objects with assigned occluders."""
    # Tạo mapping để truy xuất đối tượng nhanh theo ID
    objects_by_id = {detected.object_id: detected for detected in objects}
    
    for detected in objects:
        detected.reconstruction_mask = None
        
        # Bỏ qua nếu đối tượng không bị ai che khuất hoặc không có mặt nạ amodal/hole hợp lệ
        if (
            not detected.occluder_ids
            or detected.amodal_mask is None
            or detected.completion_hole_mask is None
        ):
            continue

        # Bước 1: Gộp tất cả modal mask (phần hiển thị) của các đối tượng che khuất (occluders)
        occluder_union = np.zeros_like(detected.modal_mask, dtype=bool)
        for occluder_id in detected.occluder_ids:
            occluder_union |= objects_by_id[occluder_id].modal_mask > 0

        # Bước 2: Giãn nở mặt nạ amodal mask của đối tượng hiện tại để bao phủ vùng biên lân cận
        expanded_support = expand_mask(
            detected.amodal_mask > 0, kernel_size
        ).astype(bool)
        
        # Bước 3: Xác định vùng che khuất thực sự liên quan
        # Lấy phần giao giữa (các occluders) và (vùng biên amodal mở rộng),
        # sau đó loại trừ phần hiển thị thực tế (modal mask) của chính đối tượng đó.
        relevant_occluder = (
            occluder_union
            & expanded_support
            & ~(detected.modal_mask > 0)
        )
        
        # Bước 4: Tạo mặt nạ phục dựng (reconstruction mask) cuối cùng bằng cách
        # gộp phần completion hole đã có với phần occluder liên quan vừa tìm được.
        detected.reconstruction_mask = (
            detected.completion_hole_mask | relevant_occluder
        )


def prepare_raw_reconstruction_masks(
    objects: Sequence[DetectedObject],
    retained_pairs: Sequence[OverlapPair],
    kernel_size: tuple[int, int],
    *,
    minimum_hole_area_pixels: int,
    minimum_hole_area_ratio: float,
    tie_tolerance_ratio: float,
) -> list[PairDecision]:
    """Decide pairwise depth and build one reconstruction mask per raw object."""
    effective_areas: dict[str, int] = {}
    for detected in objects:
        raw_area = detected.completion_hole_area
        if raw_area is None:
            continue

        effective_area = effective_hole_area(
            raw_area,
            int(np.count_nonzero(detected.modal_mask)),
            minimum_pixels=minimum_hole_area_pixels,
            minimum_modal_ratio=minimum_hole_area_ratio,
        )
        detected.effective_completion_hole_area = effective_area
        effective_areas[detected.object_id] = effective_area

    decisions = assign_pair_roles(
        retained_pairs,
        effective_areas,
        tie_tolerance_ratio=tie_tolerance_ratio,
    )
    apply_pair_decisions(objects, decisions)
    build_reconstruction_masks(objects, kernel_size)
    return decisions
