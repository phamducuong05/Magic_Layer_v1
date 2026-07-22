"""Occlusion role assignment and reconstruction mask construction."""

from collections.abc import Sequence

import numpy as np
import cv2

from ...core.layerd_refine import expand_mask
from ...core.logging import get_logger, log_event
from ...core.occlusion import (
    OverlapPair,
    PairDecision,
    assign_directional_pair_roles,
    effective_hole_area,
)
from ..types import DetectedObject


logger = get_logger(__name__)


def apply_pair_decisions(
    objects: Sequence[DetectedObject], decisions: Sequence[PairDecision]
) -> None:
    """Record each decisive pair's occluder on its occluded object."""
    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.occluder_ids.clear()
        detected.occluder_classes.clear()

    for decision in decisions:
        directions = decision.reconstruction_directions
        if not directions and not decision.ambiguous:
            directions = ((decision.occluded_id, decision.occluder_id),)
        if not directions:
            log_event(
                logger,
                "depth_ordering",
                "pair_assignment",
                first_id=decision.first_id,
                second_id=decision.second_id,
                decision="skip",
                reason="no_directional_completion_overlap",
            )
            continue
        for occluded_id, occluder_id in directions:
            occluded = objects_by_id[occluded_id]
            occluder = objects_by_id[occluder_id]
            occluded.occluder_ids.add(occluder_id)
            occluded.occluder_classes.add(occluder.semantic_class)
            log_event(
                logger,
                "depth_ordering",
                "pair_assignment",
                first_id=decision.first_id,
                second_id=decision.second_id,
                decision="assign",
                occluded_id=occluded_id,
                occluder_id=occluder_id,
            )


def build_reconstruction_masks(
    objects: Sequence[DetectedObject],
    kernel_size: tuple[int, int],
    *,
    generation_mask_dilation_pixels: int = 0,
    generation_mask_closing_pixels: int = 0,
    support_margin_pixels: int | None = None,
) -> None:
    """Build separate composition and model-generation masks."""
    for value in (
        generation_mask_dilation_pixels,
        generation_mask_closing_pixels,
    ):
        if value < 0:
            raise ValueError("reconstruction morphology settings must be non-negative")
    if support_margin_pixels is None:
        support_margin_pixels = max(kernel_size) // 2
    if support_margin_pixels < 0:
        raise ValueError("support_margin_pixels must be non-negative")
    # Tạo mapping để truy xuất đối tượng nhanh theo ID
    objects_by_id = {detected.object_id: detected for detected in objects}
    
    for detected in objects:
        detected.reconstruction_mask = None
        detected.reconstruction_generation_mask = None
        detected.reconstruction_occluder_mask = None
        
        # Bỏ qua nếu đối tượng không bị ai che khuất hoặc không có mặt nạ amodal/hole hợp lệ
        if (
            not detected.occluder_ids
            or detected.amodal_mask is None
            or detected.completion_hole_mask is None
        ):
            missing = []
            if not detected.occluder_ids:
                missing.append("no_assigned_occluder")
            if detected.amodal_mask is None:
                missing.append("missing_amodal_mask")
            if detected.completion_hole_mask is None:
                missing.append("missing_completion_hole_mask")
            log_event(
                logger,
                "reconstruction_mask",
                "decision",
                object_id=detected.object_id,
                decision="skip",
                reason=",".join(missing),
            )
            continue

        # Bước 1: Gộp tất cả modal mask (phần hiển thị) của các đối tượng che khuất (occluders)
        occluder_union = np.zeros_like(detected.modal_mask, dtype=bool)
        for occluder_id in detected.occluder_ids:
            occluder_union |= objects_by_id[occluder_id].modal_mask > 0

        # Bước 2: Giãn nở mặt nạ amodal mask của đối tượng hiện tại để bao phủ vùng biên lân cận
        support_kernel = 2 * support_margin_pixels + 1
        expanded_support = expand_mask(
            detected.amodal_mask > 0, (support_kernel, support_kernel)
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
        composition_mask = (
            detected.completion_hole_mask.astype(bool)
            & (detected.amodal_mask > 0)
            & ~(detected.modal_mask > 0)
        )
        generation_mask = composition_mask | relevant_occluder
        if generation_mask_closing_pixels:
            radius = generation_mask_closing_pixels
            closing_kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
            generation_mask = cv2.morphologyEx(
                generation_mask.astype(np.uint8),
                cv2.MORPH_CLOSE,
                closing_kernel,
            ).astype(bool)
        if generation_mask_dilation_pixels:
            radius = generation_mask_dilation_pixels
            generation_mask = expand_mask(
                generation_mask,
                (2 * radius + 1, 2 * radius + 1),
            ).astype(bool)
        generation_mask &= expanded_support
        generation_mask &= ~(detected.modal_mask > 0)
        generation_mask |= composition_mask

        detected.reconstruction_mask = composition_mask
        detected.reconstruction_generation_mask = generation_mask
        detected.reconstruction_occluder_mask = relevant_occluder
        log_event(
            logger,
            "reconstruction_mask",
            "decision",
            object_id=detected.object_id,
            decision="created",
            occluder_ids=sorted(detected.occluder_ids),
            completion_hole_pixels=int(
                np.count_nonzero(detected.completion_hole_mask)
            ),
            relevant_occluder_pixels=int(
                np.count_nonzero(relevant_occluder)
            ),
            composition_pixels=int(np.count_nonzero(composition_mask)),
            generation_pixels=int(
                np.count_nonzero(detected.reconstruction_generation_mask)
            ),
        )


def prepare_raw_reconstruction_masks(
    objects: Sequence[DetectedObject],
    retained_pairs: Sequence[OverlapPair],
    kernel_size: tuple[int, int],
    *,
    minimum_hole_area_pixels: int,
    minimum_hole_area_ratio: float,
    tie_tolerance_ratio: float,
    generation_mask_dilation_pixels: int = 0,
    generation_mask_closing_pixels: int = 0,
    support_margin_pixels: int | None = None,
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
        log_event(
            logger,
            "depth_ordering",
            "object_hole",
            object_id=detected.object_id,
            raw_hole_area=raw_area,
            effective_hole_area=effective_area,
            modal_area=int(np.count_nonzero(detected.modal_mask)),
            decision="retain" if effective_area else "suppress_as_noise",
        )

    objects_by_id = {detected.object_id: detected for detected in objects}
    directional_areas: dict[OverlapPair, int] = {}
    for first_id, second_id in retained_pairs:
        first = objects_by_id[first_id]
        second = objects_by_id[second_id]
        for target, occluder in ((first, second), (second, first)):
            raw_directional_area = int(
                np.count_nonzero(
                    target.completion_hole_mask.astype(bool)
                    & (occluder.modal_mask > 0)
                )
            )
            directional_areas[(target.object_id, occluder.object_id)] = (
                effective_hole_area(
                    raw_directional_area,
                    int(np.count_nonzero(target.modal_mask)),
                    minimum_pixels=minimum_hole_area_pixels,
                    minimum_modal_ratio=minimum_hole_area_ratio,
                )
            )

    decisions = assign_directional_pair_roles(
        retained_pairs,
        directional_areas,
        tie_tolerance_ratio=tie_tolerance_ratio,
    )
    for decision in decisions:
        log_event(
            logger,
            "depth_ordering",
            "pair_decision",
            first_id=decision.first_id,
            second_id=decision.second_id,
            first_hole_area=effective_areas[decision.first_id],
            second_hole_area=effective_areas[decision.second_id],
            first_hidden_by_second=decision.first_hidden_by_second_area,
            second_hidden_by_first=decision.second_hidden_by_first_area,
            reconstruction_directions=decision.reconstruction_directions,
            decision="ambiguous" if decision.ambiguous else "ordered",
            occluded_id=decision.occluded_id,
            occluder_id=decision.occluder_id,
        )
    apply_pair_decisions(objects, decisions)
    build_reconstruction_masks(
        objects,
        kernel_size,
        generation_mask_dilation_pixels=generation_mask_dilation_pixels,
        generation_mask_closing_pixels=generation_mask_closing_pixels,
        support_margin_pixels=support_margin_pixels,
    )
    return decisions
