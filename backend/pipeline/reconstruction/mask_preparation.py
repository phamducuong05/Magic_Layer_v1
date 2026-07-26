"""Occlusion role assignment and reconstruction mask construction."""

from collections.abc import Sequence

import cv2
import numpy as np

from ...core.layerd_refine import expand_mask
from ...core.logging import get_logger, log_event
from ...core.occlusion import (
    OverlapPair,
    PairDecision,
    assign_directional_pair_roles,
    effective_hole_area,
)
from ..roi import SquareROI, square_roi_from_support
from ..types import DetectedObject


logger = get_logger(__name__)


def _mask_inside_roi(shape: tuple[int, int], roi: SquareROI) -> np.ndarray:
    """Tạo ra một 'Mặt nạ Cửa sổ' (Window Mask) dạng boolean có kích thước bằng ảnh gốc.

    Trong ma trận trả về:
    - Tất cả các pixel nằm BÊN TRONG khung hình vuông ROI (`top:bottom, left:right`) có giá trị True (1).
    - Tất cả các pixel nằm BÊN NGOÀI khung hình vuông ROI có giá trị False (0).

    Dùng để giới hạn phạm vi mask của Occluder hoặc mask sinh ảnh không vượt ra ngoài biên của ô vuông crop ROI.
    """
    # Khởi tạo ma trận toàn bộ pixel = False (0) với kích thước bằng ảnh gốc
    result = np.zeros(shape, dtype=bool)
    # Lấy tọa độ khung viền (left, top, right, bottom) của ô vuông ROI
    left, top, right, bottom = roi.clipped_box
    # Gán True (1) cho vùng pixel nằm bên trong khung ROI
    result[top:bottom, left:right] = True
    return result


def _bbox_mask(mask: np.ndarray) -> np.ndarray:
    """Return a full-image window covering the tight bounds of ``mask``."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("target support mask must be non-empty")
    result = np.zeros_like(mask, dtype=bool)
    result[
        int(ys.min()) : int(ys.max()) + 1,
        int(xs.min()) : int(xs.max()) + 1,
    ] = True
    return result


def _fill_small_mask_holes(
    mask: np.ndarray,
    *,
    max_hole_area_pixels: int,
) -> np.ndarray:
    """Fill enclosed background components up to a configured pixel area."""
    result = mask.astype(bool).copy()
    if max_hole_area_pixels <= 0 or not np.any(result):
        return result

    background = (~result).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        background,
        connectivity=8,
    )
    height, width = result.shape
    for label in range(1, component_count):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = (
            left == 0
            or top == 0
            or left + component_width >= width
            or top + component_height >= height
        )
        if not touches_border and area <= max_hole_area_pixels:
            result[labels == label] = True
    return result


def _close_and_dilate_mask(
    mask: np.ndarray,
    *,
    closing_pixels: int,
    dilation_pixels: int,
) -> np.ndarray:
    """Apply the reconstruction morphology policy to one boolean mask."""
    result = mask.astype(bool).copy()
    if closing_pixels:
        closing_kernel = np.ones(
            (2 * closing_pixels + 1, 2 * closing_pixels + 1),
            dtype=np.uint8,
        )
        result = cv2.morphologyEx(
            result.astype(np.uint8),
            cv2.MORPH_CLOSE,
            closing_kernel,
        ).astype(bool)
    if dilation_pixels:
        result = expand_mask(
            result,
            (2 * dilation_pixels + 1, 2 * dilation_pixels + 1),
        ).astype(bool)
    return result


def _directional_reconstruction_mask(
    target: DetectedObject,
    occluder: DetectedObject,
    *,
    composition_margin_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo mask tái tạo có định hướng cho trường hợp đối tượng `target` bị che bởi `occluder`.

    Args:
        target: Đối tượng đóng vai trò bị che khuất (cần vẽ bù).
        occluder: Đối tượng đóng vai trò che khuất (nằm đè lên trên).
        composition_margin_pixels: Bán kính pixel mở rộng biên lề an toàn để bao trùm sai số ranh giới.

    Returns:
        tuple[np.ndarray, np.ndarray]: Bộ đôi gồm:
            - exact_seed: Mask hạt giống giao nhau chính xác 100% giữa lỗ khuyết của target và mask hiển thị của occluder.
            - filtered: Mask khôi phục hoàn chỉnh sau khi mở rộng biên lề, lọc thành phần liên thông và cắt xén hợp lệ.
    """
    # Lấy vùng lỗ khuyết (vùng bị che khuất) của đối tượng target
    hole = target.completion_hole_mask.astype(bool)
    # Lấy vùng phần nhìn thấy được (modal mask) của đối tượng occluder
    occluder_modal = occluder.modal_mask > 0

    # 1. Tìm vùng hạt giống giao nhau chính xác 100% (exact_seed)
    exact_seed = hole & occluder_modal
    # Nếu không có pixel nào giao nhau -> target không thực sự bị occluder này che khuất
    if not np.any(exact_seed):
        return exact_seed, np.zeros_like(hole)

    # 2. Nở rộng biên occluder thêm composition_margin_pixels để bao trùm sai số ranh giới & bóng mờ
    if composition_margin_pixels:
        radius = composition_margin_pixels
        occluder_support = expand_mask(
            occluder_modal,
            (2 * radius + 1, 2 * radius + 1),
        ).astype(bool)
    else:
        occluder_support = occluder_modal

    # Vùng ứng viên mở rộng
    candidate = hole & occluder_support

    # 3. Phân chia candidate thành các khối/đảo pixel liên thông (connectivity=8) bằng OpenCV
    _, labels = cv2.connectedComponents(
        candidate.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(candidate)

    # Chỉ giữ lại những khối pixel nào NẰM TRÊN hoặc CHỨA hạt giống xác thực (exact_seed)
    for label in np.unique(labels[exact_seed]):
        if label != 0:
            filtered |= labels == label

    # 4. Kiểm tra an toàn: Mask tái tạo phải nằm trong amodal_mask và nằm ngoài modal_mask của target
    filtered &= target.amodal_mask > 0
    filtered &= ~(target.modal_mask > 0)

    return exact_seed, filtered


def apply_pair_decisions(
    objects: Sequence[DetectedObject], decisions: Sequence[PairDecision]
) -> None:
    """Record every valid directional occluder on its target object."""
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
    foreign_modal_max_hole_area_pixels: int = 0,
    foreign_modal_dilation_pixels: int = 0,
    foreign_modal_closing_pixels: int = 0,
    replacement_domain_margin_pixels: int = 0,
    foreign_protection_dilation_pixels: int = 0,
    support_margin_pixels: int | None = None,
    composition_margin_pixels: int | None = None,
    context_ratio: float = 0.0,
) -> None:
    """Build separate directional write-back and model-generation masks."""
    for value in (
        generation_mask_dilation_pixels,
        generation_mask_closing_pixels,
        foreign_modal_max_hole_area_pixels,
        foreign_modal_dilation_pixels,
        foreign_modal_closing_pixels,
        replacement_domain_margin_pixels,
        foreign_protection_dilation_pixels,
    ):
        if value < 0:
            raise ValueError("reconstruction morphology settings must be non-negative")
    if support_margin_pixels is None:
        support_margin_pixels = max(kernel_size) // 2
    if composition_margin_pixels is None:
        composition_margin_pixels = support_margin_pixels
    if support_margin_pixels < 0 or composition_margin_pixels < 0:
        raise ValueError("reconstruction margin settings must be non-negative")

    objects_by_id = {detected.object_id: detected for detected in objects}
    for detected in objects:
        detected.reconstruction_seed_mask = None
        detected.reconstruction_mask = None
        detected.reconstruction_generation_seed_mask = None
        detected.reconstruction_generation_mask = None
        detected.reconstruction_occluder_mask = None
        detected.reconstruction_input_roi = None
        detected.reconstruction_target_bbox_mask = None
        detected.reconstruction_foreign_modal_inside_bbox = None
        detected.reconstruction_foreign_modal_outside_bbox = None
        detected.reconstruction_replacement_domain_mask = None
        detected.reconstruction_replaceable_foreign_inside = None
        detected.reconstruction_protected_foreign_inside = None
        detected.reconstruction_foreign_protection_mask = None
        detected.reconstruction_protected_mask = None
        detected.reconstruction_accepted_rgb_mask = None
        detected.raw_reconstruction_canvas = None
        detected.reconstruction_canvas = None
        detected.reconstruction_roi = None
        detected.reconstruction_evidence_alpha = None
        detected.reconstruction_extension_mask = None
        detected.reconstruction_write_alpha = None
        detected.reconstruction_write_mask = None
        detected.reconstruction_support_mask = None
        detected.reconstruction_failure_stage = None
        detected.reconstruction_failure_reason = None

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

        occluder_union = np.zeros_like(detected.modal_mask, dtype=bool)
        exact_seed = np.zeros_like(detected.modal_mask, dtype=bool)
        composition_mask = np.zeros_like(detected.modal_mask, dtype=bool)
        assigned_occluder_modal_masks: list[np.ndarray] = []
        for occluder_id in detected.occluder_ids:
            occluder = objects_by_id[occluder_id]
            occluder_modal = _fill_small_mask_holes(
                occluder.modal_mask > 0,
                max_hole_area_pixels=foreign_modal_max_hole_area_pixels,
            )
            occluder_union |= occluder_modal
            assigned_occluder_modal_masks.append(occluder_modal)
            directional_seed, directional_mask = (
                _directional_reconstruction_mask(
                    detected,
                    occluder,
                    composition_margin_pixels=composition_margin_pixels,
                )
            )
            exact_seed |= directional_seed
            composition_mask |= directional_mask

        if not np.any(composition_mask):
            log_event(
                logger,
                "reconstruction_mask",
                "decision",
                object_id=detected.object_id,
                decision="skip",
                reason="no_filtered_directional_hole",
            )
            continue

        # Freeze ROI before adding the full occluder so the crop stays centered
        # on the target rather than growing to the occluder's complete bounds.
        roi = square_roi_from_support(
            (detected.amodal_mask > 0) | composition_mask,
            context_ratio=context_ratio,
        )
        roi_mask = _mask_inside_roi(composition_mask.shape, roi)
        target_modal = detected.modal_mask > 0
        target_bbox_mask = _bbox_mask(detected.amodal_mask > 0)
        foreign_modal = np.zeros_like(target_modal)
        for other in objects:
            if other.object_id != detected.object_id:
                foreign_modal |= _fill_small_mask_holes(
                    other.modal_mask > 0,
                    max_hole_area_pixels=(
                        foreign_modal_max_hole_area_pixels
                    ),
                )
        foreign_inside_bbox = foreign_modal & target_bbox_mask & roi_mask
        foreign_outside_bbox = foreign_modal & ~target_bbox_mask & roi_mask
        foreign_outside_bbox = _close_and_dilate_mask(
            foreign_outside_bbox,
            closing_pixels=foreign_modal_closing_pixels,
            dilation_pixels=foreign_modal_dilation_pixels,
        )
        foreign_outside_bbox &= ~target_bbox_mask
        foreign_outside_bbox &= roi_mask

        assigned_occluder = (
            occluder_union & foreign_inside_bbox & ~target_modal
        )
        replacement_domain = _close_and_dilate_mask(
            (detected.amodal_mask > 0) | composition_mask,
            closing_pixels=0,
            dilation_pixels=replacement_domain_margin_pixels,
        )
        replacement_domain &= target_bbox_mask
        replacement_domain &= roi_mask
        replaceable_foreign_inside = (
            assigned_occluder
            & replacement_domain
            & roi_mask
            & ~target_modal
        )
        protected_foreign_inside = (
            foreign_inside_bbox & ~replaceable_foreign_inside
        )
        foreign_protection = _close_and_dilate_mask(
            foreign_outside_bbox | protected_foreign_inside,
            closing_pixels=0,
            dilation_pixels=foreign_protection_dilation_pixels,
        )
        foreign_protection &= ~replaceable_foreign_inside
        foreign_protection &= roi_mask

        qualifying_occluders_in_roi = np.zeros_like(target_modal)
        for occluder_modal in assigned_occluder_modal_masks:
            if np.any(occluder_modal & replaceable_foreign_inside):
                qualifying_occluders_in_roi |= occluder_modal & roi_mask

        generation_seed = composition_mask | qualifying_occluders_in_roi
        generation_mask = _close_and_dilate_mask(
            generation_seed,
            closing_pixels=generation_mask_closing_pixels,
            dilation_pixels=generation_mask_dilation_pixels,
        )
        qualifying_occluder_allowance = _close_and_dilate_mask(
            qualifying_occluders_in_roi,
            closing_pixels=generation_mask_closing_pixels,
            dilation_pixels=generation_mask_dilation_pixels,
        )
        generation_mask &= roi_mask
        generation_mask &= ~target_modal
        generation_mask &= (
            ~foreign_protection | qualifying_occluder_allowance
        )
        generation_mask |= composition_mask

        protected_mask = target_modal | foreign_protection
        accepted_rgb_mask = roi_mask & ~protected_mask

        detected.reconstruction_seed_mask = exact_seed
        detected.reconstruction_mask = composition_mask
        detected.reconstruction_generation_seed_mask = generation_seed
        detected.reconstruction_generation_mask = generation_mask
        detected.reconstruction_occluder_mask = assigned_occluder
        detected.reconstruction_input_roi = roi
        detected.reconstruction_target_bbox_mask = target_bbox_mask
        detected.reconstruction_foreign_modal_inside_bbox = (
            foreign_inside_bbox
        )
        detected.reconstruction_foreign_modal_outside_bbox = (
            foreign_outside_bbox
        )
        detected.reconstruction_replacement_domain_mask = replacement_domain
        detected.reconstruction_replaceable_foreign_inside = (
            replaceable_foreign_inside
        )
        detected.reconstruction_protected_foreign_inside = (
            protected_foreign_inside
        )
        detected.reconstruction_foreign_protection_mask = foreign_protection
        detected.reconstruction_protected_mask = protected_mask
        detected.reconstruction_accepted_rgb_mask = accepted_rgb_mask
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
            directional_seed_pixels=int(np.count_nonzero(exact_seed)),
            assigned_occluder_pixels=int(
                np.count_nonzero(assigned_occluder)
            ),
            replacement_domain_pixels=int(
                np.count_nonzero(replacement_domain)
            ),
            replaceable_foreign_inside_pixels=int(
                np.count_nonzero(replaceable_foreign_inside)
            ),
            protected_foreign_inside_pixels=int(
                np.count_nonzero(protected_foreign_inside)
            ),
            foreign_protection_pixels=int(
                np.count_nonzero(foreign_protection)
            ),
            composition_pixels=int(np.count_nonzero(composition_mask)),
            generation_pixels=int(np.count_nonzero(generation_mask)),
            accepted_rgb_pixels=int(np.count_nonzero(accepted_rgb_mask)),
            foreign_modal_inside_bbox_pixels=int(
                np.count_nonzero(foreign_inside_bbox)
            ),
            foreign_modal_outside_bbox_pixels=int(
                np.count_nonzero(foreign_outside_bbox)
            ),
            roi=(roi.x, roi.y, roi.size),
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
    foreign_modal_max_hole_area_pixels: int = 0,
    foreign_modal_dilation_pixels: int = 0,
    foreign_modal_closing_pixels: int = 0,
    replacement_domain_margin_pixels: int = 0,
    foreign_protection_dilation_pixels: int = 0,
    support_margin_pixels: int | None = None,
    composition_margin_pixels: int | None = None,
    context_ratio: float = 0.0,
) -> list[PairDecision]:
    """Filter directional holes, decide depth, and build raw-object masks."""
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

    if composition_margin_pixels is None:
        composition_margin_pixels = (
            max(kernel_size) // 2
            if support_margin_pixels is None
            else support_margin_pixels
        )
    objects_by_id = {detected.object_id: detected for detected in objects}
    directional_areas: dict[OverlapPair, int] = {}
    for first_id, second_id in retained_pairs:
        first = objects_by_id[first_id]
        second = objects_by_id[second_id]
        for target, occluder in ((first, second), (second, first)):
            _, directional_mask = _directional_reconstruction_mask(
                target,
                occluder,
                composition_margin_pixels=composition_margin_pixels,
            )
            raw_directional_area = int(np.count_nonzero(directional_mask))
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
            first_hole_area=effective_areas.get(decision.first_id, 0),
            second_hole_area=effective_areas.get(decision.second_id, 0),
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
        foreign_modal_max_hole_area_pixels=(
            foreign_modal_max_hole_area_pixels
        ),
        foreign_modal_dilation_pixels=foreign_modal_dilation_pixels,
        foreign_modal_closing_pixels=foreign_modal_closing_pixels,
        replacement_domain_margin_pixels=replacement_domain_margin_pixels,
        foreign_protection_dilation_pixels=(
            foreign_protection_dilation_pixels
        ),
        support_margin_pixels=support_margin_pixels,
        composition_margin_pixels=composition_margin_pixels,
        context_ratio=context_ratio,
    )
    return decisions
