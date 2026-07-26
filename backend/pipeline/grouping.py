"""Stable post-reconstruction grouping of same-class raw objects."""

from collections.abc import Sequence

import numpy as np
from PIL import Image

from ..core.helpers import _bbox_from_mask
from ..core.logging import get_logger, log_event
from ..core.occlusion import PairDecision
from .roi import crop_image, square_roi_from_support
from .types import DetectedObject, GroupedObject


logger = get_logger(__name__)


def _boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    """Return whether two ``(x, y, width, height)`` boxes overlap by area."""
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return (
        max(first_x, second_x)
        < min(first_x + first_width, second_x + second_width)
        and max(first_y, second_y)
        < min(first_y + first_height, second_y + second_height)
    )


def group_reconstructed_objects(
    objects: Sequence[DetectedObject],
    pair_decisions: Sequence[PairDecision] = (),
) -> list[GroupedObject]:
    """Group same-class raw objects using original modal bboxes only."""
    if not objects:
        return []

    input_order = {id(detected): index for index, detected in enumerate(objects)}
    ordered = sorted(
        objects,
        key=lambda detected: (
            detected.segmentation_index,
            input_order[id(detected)],
        ),
    )
    expected_shape = ordered[0].modal_mask.shape
    if any(detected.modal_mask.shape != expected_shape for detected in ordered):
        raise ValueError("all grouped masks must share full-image dimensions")

    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            second = ordered[second_index]
            same_class = first.semantic_class == second.semantic_class
            bbox_overlap = _boxes_overlap(
                first.original_modal_bbox,
                second.original_modal_bbox,
            )
            if same_class and bbox_overlap:
                union(first_index, second_index)
                decision = "merge"
                reason = "same_class_bbox_overlap"
            elif not same_class:
                decision = "keep_separate"
                reason = "different_semantic_class"
            else:
                decision = "keep_separate"
                reason = "original_modal_bboxes_do_not_overlap"
            log_event(
                logger,
                "grouping",
                "pair_decision",
                first_id=first.object_id,
                second_id=second.object_id,
                decision=decision,
                reason=reason,
            )

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        grouped_indices.setdefault(find(index), []).append(index)

    groups: list[GroupedObject] = []
    for indices in grouped_indices.values():
        members = tuple(ordered[index] for index in indices)
        grouped_modal = np.zeros(expected_shape, dtype=bool)
        grouped_amodal = np.zeros(expected_shape, dtype=bool)
        for member in members:
            modal = member.modal_mask > 0
            grouped_modal |= modal
            grouped_amodal |= (
                member.amodal_mask > 0
                if member.amodal_mask is not None
                else modal
            )

        bbox = _bbox_from_mask(grouped_modal)
        if bbox is None:
            raise ValueError("a final group cannot contain only empty masks")
        first = members[0]
        group = GroupedObject(
            group_id=f"group-{first.object_id}",
            semantic_class=first.semantic_class,
            display_label=first.display_label,
            member_ids=tuple(member.object_id for member in members),
            members=members,
            modal_mask=grouped_modal.astype(np.uint8) * 255,
            amodal_mask=grouped_amodal,
            bbox=bbox,
            segmentation_index=first.segmentation_index,
        )
        groups.append(group)
        log_event(
            logger,
            "grouping",
            "group_created",
            group_id=group.group_id,
            semantic_class=group.semantic_class,
            member_ids=list(group.member_ids),
            bbox=group.bbox,
        )
    if not pair_decisions or len(groups) < 2:
        return groups

    group_index_by_member = {
        member_id: index
        for index, group in enumerate(groups)
        for member_id in group.member_ids
    }
    edges: list[set[int]] = [set() for _ in groups]
    indegree = [0] * len(groups)
    for decision in pair_decisions:
        if decision.ambiguous:
            continue
        back = group_index_by_member.get(decision.occluded_id)
        front = group_index_by_member.get(decision.occluder_id)
        if back is None or front is None or back == front or front in edges[back]:
            continue
        edges[back].add(front)
        indegree[front] += 1

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    ordered_indices: list[int] = []
    while ready:
        current = ready.pop(0)
        ordered_indices.append(current)
        for following in sorted(edges[current]):
            indegree[following] -= 1
            if indegree[following] == 0:
                ready.append(following)
                ready.sort()

    if len(ordered_indices) != len(groups):
        emitted = set(ordered_indices)
        ordered_indices.extend(
            index for index in range(len(groups)) if index not in emitted
        )
        log_event(
            logger,
            "grouping",
            "depth_cycle",
            level="WARNING",
            decision="stable_segmentation_fallback",
        )

    depth_ordered = [groups[index] for index in ordered_indices]
    log_event(
        logger,
        "grouping",
        "depth_order",
        level="INFO",
        back_to_front=[group.group_id for group in depth_ordered],
    )
    return depth_ordered


def compose_group_sources(
    image: Image.Image, groups: Sequence[GroupedObject]
) -> None:
    """Attach one conflict-resolved RGB source crop to every final group."""
    source = image.convert("RGB")
    source_width, source_height = source.size

    for group in groups:
        # Cover only support backed by usable RGB; no extra padding is needed.
        roi = square_roi_from_support(
            group.effective_support_mask,
            context_ratio=0.0,
        )
        if roi.image_size != source.size:
            raise ValueError(
                "group masks must match the source image dimensions"
            )

        # Initialize the composition from the original source crop.
        composed = np.asarray(crop_image(source, roi), dtype=np.uint8).copy()

        # Visible source pixels must never be overwritten by reconstruction.
        protected_modal = np.zeros((roi.size, roi.size), dtype=bool)

        # Track previous writes for deterministic member conflict resolution.
        filled_reconstruction = np.zeros_like(protected_modal)
        reconstruction_owner = np.full(
            protected_modal.shape, -1, dtype=np.int32
        )
        conflicts: list[tuple[str, str]] = []

        group_left, group_top, group_right, group_bottom = roi.box
        clipped_left, clipped_top, clipped_right, clipped_bottom = (
            roi.clipped_box
        )
        # Map the group's visible support into crop coordinates.
        protected_modal[
            clipped_top - group_top : clipped_bottom - group_top,
            clipped_left - group_left : clipped_right - group_left,
        ] = (
            group.modal_mask[
                clipped_top:clipped_bottom,
                clipped_left:clipped_right,
            ]
            > 0
        )

        for member_index, member in enumerate(group.members):
            canvas = member.reconstruction_canvas
            member_roi = member.reconstruction_roi
            reconstruction_mask = member.reconstruction_mask
            write_mask = (
                member.reconstruction_write_mask
                if member.reconstruction_write_mask is not None
                else reconstruction_mask
            )
            write_alpha = (
                member.reconstruction_write_alpha
                if member.reconstruction_write_alpha is not None
                else (
                    write_mask.astype(np.float64)
                    if write_mask is not None
                    else None
                )
            )
            if canvas is None and member_roi is None:
                log_event(
                    logger,
                    "group_composition",
                    "member_decision",
                    group_id=group.group_id,
                    object_id=member.object_id,
                    decision="use_original_rgb",
                    reason="member_not_reconstructed",
                )
                continue
            if (
                canvas is None
                or member_roi is None
                or reconstruction_mask is None
                or write_mask is None
                or write_alpha is None
            ):
                raise ValueError(
                    f"incomplete reconstruction record for {member.object_id}"
                )
            if member_roi.image_size != source.size:
                raise ValueError(
                    f"reconstruction ROI for {member.object_id} does not "
                    "match the source image"
                )
            if canvas.size != (member_roi.size, member_roi.size):
                raise ValueError(
                    f"reconstruction canvas for {member.object_id} does not "
                    "match its ROI"
                )
            if (
                reconstruction_mask.shape != (source_height, source_width)
                or write_mask.shape != (source_height, source_width)
                or write_alpha.shape != (source_height, source_width)
            ):
                raise ValueError(
                    f"reconstruction mask for {member.object_id} does not "
                    "match the source image"
                )

            # Intersect the group and member reconstruction ROIs.
            member_left, member_top, member_right, member_bottom = (
                member_roi.box
            )
            overlap_left = max(0, group_left, member_left)
            overlap_top = max(0, group_top, member_top)
            overlap_right = min(source_width, group_right, member_right)
            overlap_bottom = min(source_height, group_bottom, member_bottom)
            if (
                overlap_left >= overlap_right
                or overlap_top >= overlap_bottom
            ):
                continue

            # Map the intersection into group and member-canvas coordinates.
            destination_y = slice(
                overlap_top - group_top,
                overlap_bottom - group_top,
            )
            destination_x = slice(
                overlap_left - group_left,
                overlap_right - group_left,
            )
            canvas_y = slice(
                overlap_top - member_top,
                overlap_bottom - member_top,
            )
            canvas_x = slice(
                overlap_left - member_left,
                overlap_right - member_left,
            )
            # Restrict writes to pixels the model was allowed to reconstruct.
            permitted = write_mask[
                overlap_top:overlap_bottom,
                overlap_left:overlap_right,
            ].astype(bool)
            permitted_alpha = np.clip(
                write_alpha[
                    overlap_top:overlap_bottom,
                    overlap_left:overlap_right,
                ].astype(np.float64),
                0.0,
                1.0,
            )
            permitted &= permitted_alpha > 0.0
            # Preserve visible pixels and reconstruction from earlier members.
            writable = (
                permitted
                & ~protected_modal[destination_y, destination_x]
                & ~filled_reconstruction[destination_y, destination_x]
            )

            conflict = (
                permitted
                & ~protected_modal[destination_y, destination_x]
                & filled_reconstruction[destination_y, destination_x]
            )
            owner_crop = reconstruction_owner[destination_y, destination_x]
            for owner_index in np.unique(owner_crop[conflict]):
                conflict_record = (
                    group.members[int(owner_index)].object_id,
                    member.object_id,
                )
                if conflict_record not in conflicts:
                    conflicts.append(conflict_record)
                    log_event(
                        logger,
                        "group_composition",
                        "conflict_decision",
                        group_id=group.group_id,
                        winner_id=conflict_record[0],
                        loser_id=conflict_record[1],
                        decision="first_reconstruction_wins",
                    )
            # Overlay the reconstructed pixels onto the composite image
            destination = composed[destination_y, destination_x]
            candidate = np.asarray(canvas.convert("RGB"), dtype=np.uint8)[
                canvas_y, canvas_x
            ]
            effective_alpha = np.where(
                writable, permitted_alpha, 0.0
            )[..., None]
            blended = np.rint(
                candidate.astype(np.float64) * effective_alpha
                + destination.astype(np.float64) * (1.0 - effective_alpha)
            ).clip(0, 255).astype(np.uint8)
            destination[writable] = blended[writable]
            # Block later members from overwriting these accepted pixels.
            filled_reconstruction[destination_y, destination_x] |= writable
            owner_crop[writable] = member_index
            log_event(
                logger,
                "group_composition",
                "member_decision",
                group_id=group.group_id,
                object_id=member.object_id,
                decision="apply_reconstructed_rgb",
                written_pixels=int(np.count_nonzero(writable)),
                protected_modal_pixels=int(
                    np.count_nonzero(
                        permitted
                        & protected_modal[destination_y, destination_x]
                    )
                ),
                conflict_pixels=int(np.count_nonzero(conflict)),
            )

        group.composed_source = Image.fromarray(composed, mode="RGB")
        group.composed_roi = roi
        group.reconstruction_conflicts = tuple(conflicts)
        log_event(
            logger,
            "group_composition",
            "group_result",
            group_id=group.group_id,
            roi=(roi.x, roi.y, roi.size),
            has_reconstruction=group.has_reconstruction,
            reconstruction_conflicts=list(group.reconstruction_conflicts),
        )
