"""Stable post-reconstruction grouping of same-class raw objects."""

from collections.abc import Sequence

import numpy as np

from ..core.helpers import _bbox_from_mask
from .types import DetectedObject, GroupedObject


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
            if (
                first.semantic_class == second.semantic_class
                and _boxes_overlap(
                    first.original_modal_bbox,
                    second.original_modal_bbox,
                )
            ):
                union(first_index, second_index)

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
        groups.append(
            GroupedObject(
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
        )
    return groups
