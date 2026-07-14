"""Pure geometry helpers for deciding which objects need completion."""

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence


BoundingBox = tuple[int, int, int, int]
OverlapPair = tuple[str, str]


@dataclass(frozen=True)
class ObjectBounds:
    """Identity, semantic class, and ``(x, y, width, height)`` bounds."""

    object_id: str
    semantic_class: str
    bbox: BoundingBox


def find_cross_class_overlaps(
    objects: Sequence[ObjectBounds],
) -> list[OverlapPair]:
    """Return input-ordered pairs whose boxes overlap with positive area."""
    overlaps: list[OverlapPair] = []

    for first, second in combinations(objects, 2):
        if first.semantic_class == second.semantic_class:
            continue

        first_x, first_y, first_width, first_height = first.bbox
        second_x, second_y, second_width, second_height = second.bbox
        intersection_width = min(
            first_x + first_width, second_x + second_width
        ) - max(first_x, second_x)
        intersection_height = min(
            first_y + first_height, second_y + second_height
        ) - max(first_y, second_y)

        if intersection_width > 0 and intersection_height > 0:
            overlaps.append((first.object_id, second.object_id))

    return overlaps
