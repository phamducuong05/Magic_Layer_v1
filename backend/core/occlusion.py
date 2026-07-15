"""Pure geometry helpers for deciding which objects need completion."""

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Optional, Sequence


BoundingBox = tuple[int, int, int, int]
OverlapPair = tuple[str, str]


@dataclass(frozen=True)
class ObjectBounds:
    """Identity, semantic class, and ``(x, y, width, height)`` bounds."""

    object_id: str
    semantic_class: str
    bbox: BoundingBox


@dataclass(frozen=True)
class PairDecision:
    """Occluded/occluder roles for one overlap pair."""

    first_id: str
    second_id: str
    # occluded_id and occluder_id can only have value of first_id or second_id
    occluded_id: Optional[str]
    occluder_id: Optional[str]

    @property
    def ambiguous(self) -> bool:
        return self.occluded_id is None


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


def assign_pair_roles(
    pairs: Sequence[OverlapPair], hole_areas: Mapping[str, int]
) -> list[PairDecision]:
    """Assign roles independently using only each pair's hole areas."""
    decisions: list[PairDecision] = []

    for first_id, second_id in pairs:
        first_area = hole_areas[first_id]
        second_area = hole_areas[second_id]

        if first_area > second_area:
            occluded_id, occluder_id = first_id, second_id
        elif second_area > first_area:
            occluded_id, occluder_id = second_id, first_id
        else:
            occluded_id = occluder_id = None

        decisions.append(
            PairDecision(
                first_id=first_id,
                second_id=second_id,
                occluded_id=occluded_id,
                occluder_id=occluder_id,
            )
        )

    return decisions
