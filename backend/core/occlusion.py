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


def effective_hole_area(
    raw_area: int,
    modal_area: int,
    *,
    minimum_pixels: int,
    minimum_modal_ratio: float,
) -> int:
    """Suppress completion-hole noise while preserving meaningful raw area."""
    if minimum_pixels < 0 or minimum_modal_ratio < 0:
        raise ValueError("hole-area noise-floor settings must be non-negative")

    relative_floor = modal_area * minimum_modal_ratio
    meaningful_floor = max(minimum_pixels, relative_floor)
    return raw_area if raw_area >= meaningful_floor else 0


def assign_pair_roles(
    pairs: Sequence[OverlapPair],
    hole_areas: Mapping[str, int],
    *,
    tie_tolerance_ratio: float = 0.0,
) -> list[PairDecision]:
    """Assign roles independently using only each pair's hole areas."""
    if tie_tolerance_ratio < 0:
        raise ValueError("tie_tolerance_ratio must be non-negative")

    decisions: list[PairDecision] = []

    for first_id, second_id in pairs:
        first_area = hole_areas[first_id]
        second_area = hole_areas[second_id]

        tie_tolerance = max(first_area, second_area) * tie_tolerance_ratio
        if abs(first_area - second_area) <= tie_tolerance:
            occluded_id = occluder_id = None
        elif first_area > second_area:
            occluded_id, occluder_id = first_id, second_id
        else:
            occluded_id, occluder_id = second_id, first_id

        decisions.append(
            PairDecision(
                first_id=first_id,
                second_id=second_id,
                occluded_id=occluded_id,
                occluder_id=occluder_id,
            )
        )

    return decisions
