"""Local mask-region selection for SmartEraser inference."""

from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label


BoundingBox = tuple[int, int, int, int]
_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class MaskGroup:
    """Connected mask components that share local inference context."""

    mask: Image.Image
    bounding_box: BoundingBox

    def to_image(self, image_size: tuple[int, int]) -> Image.Image:
        """Materialize this compact group mask at full image resolution."""

        width, height = image_size
        left, top, right, bottom = self.bounding_box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(
                "SmartEraser group bounding_box must be inside the image"
            )
        if self.mask.size != (right - left, bottom - top):
            raise ValueError(
                "SmartEraser compact group mask does not match its bounds"
            )
        full_mask = Image.new("L", image_size, 0)
        full_mask.paste(self.mask, (left, top))
        return full_mask


def validate_context_settings(
    context_scale: float,
    minimum_context_ratio: float,
) -> None:
    """Validate adaptive-context configuration."""

    if not isfinite(context_scale) or context_scale < 1.0:
        raise ValueError("SmartEraser context_scale must be at least 1.0")
    if (
        not isfinite(minimum_context_ratio)
        or not 0.0 < minimum_context_ratio <= 1.0
    ):
        raise ValueError(
            "SmartEraser minimum_context_ratio must be in (0.0, 1.0]"
        )


def adaptive_context_box(
    bounding_box: BoundingBox,
    image_size: tuple[int, int],
    context_scale: float,
    minimum_context_ratio: float,
) -> BoundingBox | None:
    """Return a clamped square context box or request padding fallback."""

    validate_context_settings(context_scale, minimum_context_ratio)
    width, height = image_size
    left, top, right, bottom = bounding_box
    if width <= 0 or height <= 0:
        raise ValueError("SmartEraser image dimensions must be positive")
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            "SmartEraser bounding_box must be non-empty and inside the image"
        )

    box_width = right - left
    box_height = bottom - top
    short_side = min(width, height)
    if max(box_width, box_height) > short_side:
        return None

    crop_side = min(
        short_side,
        ceil(
            max(
                max(box_width, box_height) * context_scale,
                short_side * minimum_context_ratio,
            )
        ),
    )
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = min(
        max(round(center_x - crop_side / 2.0), 0),
        width - crop_side,
    )
    crop_top = min(
        max(round(center_y - crop_side / 2.0), 0),
        height - crop_side,
    )
    return (
        crop_left,
        crop_top,
        crop_left + crop_side,
        crop_top + crop_side,
    )


def _boxes_intersect_or_touch(
    first: BoundingBox,
    second: BoundingBox,
) -> bool:
    return not (
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )


def _component_boxes(
    labels: np.ndarray,
    component_count: int,
) -> list[tuple[int, BoundingBox]]:
    components: list[tuple[int, BoundingBox]] = []
    for component_id, slices in enumerate(
        find_objects(labels, max_label=component_count),
        start=1,
    ):
        if slices is None:
            continue
        row_slice, column_slice = slices
        components.append(
            (
                component_id,
                (
                    int(column_slice.start),
                    int(row_slice.start),
                    int(column_slice.stop),
                    int(row_slice.stop),
                ),
            )
        )
    return components


def _cells_for_box(
    box: BoundingBox,
    cell_size: int,
) -> tuple[tuple[int, int], ...]:
    """Return spatial-grid cells touched by a closed context box."""

    left, top, right, bottom = box
    return tuple(
        (cell_x, cell_y)
        for cell_y in range(top // cell_size, bottom // cell_size + 1)
        for cell_x in range(left // cell_size, right // cell_size + 1)
    )


def _group_intersecting_boxes(
    boxes: list[BoundingBox],
    image_size: tuple[int, int],
    minimum_context_ratio: float,
) -> dict[int, list[int]]:
    """Return rectangle-intersection groups using a bounded spatial grid."""

    parents = list(range(len(boxes)))
    component_sizes = [1] * len(boxes)
    root_cells: list[set[tuple[int, int]]] = [
        set() for _ in boxes
    ]
    grid: dict[
        tuple[int, int],
        dict[int, list[int]],
    ] = {}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> int:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return first_root
        if component_sizes[first_root] < component_sizes[second_root]:
            first_root, second_root = second_root, first_root

        parents[second_root] = first_root
        component_sizes[first_root] += component_sizes[second_root]
        for cell in root_cells[second_root]:
            entries = grid[cell]
            second_members = entries.pop(second_root)
            entries.setdefault(first_root, []).extend(second_members)
            root_cells[first_root].add(cell)
        root_cells[second_root].clear()
        return first_root

    short_side = min(image_size)
    minimum_side = max(1, ceil(short_side * minimum_context_ratio))
    cell_size = max(minimum_side, ceil(short_side / 32))

    for current, current_box in enumerate(boxes):
        cells = _cells_for_box(current_box, cell_size)
        candidates: dict[int, list[int]] = {}
        for cell in cells:
            for root, members in grid.get(cell, {}).items():
                candidates.setdefault(find(root), []).extend(members)

        for root, members in candidates.items():
            if any(
                _boxes_intersect_or_touch(boxes[other], current_box)
                for other in reversed(members)
            ):
                union(current, root)

        current_root = find(current)
        for cell in cells:
            entries = grid.setdefault(cell, {})
            entries.setdefault(current_root, []).append(current)
            root_cells[current_root].add(cell)

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(boxes)):
        members_by_root.setdefault(find(index), []).append(index)
    return members_by_root


def group_mask_components(
    mask: Image.Image,
    context_scale: float,
    minimum_context_ratio: float,
) -> list[MaskGroup]:
    """Group 8-connected components whose local context boxes overlap."""

    validate_context_settings(context_scale, minimum_context_ratio)
    binary = np.asarray(
        mask.convert("L").point(lambda value: 255 if value > 127 else 0),
        dtype=np.uint8,
    ) > 0
    if not np.any(binary):
        return []

    labels, component_count = label(binary, structure=_EIGHT_CONNECTED)
    components = _component_boxes(labels, component_count)
    width, height = mask.size
    full_image_box = (0, 0, width, height)
    context_boxes = [
        adaptive_context_box(
            bounding_box,
            mask.size,
            context_scale,
            minimum_context_ratio,
        )
        or full_image_box
        for _, bounding_box in components
    ]

    members_by_root = _group_intersecting_boxes(
        context_boxes,
        mask.size,
        minimum_context_ratio,
    )

    groups: list[MaskGroup] = []
    for members in members_by_root.values():
        member_ids = [components[index][0] for index in members]
        member_boxes = [components[index][1] for index in members]
        bounding_box = (
            min(box[0] for box in member_boxes),
            min(box[1] for box in member_boxes),
            max(box[2] for box in member_boxes),
            max(box[3] for box in member_boxes),
        )
        left, top, right, bottom = bounding_box
        local_labels = labels[top:bottom, left:right]
        local_group = np.isin(local_labels, member_ids)
        compact_mask = Image.fromarray(
            local_group.astype(np.uint8) * 255,
            mode="L",
        )
        groups.append(
            MaskGroup(
                mask=compact_mask,
                bounding_box=bounding_box,
            )
        )

    groups.sort(
        key=lambda group: (
            group.bounding_box[1],
            group.bounding_box[0],
            group.bounding_box[3],
            group.bounding_box[2],
        )
    )
    return groups
