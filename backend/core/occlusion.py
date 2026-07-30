"""Các hàm hỗ trợ hình học thuần túy để xác định đối tượng nào cần khôi phục (completion/reconstruction)."""

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Optional, Sequence


BoundingBox = tuple[int, int, int, int]
OverlapPair = tuple[str, str]


@dataclass(frozen=True)
class ObjectBounds:
    """Thông tin định danh, nhãn ngữ nghĩa (semantic class) và tọa độ khung viền ``(x, y, width, height)`` của một đối tượng."""

    object_id: str
    semantic_class: str
    bbox: BoundingBox


@dataclass(frozen=True)
class BBoxContainmentMetrics:
    """Symmetric containment metrics with stable smaller/larger identities."""

    intersection_area: int
    smaller_area: int
    larger_area: int
    smaller_index: int
    larger_index: int
    containment_ratio: float
    bbox_size_ratio: float


@dataclass(frozen=True)
class PairDecision:
    """Kết quả phân tích vai trò che khuất (occluded/occluder) và hướng tái tạo cho một cặp đối tượng giao nhau.

    Attributes:
        first_id: ID của đối tượng thứ nhất trong cặp giao nhau.
        second_id: ID của đối tượng thứ hai trong cặp giao nhau.
        occluded_id: ID của đối tượng đóng vai trò bị che khuất chính (hoặc None nếu hòa/không rõ ràng).
        occluder_id: ID của đối tượng đóng vai trò che khuất chính (hoặc None nếu hòa/không rõ ràng).
        reconstruction_directions: Danh sách các hướng tái tạo `((occluded, occluder), ...)`.
            Chứa các cặp (A, B) trong đó A bị B che khuất và cần được inpainting/vẽ bù đằng sau B.
        first_hidden_by_second_area: Diện tích (pixel) đối tượng 1 bị che khuất bởi đối tượng 2.
        second_hidden_by_first_area: Diện tích (pixel) đối tượng 2 bị che khuất bởi đối tượng 1.
    """

    first_id: str
    second_id: str
    # occluded_id và occluder_id chỉ có thể nhận giá trị của first_id hoặc second_id (hoặc None)
    occluded_id: Optional[str]
    occluder_id: Optional[str]
    reconstruction_directions: tuple[OverlapPair, ...] = ()
    first_hidden_by_second_area: int = 0
    second_hidden_by_first_area: int = 0

    @property
    def ambiguous(self) -> bool:
        """Trả về True nếu không xác định được vai trò ưu tiên rõ ràng (thứ tự che khuất bị hòa)."""
        return self.occluded_id is None

    @property
    def bidirectional(self) -> bool:
        """Trả về True nếu sự che khuất xảy ra theo cả 2 chiều (cả 2 đối tượng đều cần được tái tạo đằng sau nhau)."""
        return len(self.reconstruction_directions) == 2


def bbox_area(bbox: BoundingBox) -> int:
    """Return positive bbox area, or zero for invalid dimensions."""
    _, _, width, height = bbox
    if width <= 0 or height <= 0:
        return 0
    return int(width) * int(height)


def bbox_intersection_area(
    first: BoundingBox,
    second: BoundingBox,
) -> int:
    """Return positive intersection area for two ``(x, y, w, h)`` boxes."""
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    if bbox_area(first) == 0 or bbox_area(second) == 0:
        return 0

    width = min(
        first_x + first_width,
        second_x + second_width,
    ) - max(first_x, second_x)
    height = min(
        first_y + first_height,
        second_y + second_height,
    ) - max(first_y, second_y)
    if width <= 0 or height <= 0:
        return 0
    return int(width) * int(height)


def bbox_containment_metrics(
    first: BoundingBox,
    second: BoundingBox,
) -> Optional[BBoxContainmentMetrics]:
    """Return containment metrics, or ``None`` when either box is invalid."""
    areas = (bbox_area(first), bbox_area(second))
    if min(areas) <= 0:
        return None

    smaller_index, larger_index = (
        (0, 1) if areas[0] <= areas[1] else (1, 0)
    )
    smaller_area = areas[smaller_index]
    larger_area = areas[larger_index]
    intersection_area = bbox_intersection_area(first, second)
    return BBoxContainmentMetrics(
        intersection_area=intersection_area,
        smaller_area=smaller_area,
        larger_area=larger_area,
        smaller_index=smaller_index,
        larger_index=larger_index,
        containment_ratio=intersection_area / smaller_area,
        bbox_size_ratio=smaller_area / larger_area,
    )


def find_cross_class_overlaps(
    objects: Sequence[ObjectBounds],
) -> list[OverlapPair]:
    """Tìm và trả về danh sách các cặp đối tượng thuộc 2 nhãn ngữ nghĩa (semantic class) khác nhau có BoundingBox giao nhau.

    Args:
        objects: Danh sách các đối tượng kèm thông tin khung viền BoundingBox.

    Returns:
        Danh sách các cặp ID giao nhau (first_id, second_id).
    """
    overlaps: list[OverlapPair] = []

    # Duyệt qua tất cả các cặp đối tượng kết hợp độc lập
    for first, second in combinations(objects, 2):
        # Bỏ qua nếu hai đối tượng cùng loại (cùng semantic class)
        if first.semantic_class == second.semantic_class:
            continue

        if bbox_intersection_area(first.bbox, second.bbox) > 0:
            overlaps.append((first.object_id, second.object_id))

    return overlaps


def effective_hole_area(
    raw_area: int,
    modal_area: int,
    *,
    minimum_pixels: int,
    minimum_modal_ratio: float,
) -> int:
    """Lọc bỏ nhiễu của vùng bị che khuất (completion hole area), chỉ giữ lại diện tích thực sự có ý nghĩa.

    Hàm tính toán một ngưỡng nhiễu tối thiểu dựa trên cả số pixel tuyệt đối và tỷ lệ % diện tích đối tượng hiển thị (modal_area).
    Nếu diện tích thô (raw_area) vượt quá ngưỡng này thì được giữ nguyên, ngược lại sẽ triệt tiêu về 0 (coi là nhiễu ranh giới).

    Args:
        raw_area: Diện tích vùng bị che khuất thô (pixel).
        modal_area: Diện tích vùng đối tượng nhìn thấy được trên ảnh (pixel).
        minimum_pixels: Ngưỡng số pixel tối thiểu tuyệt đối.
        minimum_modal_ratio: Ngưỡng tỷ lệ tối thiểu so với modal_area.

    Returns:
        raw_area nếu vượt quá ngưỡng nhiễu, ngược lại trả về 0.
    """
    if minimum_pixels < 0 or minimum_modal_ratio < 0:
        raise ValueError("hole-area noise-floor settings must be non-negative")

    # Ngưỡng tối thiểu tương đối tính theo tỷ lệ diện tích nhìn thấy của vật thể
    relative_floor = modal_area * minimum_modal_ratio
    # Ngưỡng sàn ý nghĩa thực tế là giá trị lớn nhất giữa ngưỡng tuyệt đối và tương đối
    meaningful_floor = max(minimum_pixels, relative_floor)
    return raw_area if raw_area >= meaningful_floor else 0


def assign_pair_roles(
    pairs: Sequence[OverlapPair],
    hole_areas: Mapping[str, int],
    *,
    tie_tolerance_ratio: float = 0.0,
) -> list[PairDecision]:
    """Phân định vai trò che khuất (occluded/occluder) dựa trên diện tích vùng khuyết tổng thể của từng đối tượng.

    Args:
        pairs: Danh sách các cặp ID đối tượng giao nhau.
        hole_areas: Dictionary ánh xạ ID đối tượng -> diện tích vùng khuyết tổng thể.
        tie_tolerance_ratio: Tỷ lệ dung sai chênh lệch diện tích để coi là hòa (ambiguous).

    Returns:
        Danh sách đối tượng PairDecision chứa kết quả phân định cho từng cặp.
    """
    if tie_tolerance_ratio < 0:
        raise ValueError("tie_tolerance_ratio must be non-negative")

    decisions: list[PairDecision] = []

    for first_id, second_id in pairs:
        first_area = hole_areas[first_id]
        second_area = hole_areas[second_id]

        # Tính ngưỡng dung sai so sánh
        tie_tolerance = max(first_area, second_area) * tie_tolerance_ratio
        if abs(first_area - second_area) <= tie_tolerance:
            # Chênh lệch diện tích nằm trong khoảng dung sai -> Coi là hòa/không rõ ràng
            occluded_id = occluder_id = None
        elif first_area > second_area:
            # Đối tượng 1 bị che nhiều hơn -> là đối tượng bị che khuất chính
            occluded_id, occluder_id = first_id, second_id
        else:
            # Đối tượng 2 bị che nhiều hơn -> là đối tượng bị che khuất chính
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


def assign_directional_pair_roles(
    pairs: Sequence[OverlapPair],
    directional_hole_areas: Mapping[OverlapPair, int],
    *,
    tie_tolerance_ratio: float = 0.0,
) -> list[PairDecision]:
    """Phân định các hướng tái tạo (reconstruction_directions) và vai trò che khuất từ diện tích bị che 2 chiều độc lập.

    `(A, B)` trong `directional_hole_areas` nghĩa là vùng bị che của đối tượng A đè lên mask của B,
    vì vậy A cần được tái tạo/inpainting đằng sau B.

    Hàm giữ lại CẢ HAI HƯỚNG tái tạo nếu có hiện tượng che khuất đan xen 2 chiều (ví dụ: bàn tay quấn quanh cuốn sách),
    đồng thời vẫn so sánh diện tích để xác định 1 chiều hiển thị ưu tiên duy nhất (phục vụ việc xếp lớp Layer Back-to-Front).

    Args:
        pairs: Danh sách các cặp ID đối tượng giao nhau.
        directional_hole_areas: Dictionary ánh xạ `(A, B)` -> diện tích A bị che bởi B.
        tie_tolerance_ratio: Tỷ lệ dung sai chênh lệch diện tích để coi thứ tự hiển thị là hòa (ambiguous).

    Returns:
        Danh sách PairDecision chứa cả reconstruction_directions lẫn occluded_id/occluder_id.
    """
    if tie_tolerance_ratio < 0:
        raise ValueError("tie_tolerance_ratio must be non-negative")

    decisions: list[PairDecision] = []
    for first_id, second_id in pairs:
        # Lấy diện tích bị che ở cả 2 hướng độc lập
        first_area = int(directional_hole_areas[(first_id, second_id)])
        second_area = int(directional_hole_areas[(second_id, first_id)])

        # Ghi nhận tất cả các hướng tái tạo có diện tích bị che > 0
        directions: list[OverlapPair] = []
        if first_area > 0:
            directions.append((first_id, second_id))
        if second_area > 0:
            directions.append((second_id, first_id))

        # So sánh diện tích 2 chiều để chọn ra đối tượng bị che khuất chính (phục vụ xếp lớp hiển thị)
        tolerance = max(first_area, second_area) * tie_tolerance_ratio
        if not directions or abs(first_area - second_area) <= tolerance:
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
                reconstruction_directions=tuple(directions),
                first_hidden_by_second_area=first_area,
                second_hidden_by_first_area=second_area,
            )
        )
    return decisions

