import logging
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CORE UTILITIES
# ═══════════════════════════════════════════════════════════════

def _bboxes_overlap(bbox_a: Tuple, bbox_b: Tuple) -> bool:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    return not (ax+aw <= bx or bx+bw <= ax or ay+ah <= by or by+bh <= ay)


def _get_interaction_zone(
    bbox_a: Tuple[int,int,int,int],
    bbox_b: Tuple[int,int,int,int],
    img_shape: Tuple[int,int],
) -> np.ndarray | None:
    """
    Vùng giao của 2 bbox, dilate để bắt cả biên.
    Dùng thay cho mask_a & mask_b vì modal mask.
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    img_h, img_w   = img_shape

    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax+aw, bx+bw)
    iy1 = min(ay+ah, by+bh)

    if ix0 >= ix1 or iy0 >= iy1:
        return None

    zone = np.zeros((img_h, img_w), dtype=bool)
    zone[iy0:iy1, ix0:ix1] = True

    zone = cv2.dilate(
        zone.astype(np.uint8) * 255,
        np.ones((11, 11), np.uint8)
    ).astype(bool)

    return zone


# ═══════════════════════════════════════════════════════════════
# SIGNAL 1 — BBox Boundary Recession
# ═══════════════════════════════════════════════════════════════

def _s_bbox_recession(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    bbox_a: Tuple,
    bbox_b: Tuple,
    interaction_zone: np.ndarray,
) -> float:
    """
    Tại cạnh bbox giáp interaction zone:
    mask lõm vào (không đầy viền) → bị che → nằm dưới.
    Score > 0.5 → A trên.
    """
    def _recession(mask, bbox):
        x, y, w, h  = bbox
        img_h, img_w = mask.shape
        strip = max(3, min(8, int(min(w, h) * 0.04)))

        total_recession = 0.0
        n_checked       = 0

        edges = {
            'top':    (slice(max(0,y),         min(img_h,y+strip)),
                       slice(max(0,x),         min(img_w,x+w))),
            'bottom': (slice(max(0,y+h-strip), min(img_h,y+h)),
                       slice(max(0,x),         min(img_w,x+w))),
            'left':   (slice(max(0,y),         min(img_h,y+h)),
                       slice(max(0,x),         min(img_w,x+strip))),
            'right':  (slice(max(0,y),         min(img_h,y+h)),
                       slice(max(0,x+w-strip), min(img_w,x+w))),
        }

        for rs, cs in edges.values():
            edge_region          = np.zeros_like(mask)
            edge_region[rs, cs]  = True

            if not np.any(edge_region & interaction_zone):
                continue

            total  = edge_region.sum()
            filled = (edge_region & mask).sum()
            total_recession += 1.0 - filled / (total + 1e-6)
            n_checked += 1

        return total_recession / n_checked if n_checked > 0 else 0.0

    rec_a = _recession(mask_a, bbox_a)
    rec_b = _recession(mask_b, bbox_b)
    total = rec_a + rec_b

    if total < 1e-6:
        return 0.5

    # rec_b cao → B lõm → B dưới → A trên → score cao
    return rec_b / total


# ═══════════════════════════════════════════════════════════════
# SIGNAL 2 — Contour Curvature
# ═══════════════════════════════════════════════════════════════

def _s_contour_curvature(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    interaction_zone: np.ndarray,
) -> float:
    """
    Contour bị bẻ vào (concave) tại interaction zone → nằm dưới.
    Dùng cross product để đo sign của curvature.
    Score > 0.5 → A trên.
    """
    def _mean_curvature(mask):
        mask_u8    = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return 0.0

        cnt = max(contours, key=cv2.contourArea).squeeze()
        if cnt.ndim < 2 or len(cnt) < 15:
            return 0.0

        window     = max(3, len(cnt) // 50)
        curvatures = []

        for i in range(window, len(cnt) - window):
            px, py = int(cnt[i][0]), int(cnt[i][1])
            if (py < 0 or py >= interaction_zone.shape[0] or
                    px < 0 or px >= interaction_zone.shape[1]):
                continue
            if not interaction_zone[py, px]:
                continue

            v1    = cnt[i]        - cnt[i - window]
            v2    = cnt[i+window] - cnt[i]
            cross = float(v1[0]*v2[1] - v1[1]*v2[0])
            norm  = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            curvatures.append(cross / norm)

        return float(np.mean(curvatures)) if curvatures else 0.0

    curv_a = _mean_curvature(mask_a)
    curv_b = _mean_curvature(mask_b)

    # curv_b âm → B concave → B dưới → A trên
    diff = curv_a - curv_b
    return float(np.clip(0.5 + diff / (abs(diff)*2 + 1e-6) * 0.5, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# SIGNAL 3 — T-Junction
# ═══════════════════════════════════════════════════════════════

def _s_t_junction(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    interaction_zone: np.ndarray,
) -> float:
    """
    Contour B kết thúc (endpoint) tại interaction zone → B dưới.
    Contour A đi xuyên qua liên tục → A trên.
    Score > 0.5 → A trên.
    """
    def _analyze(mask):
        mask_u8    = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return 0, 0

        cimg = np.zeros_like(mask_u8)
        cv2.drawContours(cimg, contours, -1, 255, thickness=1)

        in_zone = (cimg > 0) & interaction_zone
        if not np.any(in_zone):
            return 0, 0

        k  = np.ones((3, 3), np.float32)
        nc = cv2.filter2D(in_zone.astype(np.float32), -1, k)

        endpoints = int(np.sum(in_zone & (nc <= 2)))
        total_len = int(np.sum(in_zone))
        return endpoints, total_len

    ep_a, len_a = _analyze(mask_a)
    ep_b, len_b = _analyze(mask_b)

    ep_total  = ep_a  + ep_b  + 1e-6
    len_total = len_a + len_b + 1e-6

    # B endpoint nhiều → B kết thúc → B bị che → B dưới → A trên
    endpoint_score    = ep_b  / ep_total
    passthrough_score = len_a / len_total

    return 0.6 * endpoint_score + 0.4 * passthrough_score


# ═══════════════════════════════════════════════════════════════
# SIGNAL 4 — Convexity Deficit
# ═══════════════════════════════════════════════════════════════

def _s_convexity_deficit(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    interaction_zone: np.ndarray,
) -> float:
    """
    Hull bị cắt nhiều tại interaction zone → nằm dưới.
    Score > 0.5 → A trên.
    """
    def _deficit(mask):
        mask_u8    = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0

        hull_img = np.zeros_like(mask_u8)
        for cnt in contours:
            cv2.fillPoly(hull_img, [cv2.convexHull(cnt)], 255)

        deficit = (hull_img > 0) & (~mask) & interaction_zone
        return deficit.sum() / (interaction_zone.sum() + 1e-6)

    d_a = _deficit(mask_a)
    d_b = _deficit(mask_b)
    total = d_a + d_b

    if total < 1e-6:
        return 0.5

    return d_b / total


# ═══════════════════════════════════════════════════════════════
# SIGNAL 5 — Local Solidity
# ═══════════════════════════════════════════════════════════════

def _s_local_solidity(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    interaction_zone: np.ndarray,
) -> float:
    """
    Solidity cao trong ROI → hình dạng đầy đặn → nằm trên.
    Score > 0.5 → A trên.
    """
    def _solidity(mask):
        local      = (mask & interaction_zone).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0

        area      = sum(cv2.contourArea(c) for c in contours)
        hull_area = sum(
            cv2.contourArea(cv2.convexHull(c))
            for c in contours if len(c) >= 3
        )
        return area / (hull_area + 1e-6)

    sol_a = _solidity(mask_a)
    sol_b = _solidity(mask_b)
    total = sol_a + sol_b

    return sol_a / total if total > 1e-6 else 0.5


# ═══════════════════════════════════════════════════════════════
# SIGNAL 6 — Contour Length
# ═══════════════════════════════════════════════════════════════

def _s_contour_length(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    interaction_zone: np.ndarray,
) -> float:
    """
    Contour dài hơn trong zone → owns biên → nằm trên.
    Score > 0.5 → A trên.
    """
    def _length(mask):
        mask_u8    = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return 0
        cimg = np.zeros_like(mask_u8)
        cv2.drawContours(cimg, contours, -1, 255, thickness=1)
        return int(np.sum((cimg > 0) & interaction_zone))

    l_a   = _length(mask_a)
    l_b   = _length(mask_b)
    total = l_a + l_b

    return l_a / total if total > 1e-6 else 0.5


# ═══════════════════════════════════════════════════════════════
# SIGNAL 7 — Edge Gradient
# ═══════════════════════════════════════════════════════════════

def _s_edge_gradient(
    image_np: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    interaction_zone: np.ndarray,
) -> float:
    """
    Gradient mạnh tại biên trong zone → owns biên → nằm trên.
    Artwork thường có outline sắc nét.
    Score > 0.5 → A trên.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.sqrt(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0)**2 +
        cv2.Sobel(gray, cv2.CV_32F, 0, 1)**2
    )

    def _boundary_grad(mask):
        mask_u8    = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return 0.0
        cimg = np.zeros_like(mask_u8)
        cv2.drawContours(cimg, contours, -1, 255, thickness=1)
        b_in_zone = (cimg > 0) & interaction_zone
        if not np.any(b_in_zone):
            return 0.0
        return float(grad[b_in_zone].mean())

    g_a   = _boundary_grad(mask_a)
    g_b   = _boundary_grad(mask_b)
    total = g_a + g_b

    return g_a / total if total > 1e-6 else 0.5


# ═══════════════════════════════════════════════════════════════
# FALLBACKS
# ═══════════════════════════════════════════════════════════════

def _fallback_fill_ratio(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    bbox_a: Tuple,
    bbox_b: Tuple,
) -> Tuple[str, float]:
    """
    Chia interaction zone thành 4 quadrant.
    Quadrant nào mask điền đầy hơn → vật đó trên ở quadrant đó.
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    ix0   = max(ax, bx);  iy0 = max(ay, by)
    ix1   = min(ax+aw, bx+bw);  iy1 = min(ay+ah, by+bh)
    mid_x = (ix0 + ix1) // 2
    mid_y = (iy0 + iy1) // 2

    quadrants = [
        (slice(iy0, mid_y), slice(ix0, mid_x)),
        (slice(iy0, mid_y), slice(mid_x, ix1)),
        (slice(mid_y, iy1), slice(ix0, mid_x)),
        (slice(mid_y, iy1), slice(mid_x, ix1)),
    ]

    votes_a = sum(
        1 for rs, cs in quadrants
        if mask_a[rs, cs].sum() > mask_b[rs, cs].sum()
    )
    conf = abs(votes_a - 2) / 2.0   # 0 = tie, 1 = 4-0

    return ('A_on_top' if votes_a >= 2 else 'B_on_top'), conf * 0.2


def _fallback_vertical_position(
    bbox_a: Tuple,
    bbox_b: Tuple,
) -> Tuple[str, float]:
    """
    Bottom thấp hơn trong frame → foreground → trên.
    Đúng với ~70% trường hợp người đứng trong artwork.
    """
    bottom_a = bbox_a[1] + bbox_a[3]
    bottom_b = bbox_b[1] + bbox_b[3]

    if bottom_a == bottom_b:
        return 'uncertain', 0.0

    diff = abs(bottom_a - bottom_b)
    conf = min(0.15, diff / 100.0)   # confidence thấp, chỉ là heuristic

    return ('A_on_top' if bottom_a > bottom_b else 'B_on_top'), conf


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def determine_occlusion_order(
    mask_a:   np.ndarray,
    mask_b:   np.ndarray,
    bbox_a:   Tuple[int,int,int,int],
    bbox_b:   Tuple[int,int,int,int],
    image_np: np.ndarray,
    confidence_threshold: float = 0.25,
) -> Tuple[str, float, dict]:
    """
    Xác định vật thể nào đang đè lên vật thể nào.

    Args:
        mask_a:   bool (H,W) — mask của object A trên ảnh gốc
        mask_b:   bool (H,W) — mask của object B trên ảnh gốc
        bbox_a:   (x, y, w, h) — bbox của A, cắt sát mask
        bbox_b:   (x, y, w, h) — bbox của B, cắt sát mask
        image_np: RGB uint8 (H,W,3)
        confidence_threshold: ngưỡng confidence tối thiểu

    Returns:
        result:     'A_on_top' | 'B_on_top' | 'uncertain'
        confidence: float [0, 1]
        details:    dict chứa score từng signal để debug
    """
    # ── Kiểm tra bbox có giao nhau không ──────────────────────
    if not _bboxes_overlap(bbox_a, bbox_b):
        return 'no_overlap', 1.0, {}

    # ── Tính interaction zone ─────────────────────────────────
    interaction_zone = _get_interaction_zone(
        bbox_a, bbox_b, mask_a.shape
    )
    if interaction_zone is None:
        return 'no_overlap', 1.0, {}

    # ── Tính 7 signals ────────────────────────────────────────
    scores = {
        'bbox_recession': _s_bbox_recession(
            mask_a, mask_b, bbox_a, bbox_b, interaction_zone),
        'curvature':      _s_contour_curvature(
            mask_a, mask_b, interaction_zone),
        't_junction':     _s_t_junction(
            mask_a, mask_b, interaction_zone),
        'convexity':      _s_convexity_deficit(
            mask_a, mask_b, interaction_zone),
        'solidity':       _s_local_solidity(
            mask_a, mask_b, interaction_zone),
        'contour_length': _s_contour_length(
            mask_a, mask_b, interaction_zone),
        'gradient':       _s_edge_gradient(
            image_np, mask_a, mask_b, interaction_zone),
    }

    weights = {
        'bbox_recession': 0.25,
        'curvature':      0.20,
        't_junction':     0.20,
        'convexity':      0.15,
        'solidity':       0.08,
        'contour_length': 0.07,
        'gradient':       0.05,
    }

    final      = sum(scores[k] * weights[k] for k in scores)
    confidence = abs(final - 0.5) * 2

    details = {
        'scores':     scores,
        'weights':    weights,
        'final':      round(final, 4),
        'confidence': round(confidence, 4),
    }

    logger.debug(
        "[Occlusion] " +
        " | ".join(f"{k}={v:.3f}" for k, v in scores.items()) +
        f" → final={final:.3f} conf={confidence:.3f}"
    )

    # ── Cascade fallback khi confidence thấp ─────────────────
    if confidence >= confidence_threshold:
        result = 'A_on_top' if final > 0.5 else 'B_on_top'
        details['method'] = 'ensemble'
        return result, confidence, details

    # Fallback 1: fill ratio
    result_fb1, conf_fb1 = _fallback_fill_ratio(mask_a, mask_b, bbox_a, bbox_b)
    if conf_fb1 > 0.05:
        details['method'] = 'fallback_fill_ratio'
        logger.warning(
            f"[Occlusion] Low confidence {confidence:.2f} "
            f"→ fallback fill_ratio → {result_fb1}"
        )
        return result_fb1, conf_fb1, details

    # Fallback 2: vertical position
    result_fb2, conf_fb2 = _fallback_vertical_position(bbox_a, bbox_b)
    if conf_fb2 > 0.0:
        details['method'] = 'fallback_vertical'
        logger.warning(
            f"[Occlusion] → fallback vertical → {result_fb2}"
        )
        return result_fb2, conf_fb2, details

    # Không xác định được
    details['method'] = 'uncertain'
    return 'uncertain', 0.0, details


# ═══════════════════════════════════════════════════════════════
# BUILD OCCLUSION GRAPH (dùng trong pipeline)
# ═══════════════════════════════════════════════════════════════

def build_occlusion_graph(
    visible_masks: List[np.ndarray],
    bboxes:        List[Tuple],
    image_np:      np.ndarray,
    confidence_threshold: float = 0.25,
) -> Tuple[List[int], dict]:
    """
    Xây dựng đồ thị occlusion cho toàn bộ objects.

    Returns:
        depth_order:  list indices từ trên xuống dưới
        occluder_map: dict {target_idx: [occluder_idx, ...]}
    """
    n = len(visible_masks)
    occlusion_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            if not _bboxes_overlap(bboxes[i], bboxes[j]):
                continue

            result, conf, details = determine_occlusion_order(
                mask_a   = visible_masks[i],
                mask_b   = visible_masks[j],
                bbox_a   = bboxes[i],
                bbox_b   = bboxes[j],
                image_np = image_np,
                confidence_threshold = confidence_threshold,
            )

            logger.info(
                f"[Graph] ({i},{j}) → {result} "
                f"conf={conf:.2f} method={details.get('method','?')}"
            )

            if result == 'A_on_top':
                occlusion_matrix[i][j] = conf
            elif result == 'B_on_top':
                occlusion_matrix[j][i] = conf
            # uncertain → bỏ qua, không xử lý cặp này

    # Topological sort: score thấp = ít bị đè = trên
    being_occluded = occlusion_matrix.sum(axis=0)
    depth_order    = list(np.argsort(being_occluded))

    # occluder_map[j] = list các i đang đè lên j
    occluder_map = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if occlusion_matrix[i][j] > 0:
                occluder_map[j].append(i)

    return depth_order, dict(occluder_map)