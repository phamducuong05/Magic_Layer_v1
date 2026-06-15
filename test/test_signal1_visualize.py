"""
Test & Visualize Signal 1 — BBox Boundary Recession (_s_bbox_recession)

Chạy test mặc định (tự tạo mask):
    python backend/test_signal1_visualize.py

Chạy với 2 ảnh mask của bạn:
    python backend/test_signal1_visualize.py --mask_a path/to/mask_a.png --mask_b path/to/mask_b.png

Lưu ý:
    - Ảnh mask là ảnh grayscale/binary (đen=background, trắng=mask).
    - Hàm sẽ tự tính bounding box từ mask (contour detection).
    - Nếu không truyền mask → dùng mask giả lập mặc định.
"""

import sys
import os
import argparse

import numpy as np

# ── Thêm project root vào path ───────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Nếu có matplotlib → plot, nếu không → dùng OpenCV/text ──
try:
    import matplotlib
    matplotlib.use("Agg")   # không cần display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from backend.occlusion_order import (
    _get_interaction_zone,
    _s_bbox_recession,
    determine_occlusion_order,
)


# ═══════════════════════════════════════════════════════════════
# 1. TẠO MASK GIẢ LẬP
# ═══════════════════════════════════════════════════════════════

def create_test_masks(img_h=300, img_w=400):
    """
    Case 1: A trên, B dưới.
    A = hình chữ nhật đầy đủ.
    B = hình chữ nhật bị cắt góc trên-phải (bị A che).
    """
    mask_a = np.zeros((img_h, img_w), dtype=bool)
    mask_b = np.zeros((img_h, img_w), dtype=bool)

    ax, ay, aw, ah = 130, 60, 120, 80
    mask_a[ay:ay+ah, ax:ax+aw] = True

    bx, by, bw, bh = 100, 120, 160, 110
    mask_b[by:by+bh, bx:bx+bw] = True
    cut_x_start = 180
    cut_y_end   = 155
    mask_b[by:cut_y_end, cut_x_start:bx+bw] = False

    bbox_a = (ax, ay, aw, ah)
    bbox_b = (bx, by, bw, bh)

    image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for c in range(3):
        image[:, :, c] = np.linspace(50, 200, img_w, dtype=np.uint8)
    yy, xx = np.mgrid[0:img_h, 0:img_w]
    image[:, :, 0] = (image[:, :, 0].astype(int) + xx // 4).clip(0, 255).astype(np.uint8)
    image[:, :, 1] = (image[:, :, 1].astype(int) + yy // 4).clip(0, 255).astype(np.uint8)

    return mask_a, mask_b, bbox_a, bbox_b, image


def create_test_case_2(img_h=300, img_w=400):
    """
    Case 2: B trên, A dưới (ngược lại case 1).
    B = hình chữ nhật đầy đủ (nhỏ, trên).
    A = hình chữ nhật lớn (dưới), bị cắt góc trên-trái.
    """
    mask_a = np.zeros((img_h, img_w), dtype=bool)
    mask_b = np.zeros((img_h, img_w), dtype=bool)

    ax, ay, aw, ah = 80, 140, 200, 120
    mask_a[ay:ay+ah, ax:ax+aw] = True
    mask_a[ay:ay+30, ax:ax+60] = False

    bx, by, bw, bh = 120, 80, 100, 80
    mask_b[by:by+bh, bx:bx+bw] = True

    bbox_a = (ax, ay, aw, ah)
    bbox_b = (bx, by, bw, bh)

    image = np.random.randint(80, 200, (img_h, img_w, 3), dtype=np.uint8)
    return mask_a, mask_b, bbox_a, bbox_b, image


# ═══════════════════════════════════════════════════════════════
# 1b. ĐỌC MASK TỪ ẢNH FILE
# ═══════════════════════════════════════════════════════════════

def load_masks_from_images(mask_a_path=None, mask_b_path=None):
    """
    Đọc 2 ảnh mask từ file路径 truyền vào.
    Ảnh mask: grayscale/binary — trắng (255) = mask, đen (0) = background.

    Nếu mask_a_path hoặc mask_b_path là None → dùng mask giả lập mặc định.

    Trả về:
        (mask_a, mask_b, bbox_a, bbox_b, image, source_description)
        hoặc None nếu cả 2 đều None.
    """
    # ── Nếu không có path nào → trả về None để caller tự fallback ──
    if mask_a_path is None and mask_b_path is None:
        return None

    if not HAS_CV2:
        raise RuntimeError("Cần OpenCV (cv2) để đọc ảnh mask từ file.")

    def _read_mask(path, label):
        """Đọc ảnh mask và tính bbox từ contour."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file mask {label}: {path}")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Không đọc được ảnh: {path}")

        # Binarize: pixel > 127 → mask
        mask_bool = img > 127

        # Tính bbox từ contour
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError(f"Không tìm thấy contour trong mask {label}: {path}")

        # Lấy bbox bao quanh tất cả contours
        all_pts = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        bbox = (x, y, w, h)

        print(f"  [Load] Mask {label}: shape={mask_bool.shape}, "
              f"pixels={mask_bool.sum()}, bbox={bbox}, file={path}")
        return mask_bool, bbox

    # Đọc mask A
    if mask_a_path is not None:
        mask_a, bbox_a = _read_mask(mask_a_path, "A")
    else:
        mask_a, bbox_a = None, None

    # Đọc mask B
    if mask_b_path is not None:
        mask_b, bbox_b = _read_mask(mask_b_path, "B")
    else:
        mask_b, bbox_b = None, None

    # Xác định kích thước ảnh
    if mask_a is not None:
        img_h, img_w = mask_a.shape
    else:
        img_h, img_w = mask_b.shape

    # Nếu chỉ có 1 mask → tự tạo mask còn lại (giả lập)
    if mask_a is None:
        print(f"  [Warn] Không có mask A → tạo mask giả lập")
        mask_a = np.zeros((img_h, img_w), dtype=bool)
        # Tự tạo A nằm phía trên B
        if bbox_b is not None:
            bx, by, bw, bh = bbox_b
            ax = max(0, bx + bw // 4)
            ay = max(0, by - bh // 2)
            aw = bw
            ah = bh
        else:
            ax, ay, aw, ah = 50, 50, 100, 80
        mask_a[ay:ay+ah, ax:ax+aw] = True
        bbox_a = (ax, ay, aw, ah)

    if mask_b is None:
        print(f"  [Warn] Không có mask B → tạo mask giả lập")
        mask_b = np.zeros((img_h, img_w), dtype=bool)
        if bbox_a is not None:
            ax, ay, aw, ah = bbox_a
            bx = max(0, ax + aw // 4)
            by = max(0, ay + ah // 3)
            bw = aw
            bh = ah
        else:
            bx, by, bw, bh = 100, 100, 120, 100
        mask_b[by:by+bh, bx:bx+bw] = True
        bbox_b = (bx, by, bw, bh)

    # Kiểm tra size khớp
    assert mask_a.shape == mask_b.shape, (
        f"Kích thước mask không khớp: A={mask_a.shape} vs B={mask_b.shape}"
    )

    # Tạo ảnh RGB giả làm input cho ensemble
    image = np.random.randint(80, 200, (img_h, img_w, 3), dtype=np.uint8)

    source = f"A={'custom' if mask_a_path else 'synthetic'}, " \
             f"B={'custom' if mask_b_path else 'synthetic'}"
    return mask_a, mask_b, bbox_a, bbox_b, image, source


# ═══════════════════════════════════════════════════════════════
# 2. TÍNH RECESSION CHI TIẾT (instrumented version)
# ═══════════════════════════════════════════════════════════════

def compute_recession_detail(mask, bbox, interaction_zone, label="X"):
    """
    Phiên bản chi tiết của _recession() bên trong _s_bbox_recession.
    Trả về recession từng cạnh + toàn bộ thông tin debug.

    Lưu ý: chỉ đếm filled/empty trong PHẦN strip giao với interaction zone.
    """
    x, y, w, h = bbox
    img_h, img_w = mask.shape

    strip = max(4, min(10, int(min(w, h) * 0.06)))

    edges = {
        'top':    (slice(max(0,y),         min(img_h, y+strip)),
                   slice(max(0,x),         min(img_w, x+w))),
        'bottom': (slice(max(0,y+h-strip), min(img_h, y+h)),
                   slice(max(0,x),         min(img_w, x+w))),
        'left':   (slice(max(0,y),         min(img_h, y+h)),
                   slice(max(0,x),         min(img_w, x+strip))),
        'right':  (slice(max(0,y),         min(img_h, y+h)),
                   slice(max(0,x+w-strip), min(img_w, x+w))),
    }

    edge_results = {}
    total_recession = 0.0
    n_checked = 0

    for edge_name, (rs, cs) in edges.items():
        edge_region = np.zeros_like(mask)
        edge_region[rs, cs] = True

        zone_overlap = edge_region & interaction_zone
        has_overlap = np.any(zone_overlap)

        if not has_overlap:
            edge_results[edge_name] = {
                'skip': True,
                'reason': 'No overlap with interaction zone',
                'strip_slices': (rs, cs),
            }
            continue

        # ── CHỈ ĐẾM trong vùng giao strip & zone ──
        band = edge_region & interaction_zone
        total = int(band.sum())
        filled = int((band & mask).sum())
        empty = total - filled
        recession = 1.0 - filled / (total + 1e-6)

        edge_results[edge_name] = {
            'skip': False,
            'total': total,
            'filled': filled,
            'empty': empty,
            'recession': recession,
            'fill_ratio': filled / (total + 1e-6),
            'strip_slices': (rs, cs),
            'strip_thickness': (
                rs.stop - rs.start if rs.start != rs.stop else cs.stop - cs.start
            ),
        }

        total_recession += recession
        n_checked += 1

    avg_recession = total_recession / n_checked if n_checked > 0 else 0.0

    return {
        'label': label,
        'bbox': bbox,
        'strip_width': strip,
        'edges': edge_results,
        'total_recession': total_recession,
        'n_checked': n_checked,
        'avg_recession': avg_recession,
    }


# ═══════════════════════════════════════════════════════════════
# 3. CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════

def print_recession_report(detail_a, detail_b, score):
    """In báo cáo chi tiết recession ra console."""
    W = 70
    sep = "═" * W
    subsep = "─" * W

    print(f"\n{sep}")
    print("  SIGNAL 1 — BBox Boundary Recession  |  Báo cáo chi tiết")
    print(sep)

    for detail in [detail_a, detail_b]:
        label = detail['label']
        bbox = detail['bbox']
        print(f"\n  ▌ Vật thể {label}  |  bbox = {bbox}"
              f"  |  strip_width = {detail['strip_width']}px")
        print(f"  {subsep}")

        for edge_name, info in detail['edges'].items():
            if info.get('skip', False):
                print(f"    {edge_name:>8s}: SKIP — {info['reason']}")
            else:
                bar_len = 30
                filled_len = int(info['fill_ratio'] * bar_len)
                empty_len = bar_len - filled_len
                bar = "█" * filled_len + "░" * empty_len

                print(f"    {edge_name:>8s}: "
                      f"filled {info['filled']:>4d}/{info['total']:>4d}  "
                      f"[{bar}]  "
                      f"recession={info['recession']:.3f}")

        print(f"    {'':>8s}  ──────────────────────────────────")
        print(f"    {'RESULT':>8s}: avg_recession = {detail['avg_recession']:.4f}"
              f"  ({detail['n_checked']} edges checked)")

    print(f"\n{sep}")
    print(f"  TÍNH TOÁN SIGNAL:")
    print(f"    rec_A = {detail_a['avg_recession']:.4f}")
    print(f"    rec_B = {detail_b['avg_recession']:.4f}")
    print(f"    total = rec_A + rec_B = {detail_a['avg_recession'] + detail_b['avg_recession']:.4f}")
    print(f"    score = rec_B / total = {detail_b['avg_recession']:.4f} / "
          f"{detail_a['avg_recession'] + detail_b['avg_recession']:.4f}"
          f"  = {score:.4f}")
    print(f"")
    if score > 0.5:
        prediction = "A TRÊN (A_on_top)"
        reason = f"score {score:.4f} > 0.5  →  B lõm nhiều hơn → B dưới"
    elif score < 0.5:
        prediction = "B TRÊN (B_on_top)"
        reason = f"score {score:.4f} < 0.5  →  A lõm nhiều hơn → A dưới"
    else:
        prediction = "KHÔNG XÁC ĐỊNH"
        reason = "score = 0.5 → cả hai đều lõm như nhau"
    print(f"    ➤ Dự đoán: {prediction}")
    print(f"    ➤ Lý do:   {reason}")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ═══════════════════════════════════════════════════════════════

# Bảng màu
COLOR_A = np.array([30, 120, 255], dtype=np.float64)    # xanh dương
COLOR_B = np.array([255, 80, 80], dtype=np.float64)     # đỏ
COLOR_ZONE = np.array([255, 255, 0], dtype=np.float64)  # vàng

# Màu cho từng cạnh strip — FILLED (pixel strip trong zone chạm mask)
EDGE_COLORS_FILLED = {
    'top':    np.array([0, 220, 0], dtype=np.float64),      # xanh lá đậm
    'bottom': np.array([0, 220, 220], dtype=np.float64),    # cyan đậm
    'left':   np.array([255, 180, 0], dtype=np.float64),    # cam đậm
    'right':  np.array([200, 0, 255], dtype=np.float64),    # tím đậm
}

# Màu cho pixel strip trong zone KHÔNG chạm mask (empty / recession)
EDGE_COLORS_EMPTY = {
    'top':    np.array([180, 255, 180], dtype=np.float64),  # xanh lá rất nhạt
    'bottom': np.array([180, 255, 255], dtype=np.float64),  # cyan rất nhạt
    'left':   np.array([255, 240, 180], dtype=np.float64),  # kem nhạt
    'right':  np.array([230, 180, 255], dtype=np.float64),  # tím nhạt
}

# Màu cho phần strip NGOÀI interaction zone (không dùng trong tính toán)
COLOR_STRIP_OUTSIDE_ZONE = np.array([60, 60, 60], dtype=np.float64)  # xám đậm


def _draw_bbox(ax, bbox, color, label, linestyle="--"):
    """Vẽ bounding box lên axes."""
    rect = mpatches.Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3],
        linewidth=2.5, edgecolor=color / 255.0,
        facecolor="none", linestyle=linestyle
    )
    ax.add_patch(rect)
    ax.text(bbox[0] + 3, bbox[1] - 6, label,
            color=color / 255.0, fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))


def visualize_case(mask_a, mask_b, bbox_a, bbox_b, interaction_zone,
                   detail_a, detail_b, score, ground_truth, case_name,
                   output_path):
    """
    Vẽ 1 figure với 4 subplots cho 1 test case:

      ┌────────────────────┬────────────────────┐
      │  Subplot 1:        │  Subplot 2:        │
      │  Mask + BBox +     │  Interaction Zone  │
      │  Interaction Zone  │  + BBox            │
      ├────────────────────┼────────────────────┤
      │  Subplot 3:        │  Subplot 4:        │
      │  Strips của A      │  Strips của B      │
      │  (filled/empty/    │  (filled/empty/    │
      │   outside)         │   outside)         │
      └────────────────────┴────────────────────┘
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(
        f"{case_name}\n"
        f"Signal 1 score = {score:.4f}  →  "
        f"Dự đoán: {'A TRÊN' if score > 0.5 else 'B TRÊN' if score < 0.5 else 'UNCERTAIN'}  |  "
        f"Ground truth: {ground_truth}",
        fontsize=14, fontweight="bold",
        color="green" if (
            (score > 0.5 and ground_truth == "A_on_top") or
            (score < 0.5 and ground_truth == "B_on_top")
        ) else "red"
    )

    img_h, img_w = mask_a.shape

    # ═══════════════════════════════════════════════════════════
    # Subplot 1: Mask + BBox + Interaction Zone
    # ═══════════════════════════════════════════════════════════
    ax = axes[0, 0]
    canvas = np.zeros((img_h, img_w, 3), dtype=np.float64)

    canvas[mask_a] = COLOR_A
    canvas[mask_b] = COLOR_B
    both = mask_a & mask_b
    canvas[both] = (COLOR_A + COLOR_B) / 2.0

    zone_only = interaction_zone & ~(mask_a | mask_b)
    canvas[zone_only] = (canvas[zone_only] * 0.4 + COLOR_ZONE * 0.6)

    ax.imshow(canvas.astype(np.uint8))
    _draw_bbox(ax, bbox_a, COLOR_A, "A (bbox)")
    _draw_bbox(ax, bbox_b, COLOR_B, "B (bbox)")

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_A / 255.0, edgecolor="black", label="Mask A"),
        mpatches.Patch(facecolor=COLOR_B / 255.0, edgecolor="black", label="Mask B"),
        mpatches.Patch(facecolor=(COLOR_A + COLOR_B) / 510.0, edgecolor="black",
                       label="A ∩ B (overlap)"),
        mpatches.Patch(facecolor=COLOR_ZONE / 255.0, edgecolor="black",
                       label="Interaction Zone (only)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_title("1. Mask + BBox + Interaction Zone", fontsize=12, fontweight="bold")
    ax.axis("off")

    # ═══════════════════════════════════════════════════════════
    # Subplot 2: Interaction Zone chi tiết + BBox
    # ═══════════════════════════════════════════════════════════
    ax = axes[0, 1]
    zone_canvas = np.zeros((img_h, img_w, 3), dtype=np.float64)

    zone_canvas[interaction_zone] = COLOR_ZONE
    a_in_zone = mask_a & interaction_zone
    b_in_zone = mask_b & interaction_zone
    zone_canvas[a_in_zone] = COLOR_A
    zone_canvas[b_in_zone] = COLOR_B
    both_in_zone = mask_a & mask_b & interaction_zone
    zone_canvas[both_in_zone] = (COLOR_A + COLOR_B) / 2.0

    ax.imshow(zone_canvas.astype(np.uint8))
    _draw_bbox(ax, bbox_a, COLOR_A, "A")
    _draw_bbox(ax, bbox_b, COLOR_B, "B")

    zone_area = int(interaction_zone.sum())
    ax.text(5, 5, f"Zone area: {zone_area} px",
            color="black", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_title("2. Interaction Zone (dilate 11px)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    # ═══════════════════════════════════════════════════════════
    # Hàm vẽ strips cho 1 vật thể (dùng cho cả subplot 3 và 4)
    # ═══════════════════════════════════════════════════════════
    def _draw_strip_subplot(ax, detail, mask_obj, bbox_obj, color_obj_label,
                            interaction_zone, edge_colors_filled,
                            edge_colors_empty):
        """
        Vẽ strips. Mỗi pixel strip có 1 trong 3 trạng thái:
          1. strip ∩ zone ∩ mask   → FILLED  (màu đậm theo cạnh)
          2. strip ∩ zone ∩ ~mask  → EMPTY   (màu nhạt theo cạnh)
          3. strip ∩ ~zone         → OUTSIDE (xám đậm, không tính)

        Quan trọng: LUÔN vẽ strip outline dù edge bị skip, để thấy vùng
        strip nằm ở đâu trên bbox.
        """
        strip_canvas = np.zeros((img_h, img_w, 3), dtype=np.float64)

        # Nền: mask → màu vật thể nhạt để thấy ngữ cảnh
        strip_canvas[mask_obj] = (color_obj_label * 0.2)

        legend_items = []
        has_active_edge = False

        for edge_name, info in detail['edges'].items():
            rs, cs = info['strip_slices']

            # Luôn vẽ strip region dù skip hay không
            strip_region = np.zeros((img_h, img_w), dtype=bool)
            strip_region[rs, cs] = True

            outside_part = strip_region & (~interaction_zone)
            # Strip ngoài zone → luôn tô xám đậm
            strip_canvas[outside_part] = COLOR_STRIP_OUTSIDE_ZONE

            if info.get('skip', False):
                # Strip hoàn toàn ngoài zone → không vẽ filled/empty,
                # nhưng đã tô outside ở trên.
                # Thêm placeholder vào legend (không vẽ patch để tránh clutter)
                continue

            # ── Strip có giao zone → tô filled / empty ──
            has_active_edge = True
            zone_part = strip_region & interaction_zone

            filled = zone_part & mask_obj
            empty = zone_part & (~mask_obj)

            # FILLED → màu đậm
            strip_canvas[filled] = edge_colors_filled[edge_name]
            # EMPTY → màu nhạt
            strip_canvas[empty] = edge_colors_empty[edge_name]

            legend_items.append(mpatches.Patch(
                facecolor=edge_colors_filled[edge_name] / 255.0,
                edgecolor="black",
                label=(f"{edge_name}: "
                       f"filled={info['filled']}  empty={info['empty']}  "
                       f"recession={info['recession']:.3f}")
            ))

        # Thêm legend cho vùng outside
        legend_items.append(mpatches.Patch(
            facecolor=COLOR_STRIP_OUTSIDE_ZONE / 255.0,
            edgecolor="black",
            label="strip ngoài zone"
        ))

        # Nếu không có edge active → ghi chú
        if not has_active_edge:
            legend_items.append(mpatches.Patch(
                facecolor="none", edgecolor="none",
                label="(không có strip giao zone)"
            ))

        ax.imshow(strip_canvas.astype(np.uint8))
        _draw_bbox(ax, bbox_obj, color_obj_label, detail['label'])
        ax.legend(handles=legend_items, loc="lower right", fontsize=9,
                  title="FILLED=đậm  EMPTY=nhạt  xám=bỏ qua")

    # ── Subplot 3: Strips của A ────────────────────────────
    ax = axes[1, 0]
    _draw_strip_subplot(ax, detail_a, mask_a, bbox_a, COLOR_A,
                        interaction_zone, EDGE_COLORS_FILLED, EDGE_COLORS_EMPTY)
    ax.set_title(
        f"3. Strips Vật A  |  avg_recession = {detail_a['avg_recession']:.4f}",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    # ── Subplot 4: Strips của B ────────────────────────────
    ax = axes[1, 1]
    _draw_strip_subplot(ax, detail_b, mask_b, bbox_b, COLOR_B,
                        interaction_zone, EDGE_COLORS_FILLED, EDGE_COLORS_EMPTY)
    ax.set_title(
        f"4. Strips Vật B  |  avg_recession = {detail_b['avg_recession']:.4f}",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Đã lưu: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
# 5. CHẠY TEST CHÍNH
# ═══════════════════════════════════════════════════════════════

def run_test_case(case_name, mask_a, mask_b, bbox_a, bbox_b, image,
                   ground_truth, output_path):
    print(f"\n{'━' * 70}")
    print(f"  TEST CASE: {case_name}")
    print(f"  Ground truth: {ground_truth}")
    print(f"{'━' * 70}")

    # Tính interaction zone
    interaction_zone = _get_interaction_zone(bbox_a, bbox_b, mask_a.shape)

    # Tính recession chi tiết
    detail_a = compute_recession_detail(mask_a, bbox_a, interaction_zone, "A")
    detail_b = compute_recession_detail(mask_b, bbox_b, interaction_zone, "B")

    # Tính score thật
    score = _s_bbox_recession(mask_a, mask_b, bbox_a, bbox_b, interaction_zone)

    # In report
    print_recession_report(detail_a, detail_b, score)

    # Kiểm tra ensemble full
    result, conf, details = determine_occlusion_order(
        mask_a, mask_b, bbox_a, bbox_b, image, confidence_threshold=0.1
    )
    print(f"  [Full Ensemble] result={result}  conf={conf:.4f}  method={details['method']}")
    print(f"  Scores: {details['scores']}")
    print()

    is_correct_signal1 = (
        (score > 0.5 and ground_truth == "A_on_top") or
        (score < 0.5 and ground_truth == "B_on_top")
    )
    is_correct_ensemble = (result == ground_truth)
    print(f"  Signal 1 dự đoán ĐÚNG?      {'✅ CÓ' if is_correct_signal1 else '❌ KHÔNG'}")
    print(f"  Full ensemble dự đoán ĐÚNG? {'✅ CÓ' if is_correct_ensemble else '❌ KHÔNG'}")

    # Visualize
    if HAS_PLT:
        visualize_case(
            mask_a, mask_b, bbox_a, bbox_b,
            interaction_zone, detail_a, detail_b,
            score, ground_truth, case_name, output_path
        )
    else:
        print("\n  (matplotlib không có, bỏ qua visualization)")

    return is_correct_signal1


def main():
    parser = argparse.ArgumentParser(
        description="Test & Visualize Signal 1 — BBox Boundary Recession"
    )
    parser.add_argument("--mask_a", type=str, default=None,
                        help="Đường dẫn ảnh mask A (grayscale/binary)")
    parser.add_argument("--mask_b", type=str, default=None,
                        help="Đường dẫn ảnh mask B (grayscale/binary)")
    parser.add_argument("--gt", type=str, default=None,
                        choices=["A_on_top", "B_on_top"],
                        help="Ground truth (mặc định tự đoán theo mask lớn hơn)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  SIGNAL 1 — BBox Boundary Recession  |  Test & Visualize          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  matplotlib: {'CÓ ✓' if HAS_PLT else 'KHÔNG ✗'}")
    print(f"  opencv:     {'CÓ ✓' if HAS_CV2 else 'KHÔNG ✗'}")

    base_dir = os.path.dirname(__file__)

    # ═══════════════════════════════════════════════════════════
    # Thử load mask từ file nếu có
    # ═══════════════════════════════════════════════════════════
    masks_from_file = None
    if args.mask_a is not None or args.mask_b is not None:
        try:
            masks_from_file = load_masks_from_images(args.mask_a, args.mask_b)
        except Exception as e:
            print(f"\n  [ERROR] Không thể load mask từ file: {e}")
            print("  → Fallback sang mask giả lập mặc định.\n")
            masks_from_file = None

    cases = []

    if masks_from_file is not None:
        mask_a, mask_b, bbox_a, bbox_b, image, source = masks_from_file
        # Nếu user truyền ground truth → dùng. Nếu không → mặc định A_on_top
        gt = args.gt if args.gt else "A_on_top"
        cases.append({
            "name": f"Custom (load từ file) [{source}]",
            "mask_a": mask_a, "mask_b": mask_b,
            "bbox_a": bbox_a, "bbox_b": bbox_b,
            "image": image, "gt": gt,
            "output": os.path.join(base_dir, "signal1_custom.png"),
        })
    else:
        # ── Test Case 1: A TRÊN ──
        mask_a, mask_b, bbox_a, bbox_b, image = create_test_masks()
        cases.append({
            "name": "Case 1: A trên, B dưới (B bị cắt góc trên-phải)",
            "mask_a": mask_a, "mask_b": mask_b,
            "bbox_a": bbox_a, "bbox_b": bbox_b,
            "image": image, "gt": "A_on_top",
            "output": os.path.join(base_dir, "signal1_case1.png"),
        })

        # ── Test Case 2: B TRÊN ──
        mask_a2, mask_b2, bbox_a2, bbox_b2, image2 = create_test_case_2()
        cases.append({
            "name": "Case 2: B trên, A dưới (A bị cắt góc trên-trái)",
            "mask_a": mask_a2, "mask_b": mask_b2,
            "bbox_a": bbox_a2, "bbox_b": bbox_b2,
            "image": image2, "gt": "B_on_top",
            "output": os.path.join(base_dir, "signal1_case2.png"),
        })

    # ── Chạy tất cả cases ──
    results = []
    for case in cases:
        ok = run_test_case(
            case["name"], case["mask_a"], case["mask_b"],
            case["bbox_a"], case["bbox_b"], case["image"],
            case["gt"], case["output"]
        )
        results.append(ok)

    # ── Tổng kết ──
    print(f"\n{'═' * 70}")
    print(f"  TỔNG KẾT")
    print(f"{'═' * 70}")
    total = len(results)
    passed = sum(results)
    print(f"  Signal 1: {passed}/{total} test dự đoán đúng")
    if passed == total:
        print(f"  ✅ PASS — Signal 1 hoạt động chính xác!")
    else:
        print(f"  ⚠️  {total - passed} test sai — cần kiểm tra lại logic")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
