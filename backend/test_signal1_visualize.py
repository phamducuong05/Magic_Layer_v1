"""
Test & Visualize Signal 1 — BBox Boundary Recession (_s_bbox_recession)

Tạo 2 mask giả lập A đè lên B, rồi:
  1. Vẽ mask A, mask B, interaction zone
  2. Vẽ từng cạnh bbox với strip (dải viền)
  3. Highlight pixel filled vs empty trong mỗi strip
  4. Tính & in ra recession ratio từng cạnh, từng vật thể
  5. Cho biết signal dự đoán đúng hay sai

Cách chạy:
    python backend/test_signal1_visualize.py
"""

import sys
import os
import math

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

from backend.occlusion_order import (
    _get_interaction_zone,
    _s_bbox_recession,
    determine_occlusion_order,
)


# ═══════════════════════════════════════════════════════════════
# 1. TẠO 2 MASK GIẢ LẬP
# ═══════════════════════════════════════════════════════════════

def create_test_masks(img_h=300, img_w=400):
    """
    Tạo 2 mask: A (hình chữ nhật trên, đầy đủ) và B (hình chữ nhật dưới,
    bị 'cắt' ở góc trên-phải bởi A → cho thấy recession).

    A nằm TRÊN B (ground truth: A_on_top).

    Kịch bản:
      A = hình chữ nhật 120×80 ở vị trí (130, 60)
      B = hình chữ nhật 140×100 ở vị trí (100, 120)

           ┌──────────────────────────┐
           │   Ảnh 400×300            │
           │                          │
           │    ┌──────────────┐      │
           │    │   A (trên)   │      │
           │    │  full mask   │      │
           │    └──────┬───────┘      │
           │    ┌──────┴───────────┐  │
           │    │   B (dưới)      │  │
           │    │   bị cắt góc    │  │
           │    │   trên-phải     │  │
           │    └──────────────────┘  │
           │                          │
           └──────────────────────────┘
    """
    mask_a = np.zeros((img_h, img_w), dtype=bool)
    mask_b = np.zeros((img_h, img_w), dtype=bool)

    # ── Vật thể A: hình chữ nhật đầy đủ ──
    ax, ay, aw, ah = 130, 60, 120, 80
    mask_a[ay:ay+ah, ax:ax+aw] = True

    # ── Vật thể B: hình chữ nhật rộng hơn, nằm dưới & chồng lên A ──
    bx, by, bw, bh = 100, 120, 160, 110
    mask_b[by:by+bh, bx:bx+bw] = True

    # ── Cắt góc trên-phải của B để tạo hiệu ứng "bị che" ──
    # Giả lập rằng B bị A che mất vùng góc trên-phải
    # (giống trong thực tế: modal mask chỉ thấy phần không bị che)
    cut_x_start = 180   # góc phải bị cắt
    cut_y_end   = 155   # phần trên bị cắt
    mask_b[by:cut_y_end, cut_x_start:bx+bw] = False

    bbox_a = (ax, ay, aw, ah)
    bbox_b = (bx, by, bw, bh)

    # Ảnh giả (RGB) — gradient nhẹ cho gradient signal hoạt động
    image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for c in range(3):
        image[:, :, c] = np.linspace(50, 200, img_w, dtype=np.uint8)
    # Thêm một chút pattern để gradient có ý nghĩa
    yy, xx = np.mgrid[0:img_h, 0:img_w]
    image[:, :, 0] = (image[:, :, 0].astype(int) + xx // 4).clip(0, 255).astype(np.uint8)
    image[:, :, 1] = (image[:, :, 1].astype(int) + yy // 4).clip(0, 255).astype(np.uint8)

    return mask_a, mask_b, bbox_a, bbox_b, image


def create_test_case_2(img_h=300, img_w=400):
    """
    Kịch bản 2: B trên, A dưới (ngược lại case 1).
    B là vật nhỏ nằm trên A.
    """
    mask_a = np.zeros((img_h, img_w), dtype=bool)
    mask_b = np.zeros((img_h, img_w), dtype=bool)

    # A lớn, nằm dưới
    ax, ay, aw, ah = 80, 140, 200, 120
    mask_a[ay:ay+ah, ax:ax+aw] = True
    # Cắt góc trên-trái để giả lập bị B che
    mask_a[ay:ay+30, ax:ax+60] = False

    # B nhỏ, nằm trên
    bx, by, bw, bh = 120, 80, 100, 80
    mask_b[by:by+bh, bx:bx+bw] = True

    bbox_a = (ax, ay, aw, ah)
    bbox_b = (bx, by, bw, bh)

    image = np.random.randint(80, 200, (img_h, img_w, 3), dtype=np.uint8)
    return mask_a, mask_b, bbox_a, bbox_b, image


# ═══════════════════════════════════════════════════════════════
# 2. TÍNH RECESSION CHI TIẾT (instrumented version)
# ═══════════════════════════════════════════════════════════════

def compute_recession_detail(mask, bbox, interaction_zone, label="X"):
    """
    Phiên bản chi tiết của _recession() bên trong _s_bbox_recession.
    Trả về recession từng cạnh + toàn bộ thông tin debug.
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

        total = int(edge_region.sum())
        filled = int((edge_region & mask).sum())
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
    W = 70  # chiều rộng separator
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
# 4. VISUALIZATION (Matplotlib hoặc Text)
# ═══════════════════════════════════════════════════════════════

def visualize_masks_matplotlib(mask_a, mask_b, bbox_a, bbox_b,
                                interaction_zone, detail_a, detail_b,
                                score, ground_truth):
    """
    Vẽ 6 subplots:
      1. Original masks overlay
      2. Interaction zone
      3. A strips (tiled view, từng cạnh)
      4. B strips (tiled view, từng cạnh)
      5. Filled vs Empty pixels heatmap cho A
      6. Filled vs Empty pixels heatmap cho B
    """
    COLOR_A = [30, 120, 255]    # xanh dương
    COLOR_B = [255, 80, 80]     # đỏ
    COLOR_ZONE = [255, 255, 0]  # vàng

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Signal 1 — BBox Boundary Recession\n"
        f"Score={score:.4f}  ➤  Dự đoán: "
        f"{'A TRÊN (A_on_top)' if score > 0.5 else 'B TRÊN (B_on_top)' if score < 0.5 else 'KHÔNG XÁC ĐỊNH'}  |  "
        f"Ground truth: {ground_truth}",
        fontsize=14, fontweight="bold",
        color="green" if (
            (score > 0.5 and ground_truth == "A_on_top") or
            (score < 0.5 and ground_truth == "B_on_top")
        ) else "red"
    )

    img_h, img_w = mask_a.shape

    # ── Subplot 1: Mask Overlay ──────────────────────────
    ax = axes[0, 0]
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    canvas[mask_a] = COLOR_A
    canvas[mask_b] = COLOR_B
    # Giao nhau → tím
    both = mask_a & mask_b
    canvas[both] = [(a + b) // 2 for a, b in zip(COLOR_A, COLOR_B)]
    ax.imshow(canvas)
    # Vẽ bbox
    for bbox, color, lbl in [(bbox_a, COLOR_A, "A"), (bbox_b, COLOR_B, "B")]:
        rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                   linewidth=2, edgecolor=np.array(color)/255,
                                   facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(bbox[0]+2, bbox[1]-5, lbl, color=np.array(color)/255,
                fontsize=12, fontweight="bold")
    ax.set_title("1. Mask Overlay\n(A=xanh, B=đỏ, chung=tím)")
    ax.axis("off")

    # ── Subplot 2: Interaction Zone ──────────────────────
    ax = axes[0, 1]
    zone_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    zone_canvas[mask_a] = [c // 4 for c in COLOR_A]
    zone_canvas[mask_b] = [c // 4 for c in COLOR_B]
    zone_canvas[interaction_zone] = COLOR_ZONE
    ax.imshow(zone_canvas)
    for bbox, color, lbl in [(bbox_a, COLOR_A, "A"), (bbox_b, COLOR_B, "B")]:
        rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                   linewidth=2, edgecolor=np.array(color)/255,
                                   facecolor="none")
        ax.add_patch(rect)
    ax.set_title("2. Interaction Zone\n(vùng vàng = zone)")
    ax.axis("off")

    # ── Subplot 3 & 4: Strips cho A và B ─────────────────
    for idx, (detail, color, sp_row_sp2) in enumerate([
        (detail_a, COLOR_A, 3),
        (detail_b, COLOR_B, 4),
    ]):
        ax = axes[1, idx]
        strip_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        # Nền mask nhạt
        m_mask = detail['label'] == 'A'
        mask_obj = mask_a if m_mask else mask_b
        strip_canvas[mask_obj] = [c // 5 for c in color]
        # Vẽ strip từng cạnh
        EDGE_COLORS = {
            'top': [0, 255, 0],
            'bottom': [0, 200, 200],
            'left': [255, 200, 0],
            'right': [200, 0, 255],
        }
        legend_handles = []
        for edge_name, info in detail['edges'].items():
            if info.get('skip', False):
                continue
            rs, cs = info['strip_slices']
            ec = EDGE_COLORS.get(edge_name, [255, 255, 255])
            # Highlight strip region
            strip_region = np.zeros((img_h, img_w), dtype=bool)
            strip_region[rs, cs] = True
            filled_part = strip_region & mask_obj
            empty_part = strip_region & (~mask_obj)
            strip_canvas[filled_part] = ec
            # Đánh dấu empty bằng màu tối hơn / mẫu hatching
            strip_canvas[empty_part] = [c // 3 for c in ec]

            patch = mpatches.Patch(color=np.array(ec)/255, label=edge_name)
            legend_handles.append(patch)

        ax.imshow(strip_canvas)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
        ax.set_title(
            f"{sp_row_sp2}. Strips Vật {detail['label']}\n"
            f"(sáng=fill, tối=empty trong strip)"
        )
        ax.axis("off")

    # ── Subplot 5: Summary text ──────────────────────────
    ax = axes[0, 2]
    ax.axis("off")

    # Build text
    lines = []
    lines.append("══════════════════════════════════")
    lines.append("  RECESSION REPORT")
    lines.append("══════════════════════════════════")
    lines.append("")

    for detail in [detail_a, detail_b]:
        lines.append(f"Vật {detail['label']} (bbox={detail['bbox']}):")
        for ename, info in detail['edges'].items():
            if info.get('skip', False):
                lines.append(f"  {ename:>8s}: SKIP (no zone overlap)")
            else:
                lines.append(
                    f"  {ename:>8s}: "
                    f"fill={info['filled']:>4d}/{info['total']:>4d}  "
                    f"recession={info['recession']:.3f}"
                )
        lines.append(f"  {'avg':>8s}: {detail['avg_recession']:.4f}")
        lines.append("")

    lines.append("──────────────────────────────────")
    lines.append(f"  score = rec_B / (rec_A + rec_B)")
    lines.append(f"        = {detail_b['avg_recession']:.4f} / "
                 f"({detail_a['avg_recession']:.4f} + {detail_b['avg_recession']:.4f})")
    lines.append(f"        = {score:.4f}")
    lines.append("")

    if score > 0.5:
        pred = "A TRÊN ✓" if ground_truth == "A_on_top" else "A TRÊN ✗"
        lines.append(f"  ➤ {pred}")
    elif score < 0.5:
        pred = "B TRÊN ✓" if ground_truth == "B_on_top" else "B TRÊN ✗"
        lines.append(f"  ➤ {pred}")
    else:
        lines.append("  ➤ UNCERTAIN")
    lines.append("")
    lines.append(f"  Ground truth: {ground_truth}")

    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    # ── Subplot 6: Confusion / verdict ────────────────────
    ax = axes[1, 2]
    ax.axis("off")

    is_correct = (
        (score > 0.5 and ground_truth == "A_on_top") or
        (score < 0.5 and ground_truth == "B_on_top")
    )
    verdict_color = "#2ecc71" if is_correct else "#e74c3c"

    verdict_lines = [
        "VERDICT",
        "──────────────",
        "",
        f"Score:      {score:.4f}",
        f"Prediction: {'A_on_top' if score > 0.5 else 'B_on_top' if score < 0.5 else 'uncertain'}",
        f"Truth:      {ground_truth}",
        "",
        "CHÍNH XÁC ✓" if is_correct else "SAI ✗",
    ]
    ax.text(0.5, 0.5, "\n".join(verdict_lines),
            transform=ax.transAxes, fontsize=14,
            fontfamily="monospace",
            verticalalignment="center", horizontalalignment="center",
            color=verdict_color, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__),
                             "signal1_visualization.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Đã lưu hình: {out_path}")
    return out_path


def visualize_text_only(mask_a, mask_b, bbox_a, bbox_b,
                          interaction_zone):
    """Nếu không có matplotlib, in ra text representation đơn giản."""
    H, W = mask_a.shape
    # Lấy bounding box của region để crop hiển thị
    all_mask = mask_a | mask_b | interaction_zone
    rows = np.any(all_mask, axis=1)
    cols = np.any(all_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Padding
    pad = 10
    rmin = max(0, rmin - pad)
    rmax = min(H - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(W - 1, cmax + pad)

    LEGEND = (
        "  Ký hiệu:\n"
        "    A = pixel thuộc mask A\n"
        "    B = pixel thuộc mask B\n"
        "    # = cả A và B (overlap)\n"
        "    + = interaction zone (ngoài mask)\n"
        "    . = background\n"
    )
    print(f"\n  Text visualization (crop [{rmin}:{rmax+1}, {cmin}:{cmax+1}]):")
    print(f"  {'─' * (cmax - cmin + 1)}")
    for r in range(rmin, rmax + 1):
        row_chars = []
        for c in range(cmin, cmax + 1):
            in_a = mask_a[r, c]
            in_b = mask_b[r, c]
            in_z = interaction_zone[r, c]
            if in_a and in_b:
                row_chars.append('#')
            elif in_a:
                row_chars.append('A')
            elif in_b:
                row_chars.append('B')
            elif in_z:
                row_chars.append('+')
            else:
                row_chars.append('.')
        print(f"  {''.join(row_chars)}")
    print(f"  {'─' * (cmax - cmin + 1)}")
    print(LEGEND)


# ═══════════════════════════════════════════════════════════════
# 5. CHẠY TEST CHÍNH
# ═══════════════════════════════════════════════════════════════

def run_test_case(case_name, mask_a, mask_b, bbox_a, bbox_b, image,
                   ground_truth):
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

    # Kiểm tra ensemble full (optional)
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
    print(f"  Signal 1 dự đoán ĐÚNG?   {'✅ CÓ' if is_correct_signal1 else '❌ KHÔNG'}")
    print(f"  Full ensemble dự đoán ĐÚNG? {'✅ CÓ' if is_correct_ensemble else '❌ KHÔNG'}")

    # Visualize
    if HAS_PLT:
        visualize_masks_matplotlib(
            mask_a, mask_b, bbox_a, bbox_b,
            interaction_zone, detail_a, detail_b,
            score, ground_truth
        )
    else:
        print("\n  (matplotlib không có, chỉ hiển thị text)")
        visualize_text_only(mask_a, mask_b, bbox_a, bbox_b, interaction_zone)

    return is_correct_signal1


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  SIGNAL 1 — BBox Boundary Recession  |  Test & Visualize          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  matplotlib: {'CÓ ✓' if HAS_PLT else 'KHÔNG ✗'}")

    # ── Test Case 1: A TRÊN ──
    mask_a, mask_b, bbox_a, bbox_b, image = create_test_masks()
    ok1 = run_test_case(
        "Case 1: A trên, B dưới (B bị cắt góc trên-phải)",
        mask_a, mask_b, bbox_a, bbox_b, image,
        ground_truth="A_on_top"
    )

    # ── Test Case 2: B TRÊN ──
    mask_a2, mask_b2, bbox_a2, bbox_b2, image2 = create_test_case_2()
    ok2 = run_test_case(
        "Case 2: B trên, A dưới (A bị cắt góc trên-trái)",
        mask_a2, mask_b2, bbox_a2, bbox_b2, image2,
        ground_truth="B_on_top"
    )

    # ── Tổng kết ──
    print(f"\n{'═' * 70}")
    print(f"  TỔNG KẾT")
    print(f"{'═' * 70}")
    total = 2
    passed = sum([ok1, ok2])
    print(f"  Signal 1: {passed}/{total} test dự đoán đúng")
    if passed == total:
        print(f"  ✅ PASS — Signal 1 hoạt động chính xác!")
    else:
        print(f"  ⚠️  {total - passed} test sai — cần kiểm tra lại logic")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
