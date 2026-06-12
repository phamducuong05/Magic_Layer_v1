# Occlusion Order — Giải thích chi tiết

> Tài liệu giải thích toàn bộ code trong `backend/occlusion_order.py`: ý tưởng, thuật toán, workflow, và ví dụ minh họa.

---

## 1. Tổng quan

### Vấn đề cần giải quyết

Trong một ảnh có nhiều vật thể chồng lên nhau, khi kéo thả vật thể ra khỏi nền, **vùng bị che (occluded region)** bị mất thông tin pixel. Để "đoán" lại vùng bị che, hệ thống cần biết **vật thể nào đang nằm trên, vật thể nào nằm dưới**. Thứ tự này gọi là **occlusion order** (thứ tự che chắn).

```
┌─────────────────────────────────────────────┐
│  Ảnh gốc: 2 người đứng chồng lên nhau      │
│                                             │
│   ┌──────────┐                              │
│   │ Người A  │ ← đang đè lên người B       │
│   │    ┌─────┼──────┐                       │
│   │    │overlap│      │                     │
│   └────┼─────┘      │                       │
│        │  Người B   │                       │
│        └────────────┘                       │
│                                             │
│  Câu hỏi: Ai trên? Ai dưới?                │
│  → Cần occlusion order để biết.            │
└─────────────────────────────────────────────┘
```

### Vai trò trong pipeline

```
Ảnh gốc + SAM3 masks
        │
        ▼
┌──────────────────┐
│  Occlusion Order │ ← File này (occlusion_order.py)
│  (ai đè ai?)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Amodal Completion│ ← Pix2Gestalt
│  (sinh pixel mới) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Inpaint / Blend  │ ← LaMa
│  (hoàn thiện)     │
└──────────────────┘
```

File `occlusion_order.py` cung cấp **2 hàm chính** cho pipeline:

| Hàm | Vai trò |
|---|---|
| `determine_occlusion_order()` | So sánh **2 vật thể** → ai trên, ai dưới |
| `build_occlusion_graph()` | Xây dựng **đồ thị occlusion** cho **tất cả** vật thể trong ảnh |

---

## 2. Kiến trúc tổng thể

```
                        ┌─────────────────────┐
                        │  CORE UTILITIES      │
                        │  _bboxes_overlap()   │
                        │  _get_interaction_   │
                        │      zone()          │
                        └──────────┬──────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
   ┌────────────────┐   ┌────────────────┐    ┌────────────────┐
   │  SIGNAL 1      │   │  SIGNAL 2-7    │    │  FALLBACKS     │
   │  BBox Boundary │   │  Contour,      │    │  Fill Ratio    │
   │  Recession     │   │  Curvature,    │    │  Vertical Pos  │
   │                │   │  T-Junction,   │    │                │
   │                │   │  Convexity,    │    │                │
   │                │   │  Solidity,     │    │                │
   │                │   │  Edge Gradient │    │                │
   └───────┬────────┘   └───────┬────────┘    └───────┬────────┘
           │                    │                     │
           └────────────────────┼─────────────────────┘
                                │
                                ▼
                   ┌────────────────────────┐
                   │  determine_occlusion_  │
                   │  order()               │
                   │  (Ensemble 7 signals   │
                   │   + cascade fallback)  │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │  build_occlusion_graph │
                   │  ()                    │
                   │  (Topological sort     │
                   │   over all pairs)      │
                   └────────────────────────┘
```

---

## 3. Core Utilities

### 3.1. `_bboxes_overlap(bbox_a, bbox_b)`

**Mục đích:** Kiểm tra hai bounding box có giao nhau không.

**Tham số:**
- `bbox_a`: `(x, y, w, h)` — bounding box của vật thể A
- `bbox_b`: `(x, y, w, h)` — bounding box của vật thể B

**Thuật toán:** Dùng phủ định của điều kiện "không giao nhau":
```
Không giao nếu:  A hoàn toàn bên trái B
               OR A hoàn toàn bên phải B
               OR A hoàn toàn phía trên B
               OR A hoàn toàn phía dưới B
```

**Code logic:**
```python
return not (ax+aw <= bx or bx+bw <= ax or ay+ah <= by or by+bh <= ay)
```

**Ví dụ:**
```
bbox_a = (10, 10, 50, 80)   # x=10, y=10, w=50, h=80
bbox_b = (30, 40, 60, 70)   # x=30, y=40, w=60, h=70

A: từ (10,10) đến (60,90)
B: từ (30,40) đến (90,110)
→ Có giao nhau tại (30,40)-(60,90)
→ Trả về True
```

---

### 3.2. `_get_interaction_zone(bbox_a, bbox_b, img_shape)`

**Mục đích:** Tính **vùng tương tác** (interaction zone) — vùng giao của 2 bbox, sau đó **dilate** (giãn nở) để bắt thêm cả vùng biên xung quanh.

**Tại sao cần interaction zone?**
- Modal mask (mask quan được) chỉ cho biết vật thể ở đâu, không cho biết ranh giới thực sự.
- Vùng giao của 2 bbox là nơi occlusion xảy ra → đây là "bàn cờ" để các signal so sánh.
- Dilate thêm để bắt cả đường biên ngay sát vùng giao (vì mask có thể không khít pixel).

**Tham số:**
- `bbox_a`, `bbox_b`: `(x, y, w, h)`
- `img_shape`: `(H, W)` — kích thước ảnh

**Thuật toán:**
```
1. Tính hình chữ nhật giao nhau:
   ix0 = max(ax, bx)    iy0 = max(ay, by)
   ix1 = min(ax+aw, bx+bw)  iy1 = min(ay+ah, by+bh)

2. Nếu không giao → trả về None

3. Tạo mask nhị phân (H,W) = False
4. Gán vùng giao = True

5. Dilate với kernel 11×11 để giãn nở biên
```

**Ví dụ minh họa:**
```
Ảnh 200×300

bbox_a = (50, 50, 100, 120)  → A từ (50,50) đến (150,170)
bbox_b = (100, 100, 100, 100) → B từ (100,100) đến (200,200)

Giao nhau (trước dilate):
  (100, 100) → (150, 170)  — hình chữ nhật 50×70

Sau dilate 11×11:
  (~95, ~95) → (~155, ~175)  — giãn ra mỗi bên ~5 pixel
```

**Trả về:** `np.ndarray` bool `(H, W)` hoặc `None`

---

## 4. Bảy Signals (Tín hiệu)

Mỗi signal trả về một **score trong khoảng [0, 1]**:
- **Score > 0.5** → A đang nằm trên B
- **Score < 0.5** → B đang nằm trên A
- **Score = 0.5** → Không xác định được

### 4.1. Signal 1 — BBox Boundary Recession (`_s_bbox_recession`)

**Ý tưởng:** Tại cạnh bbox giáp interaction zone, mask nào **lõm vào** (không đầy viền) → vật đó đang bị che → nằm dưới.

**Ví dụ trực quan:**
```
Interaction zone (vùng giao):

   ┌─────────────────────────┐
   │  Vật A (trên):          │
   │  Mask đầy đủ đến viền   │ ← Không lõm → A trên
   │██████████████████████   │
   │██████████████████████   │
   │██████████┌──────────────┼──────┐
   │██████████│  Vật B (dưới)│      │
   │██████████│  Mask lõm    │      │
   └──────────│  vào trong   │      │
              │   (recession)│      │
              └──────────────┘      │
```

**Thuật toán chi tiết:**
```
1. Với mỗi vật thể, lấy 4 cạnh bbox (top, bottom, left, right)
2. Mỗi cạnh → tạo một "strip" (dải) dày 3-8 pixel
3. Chỉ xét strip nằm trong interaction zone
4. Tính tỷ lệ lõm = 1 - (pixel mask filled / tổng pixel strip)
5. Trung bình recession của 4 cạnh
6. Score = recession_B / (recession_A + recession_B)
   → B lõm nhiều hơn → score cao → A trên ✓
```

**Trọng số trong ensemble: 0.25** (cao nhất)

---

### 4.2. Signal 2 — Contour Curvature (`_s_contour_curvature`)

**Ý tưởng:** Đường contour bị **bẻ vào (concave)** tại interaction zone → vật đó đang bị che → nằm dưới.

**Ví dụ trực quan:**
```
   Vật A (trên):                Vật B (dưới):
   Contour lồi ra ngoài         Contour lõm vào trong
        ╭──╮                         ╭──╮
       │    │                       │    │
   ────╯    ╰────               ────╯    ╰────
   (convex, cross > 0)          (concave, cross < 0)
```

**Thuật toán chi tiết:**
```
1. Tìm contour lớn nhất của mỗi mask (cv2.findContours)
2. Với mỗi điểm trên contour nằm trong interaction zone:
   a. Lấy vector v1 = point[i] - point[i-window]
   b. Lấy vector v2 = point[i+window] - point[i]
   c. Cross product: cross = v1.x * v2.y - v1.y * v2.x
   d. Chuẩn hóa: curvature = cross / (|v1| * |v2|)
3. Trung bình curvature trong interaction zone
4. Score = clip(0.5 + (curv_A - curv_B) / (2*|diff| + ε) * 0.5)
   → curv_B âm (concave) → diff dương → score > 0.5 → A trên ✓
```

**Trọng số: 0.20**

---

### 4.3. Signal 3 — T-Junction (`_s_t_junction`)

**Ý tưởng:** Contour nào **kết thúc (endpoint)** trong interaction zone → vật đó bị che → nằm dưới. Contour nào **đi xuyên qua liên tục** → vật đó trên.

**Ví dụ trực quan:**
```
   Vật A (trên):                Vật B (dưới):
   Contour đi xuyên qua         Contour kết thúc tại zone
   ═══════════════════          ──────╮
   (passthrough, liên tục)      ╭─────╯ (endpoint)
                                (bị cắt bởi A)
```

**Thuật toán chi tiết:**
```
1. Vẽ contour thành ảnh 1-pixel thick
2. Chỉ giữ phần nằm trong interaction zone
3. Dùng filter 3×3 đếm số hàng xóm của mỗi pixel contour:
   - Endpoint: pixel có ≤ 2 hàng xóm (điểm cuối)
4. Đếm endpoints và tổng độ dài contour trong zone
5. Score = 0.6 * (ep_B / ep_total) + 0.4 * (len_A / len_total)
   → B nhiều endpoint hơn → score cao → A trên ✓
   → A dài hơn (xuyên qua) → score cao → A trên ✓
```

**Trọng số: 0.20**

---

### 4.4. Signal 4 — Convexity Deficit (`_s_convexity_deficit`)

**Ý tưởng:** Vật nào có **convex hull bị cắt nhiều** hơn trong interaction zone → vật đó bị che → nằm dưới.

**Ví dụ trực quan:**
```
   Vật A (trên):                Vật B (dưới):
   Hull ≈ Mask (ít deficit)     Hull >> Mask (nhiều deficit)
      ╭─────╮                      ╭─────────╮
     │ ███ │                     │ ░░░░░░░░░│  ← hull
     │ ███ │                     │ ░█████░░░│  ← mask
      ╰─────╯                      ╰─────────╯
                                   ░░ = deficit (bị cắt)
```

**Thuật toán chi tiết:**
```
1. Tính convex hull của mask (cv2.convexHull + fillPoly)
2. Deficit = (hull - mask) ∩ interaction_zone
3. Deficit ratio = deficit.sum() / interaction_zone.sum()
4. Score = deficit_B / (deficit_A + deficit_B)
   → B deficit nhiều hơn → score cao → A trên ✓
```

**Trọng số: 0.15**

---

### 4.5. Signal 5 — Local Solidity (`_s_local_solidity`)

**Ý tưởng:** Vật nào có **solidity cao hơn** (hình dạng đầy đặn hơn) trong interaction zone → nằm trên. Vật thường có hình dạng "đầy" hơn khi nó là foreground.

**Thuật toán chi tiết:**
```
1. Cắt mask ∩ interaction_zone
2. Tìm contour trong vùng cắt
3. Solidity = area / hull_area
4. Score = solidity_A / (solidity_A + solidity_B)
   → A đầy đặn hơn → score cao → A trên ✓
```

**Trọng số: 0.08**

---

### 4.6. Signal 6 — Contour Length (`_s_contour_length`)

**Ý tưởng:** Vật nào **sở hữu đường biên dài hơn** trong interaction zone → vật đó "owns" biên → nằm trên. Vật trên thường tạo ra đường biên rõ ràng hơn.

**Thuật toán chi tiết:**
```
1. Vẽ contour 1-pixel thick
2. Đếm pixel contour nằm trong interaction_zone
3. Score = length_A / (length_A + length_B)
   → A có biên dài hơn → score cao → A trên ✓
```

**Trọng số: 0.07**

---

### 4.7. Signal 7 — Edge Gradient (`_s_edge_gradient`)

**Ý tưởng:** Vật nào có **gradient mạnh hơn** tại biên trong zone → vật đó owns biên → nằm trên. Artwork thường có outline sắc nét.

**Thuật toán chi tiết:**
```
1. Chuyển ảnh sang grayscale
2. Tính gradient bằng Sobel (√(Sobel_x² + Sobel_y²))
3. Vẽ contour 1-pixel thick
4. Tính trung bình gradient tại contour ∩ interaction_zone
5. Score = grad_A / (grad_A + grad_B)
   → A có gradient mạnh hơn → score cao → A trên ✓
```

**Trọng số: 0.05** (thấp nhất, chỉ là gợi ý bổ sung)

---

## 5. Bảng tổng hợp Signals

| # | Signal | Ý tưởng chính | Trọng số | Dựa trên |
|---|---|---|---|---|
| 1 | BBox Boundary Recession | Mask lõm vào = bị che = dưới | 0.25 | Mask + BBox |
| 2 | Contour Curvature | Contour concave = bị che = dưới | 0.20 | Mask contour |
| 3 | T-Junction | Contour kết thúc = bị che = dưới | 0.20 | Mask contour |
| 4 | Convexity Deficit | Hull bị cắt nhiều = bị che = dưới | 0.15 | Mask + Hull |
| 5 | Local Solidity | Solidity cao = đầy đặn = trên | 0.08 | Mask area |
| 6 | Contour Length | Biên dài = owns biên = trên | 0.07 | Mask contour |
| 7 | Edge Gradient | Gradient mạnh = owns biên = trên | 0.05 | Ảnh gốc |

---

## 6. Fallbacks (Dự phòng)

Khi ensemble 7 signals cho confidence thấp (dưới ngưỡng), hệ thống chuyển sang các phương pháp đơn giản hơn.

### 6.1. `_fallback_fill_ratio(mask_a, mask_b, bbox_a, bbox_b)`

**Ý tưởng:** Chia interaction zone thành **4 quadrant**. Quadrant nào mask điền đầy hơn → vật đó trên ở quadrant đó.

```
┌─────────┬─────────┐
│  Q0     │  Q1     │
│ A: 80%  │ A: 30%  │
│ B: 20%  │ B: 70%  │
├─────────┼─────────┤
│  Q2     │  Q3     │
│ A: 90%  │ A: 40%  │
│ B: 10%  │ B: 60%  │
└─────────┴─────────┘

A thắng Q0, Q2 → 2 votes
B thắng Q1, Q3 → 2 votes
→ Tie → confidence = 0
```

**Confidence:** `|votes_A - 2| / 2 * 0.2` (tối đa 0.2 khi 4-0)

### 6.2. `_fallback_vertical_position(bbox_a, bbox_b)`

**Ý tưởng:** Vật có **bottom thấp hơn** trong frame → foreground → trên. Đúng với ~70% trường hợp người đứng trong artwork (vì chân gần đáy = gần camera = foreground).

```
Ảnh:
┌────────────────────┐
│                    │
│   ╔═══╗            │
│   ║ A ║ ← trên     │
│   ╚═══╝            │
│      ┌──────────┐  │
│      │ B ← dưới │  │
│      │ bottom   │  │
│      │ thấp hơn │  │
│      └──────────┘  │
└────────────────────┘
```

**Confidence:** `min(0.15, |bottom_A - bottom_B| / 100)` — luôn thấp vì chỉ là heuristic.

---

## 7. Hàm chính: `determine_occlusion_order()`

### Signature

```python
def determine_occlusion_order(
    mask_a:   np.ndarray,              # bool (H, W) — mask của A
    mask_b:   np.ndarray,              # bool (H, W) — mask của B
    bbox_a:   Tuple[int,int,int,int],  # (x, y, w, h) — bbox của A
    bbox_b:   Tuple[int,int,int,int],  # (x, y, w, h) — bbox của B
    image_np: np.ndarray,              # RGB uint8 (H, W, 3) — ảnh gốc
    confidence_threshold: float = 0.25 # ngưỡng confidence
) -> Tuple[str, float, dict]:
```

### Workflow chi tiết

```
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 0: Kiểm tra đầu vào                                   │
│                                                             │
│ if bbox_a không giao bbox_b:                                │
│     return ('no_overlap', 1.0, {})                          │
│     → Không cần xử lý, 2 vật không chồng nhau              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 1: Tính interaction zone                              │
│                                                             │
│ zone = _get_interaction_zone(bbox_a, bbox_b, shape)         │
│ if zone is None:                                            │
│     return ('no_overlap', 1.0, {})                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 2: Tính 7 signals                                     │
│                                                             │
│ scores = {                                                  │
│     'bbox_recession':  _s_bbox_recession(...)    → 0.72     │
│     'curvature':       _s_contour_curvature(...) → 0.65     │
│     't_junction':      _s_t_junction(...)        → 0.80     │
│     'convexity':       _s_convexity_deficit(...) → 0.60     │
│     'solidity':        _s_local_solidity(...)    → 0.55     │
│     'contour_length':  _s_contour_length(...)    → 0.58     │
│     'gradient':        _s_edge_gradient(...)     → 0.52     │
│ }                                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 3: Weighted ensemble                                  │
│                                                             │
│ final = Σ(score_i × weight_i)                               │
│       = 0.72×0.25 + 0.65×0.20 + 0.80×0.20                   │
│       + 0.60×0.15 + 0.55×0.08 + 0.58×0.07 + 0.52×0.05      │
│       ≈ 0.665                                               │
│                                                             │
│ confidence = |final - 0.5| × 2 = |0.665 - 0.5| × 2 = 0.33  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 4: Quyết định                                         │
│                                                             │
│ if confidence >= threshold (0.25):                          │
│     → Dùng kết quả ensemble                                 │
│     final > 0.5 → 'A_on_top'                                │
│     final < 0.5 → 'B_on_top'                                │
│                                                             │
│ else:  ← Cascade fallback                                   │
│     → Fallback 1: fill_ratio                                │
│       if conf > 0.05: dùng kết quả này                     │
│     → Fallback 2: vertical_position                         │
│       if conf > 0.0: dùng kết quả này                      │
│     → else: 'uncertain'                                     │
└─────────────────────────────────────────────────────────────┘
```

### Trả về

```python
(result: str, confidence: float, details: dict)
```

**`result`** — một trong các giá trị:
| Giá trị | Ý nghĩa |
|---|---|
| `'A_on_top'` | Vật thể A đang đè lên B |
| `'B_on_top'` | Vật thể B đang đè lên A |
| `'no_overlap'` | Hai vật không chồng nhau |
| `'uncertain'` | Không xác định được |

**`confidence`** — float [0, 1], mức độ tự tin.

**`details`** — dict chứa thông tin debug:
```python
{
    'scores': {
        'bbox_recession': 0.72,
        'curvature': 0.65,
        't_junction': 0.80,
        'convexity': 0.60,
        'solidity': 0.55,
        'contour_length': 0.58,
        'gradient': 0.52,
    },
    'weights': { ... },
    'final': 0.665,
    'confidence': 0.33,
    'method': 'ensemble',  # hoặc 'fallback_fill_ratio', 'fallback_vertical', 'uncertain'
}
```

---

## 8. Hàm chính: `build_occlusion_graph()`

### Signature

```python
def build_occlusion_graph(
    visible_masks: List[np.ndarray],           # List bool (H, W)
    bboxes:        List[Tuple],                # List (x, y, w, h)
    image_np:      np.ndarray,                 # RGB uint8 (H, W, 3)
    confidence_threshold: float = 0.25
) -> Tuple[List[int], dict]:
```

### Mục đíu

Xây dựng **đồ thị occlusion** cho **tất cả** vật thể trong ảnh, không chỉ 2 vật.

### Workflow chi tiết

```
Cho N vật thể:

┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 1: Tạo occlusion matrix (N × N)                       │
│                                                             │
│ occlusion_matrix[i][j] = confidence nếu i đè lên j          │
│ occlusion_matrix[i][j] = 0 nếu j không bị i đè             │
│                                                             │
│ for i in range(N):                                          │
│     for j in range(i+1, N):                                 │
│         if bbox_i giao bbox_j:                              │
│             result, conf, details =                          │
│                 determine_occlusion_order(mask_i, mask_j, ...)│
│             if result == 'A_on_top':                        │
│                 matrix[i][j] = conf                         │
│             elif result == 'B_on_top':                      │
│                 matrix[j][i] = conf                         │
│             # uncertain → bỏ qua (không gán)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 2: Topological sort                                   │
│                                                             │
│ being_occluded[j] = Σ_i matrix[i][j]  ← tổng bị đè         │
│                                                             │
│ depth_order = argsort(being_occluded)                       │
│ → Score thấp = ít bị đè = nằm trên cùng                    │
│ → Score cao = bị nhiều vật đè = nằm dưới cùng              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 3: Xây occluder_map                                   │
│                                                             │
│ occluder_map[j] = [i1, i2, ...]  ← các vật đè lên j        │
└─────────────────────────────────────────────────────────────┘
```

### Ví dụ minh họa

```
Ảnh có 4 vật thể: [0] bóng bay, [1] người, [2] cây, [3] nhà

Occlusion matrix (sau khi so sạch tất cả cặp):
        đè→  0    1    2    3
    ┌─────┬────┬────┬────┬────┐
  0 │bóng│  0 │0.3 │ 0  │0.5 │  ← bóng đè người (0.3), đè nhà (0.5)
  1 │người│ 0 │ 0  │0.4 │ 0  │  ← người đè cây (0.4)
  2 │cây │  0 │ 0  │ 0  │0.2 │  ← cây đè nhà (0.2)
  3 │nhà │  0 │ 0  │ 0  │ 0  │  ← nhà không đè ai
    └─────┴────┴────┴────┴────┘

being_occluded = [0, 0.3, 0.4, 0.7]  ← tổng theo cột

depth_order = argsort([0, 0.3, 0.4, 0.7]) = [0, 1, 2, 3]

Thứ tự từ trên xuống dưới:
  Layer 0 (trên cùng): bóng bay (0)  — không bị ai đè
  Layer 1:             người (1)     — bị bóng đè 0.3
  Layer 2:             cây (2)      — bị bóng+người đè 0.4
  Layer 3 (dưới cùng): nhà (3)      — bị cả 3 đè 0.7

occluder_map = {
    1: [0],       # người bị bóng đè
    2: [1],       # cây bị người đè
    3: [0, 2],    # nhà bị bóng và cây đè
}
```

### Trả về

```python
(depth_order: List[int], occluder_map: dict)
```

- `depth_order`: danh sách chỉ số vật thể, **từ trên xuống dưới**
- `occluder_map`: dict `{target_idx: [occluder_idx, ...]}` — ai đè lên ai

---

## 9. Ví dụ sử dụng

### Ví dụ 1: So sánh 2 vật thể

```python
import numpy as np
from backend.occlusion_order import determine_occlusion_order

# Giả lập dữ liệu
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Vật thể A: hình vuông 100×100 ở giữa
mask_a = np.zeros((480, 640), dtype=bool)
mask_a[100:200, 150:250] = True
bbox_a = (150, 100, 100, 100)

# Vật thể B: hình vuông 120×120 chồng lên A
mask_b = np.zeros((480, 640), dtype=bool)
mask_b[150:270, 180:300] = True
bbox_b = (180, 150, 120, 120)

# Xác định occlusion order
result, confidence, details = determine_occlusion_order(
    mask_a=mask_a,
    mask_b=mask_b,
    bbox_a=bbox_a,
    bbox_b=bbox_b,
    image_np=image,
    confidence_threshold=0.25,
)

print(f"Kết quả: {result}")
print(f"Confidence: {confidence:.2f}")
print(f"Method: {details['method']}")
print(f"Scores: {details['scores']}")
```

### Ví dụ 2: Xây dựng đồ thị cho nhiều vật thể

```python
from backend.occlusion_order import build_occlusion_graph

# Giả lập 3 vật thể
masks = [mask_a, mask_b, mask_c]  # mask_c là vật thể thứ 3
bboxes = [bbox_a, bbox_b, bbox_c]

depth_order, occluder_map = build_occlusion_graph(
    visible_masks=masks,
    bboxes=bboxes,
    image_np=image,
    confidence_threshold=0.25,
)

print(f"Thứ tự từ trên xuống dưới: {depth_order}")
print(f"Occluder map: {occluder_map}")

# In kết quả
object_names = ["Bóng bay", "Người", "Cây"]
for layer, idx in enumerate(depth_order):
    occluders = occluder_map.get(idx, [])
    occluder_names = [object_names[o] for o in occluders]
    print(f"  Layer {layer}: {object_names[idx]}"
          f" (bị đè bởi: {occluder_names if occluder_names else 'không ai'})")
```

---

## 10. Cascade Fallback — Chi tiết

Khi ensemble cho confidence thấp, hệ thống không dừng lại mà **cascade** qua các phương pháp dự phòng:

```
Ensemble confidence = 0.15 (< 0.25)
         │
         ▼
┌──────────────────────────────┐
│ Fallback 1: Fill Ratio       │
│ conf = 0.12                  │
│ 0.12 > 0.05? → CÓ            │
│ → Dùng kết quả này           │
│ → return ('A_on_top', 0.12)  │
└──────────────────────────────┘

Nếu conf ≤ 0.05:
         │
         ▼
┌──────────────────────────────┐
│ Fallback 2: Vertical Position│
│ conf = 0.08                  │
│ 0.08 > 0.0? → CÓ             │
│ → Dùng kết quả này           │
│ → return ('B_on_top', 0.08)  │
└──────────────────────────────┘

Nếu conf = 0.0:
         │
         ▼
┌──────────────────────────────┐
│ return ('uncertain', 0.0)    │
│ → Pipeline cần xử lý riêng   │
└──────────────────────────────┘
```

**Tại sao cascade?**
- Ensemble 7 signals mạnh nhấn cũng có thể thất bại (ảnh phức tạp, mask nhiễu).
- Fill ratio đơn giản nhưng đáng tin cậy khi 2 vật có sự khác biệt rõ ràng.
- Vertical position yếu nhất nhưng vẫn tốt hơn không có gì.
- Cuối cùng, `uncertain` cho pipeline biết cần xử lý đặc biệt (ví dụ: hỏi user).

---

## 11. Tích hợp với Pipeline

Trong hệ thống Magic Layer, `occlusion_order.py` được gọi theo flow:

```
1. SAM3 tách ảnh thành N vật thể
   → visible_masks[], bboxes[]

2. build_occlusion_graph(visible_masks, bboxes, image)
   → depth_order, occluder_map

3. Với mỗi vật thể j bị che:
   a. Xác định occluder(s) từ occluder_map[j]
   b. Tạo amodal mask = visible_mask + occluded region
   c. Pix2Gestalt sinh pixel cho vùng bị che
   d. LaMa inpaint hoàn thiện

4. Sắp xếp lại các layer theo depth_order
   → Render từ dưới lên trên
```

---

## 12. Lưu ý và Hạn chế

### Điểm mạnh
- **Ensemble 7 signals** từ nhiều góc độ khác nhau → robust hơn một phương pháp đơn lẻ.
- **Cascade fallback** đảm bảo luôn có kết quả (dù confidence có thể thấp).
- **Topological sort** xử lý được N vật thể, không chỉ 2.

### Hạn chế
- **Modal mask**: Tất cả signals dựa trên mask quan được (visible mask). Nếu mask bị nhiễu hoặc thiếu chính xác, kết quả occlusion order cũng ảnh hưởng.
- **Confidence thấp**: Khi 2 vật có hình dạng đối xứng hoặc occlusion rất nhỏ, ensemble có thể không tự tin → phải dùng fallback.
- **Vertical heuristic**: Fallback dựa trên vị trí dọc chỉ đúng ~70% trường hợp.
- **Không xử lý trường hợp vòng (cycle)**: Nếu A đè B, B đè C, C đè A (trong các cặp khác nhau), topological sort vẫn hoạt động nhưng kết quả có thể không chính xác về mặt vật lý.

### Cải tiến có thể
- Thêm signal dựa trên **texture/color consistency** (vật trên thường có texture nhất quán hơn).
- Dùng **machine learning** để học trọng số ensemble thay vì cố định.
- Xử lý **trường hợp vòng** bằng cách phá vòng ở cạnh yếu nhất.
