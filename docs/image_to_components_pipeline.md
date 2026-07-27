# Pipeline tách ảnh thành các thành phần (Image-to-Components)

## Tổng quan

`POST /api/process-image` nhận vào một ảnh và danh sách prompt văn bản phân cách
bởi dấu phẩy (`keywords`). API kiểm tra đầu vào, giảm kích thước nếu cần, rồi
gọi `backend.image_processor.process_image` (điều phối bởi
`backend.pipeline.orchestrator.process_image`).

Pipeline chạy 8 bước tuần tự dưới một khoá reentrant toàn process
(`_PIPELINE_LOCK`) nhằm ngăn các request đồng thời giải phóng model GPU của
nhau. Mỗi model được tải lười (lazy-load) bởi `ModelManager` và giải phóng ngay
sau khi bước của nó hoàn tất.

```text
Ảnh đầu vào + Prompts
   │
   ├─► Bước 1  Phân đoạn (Segmentation)          — SAM3
   ├─► Bước 2  Hoàn thiện Amodal (Completion)     — SDAmodal
   ├─► Bước 3  Xếp thứ tự & Tạo mask             — Phân tích che khuất
   ├─► Bước 4  Khôi phục đối tượng                — HD-Painter
   ├─► Bước 5  Tinh chỉnh support                 — BiRefNet lần 1
   ├─► Bước 6  Gom nhóm & Hợp thành               — Union-find + Alpha blend
   ├─► Bước 7  Matting nhóm cuối                   — BiRefNet lần 2
   └─► Bước 8  Tách lớp & Inpaint nền             — LaMa / SDXL
```

---

## Bước 1 — Phân đoạn có hướng dẫn văn bản (Text-guided Segmentation)

| Thông tin | Chi tiết |
|-----------|----------|
| **Module** | [segmentation.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/segmentation.py) |
| **Hàm chính** | [extract_raw_objects](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/segmentation.py#L19-L108) |
| **Model** | SAM3 (`checkpoints/sam3.pt`) |
| **GPU giải phóng** | Có — [orchestrator.py#L234-L235](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L234-L235) |
| **Điều phối tại** | [orchestrator.py#L226-L235](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L226-L235) |

### Luồng xử lý

1. **Tính embedding ảnh một lần duy nhất:**
   SAM3 processor gọi `processor.set_image(image)` để tính embedding cho toàn bộ ảnh.
   → [segmentation.py#L26](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/segmentation.py#L26)

2. **Duyệt từng prompt:**
   Với mỗi keyword trong danh sách `keywords`:
   - Chuỗi rỗng bị bỏ qua.
   - `processor.reset_all_prompts(state)` xoá trạng thái prompt trước, sau đó `processor.set_text_prompt(state, prompt=keyword)` chạy suy luận dựa trên văn bản.
   - Mỗi prompt có thể tạo ra một hoặc nhiều mask nhị phân.
   → [segmentation.py#L27-L59](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/segmentation.py#L27-L59)

3. **Chuẩn hoá mask:**
   Mỗi mask được chuẩn hoá (`_normalise_mask`) thành mảng uint8 nhị phân ở độ phân giải gốc. Bounding box chặt được tính bởi `_bbox_from_mask`. Mask rỗng bị loại bỏ.
   → [segmentation.py#L61-L80](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/segmentation.py#L61-L80)

4. **Tạo `DetectedObject`:**
   Mỗi mask hợp lệ được đóng gói thành một `DetectedObject`:
   - `object_id`: đánh số tuần tự `"object-0"`, `"object-1"`, …
   - `semantic_class`: nội dung text prompt
   - `display_label`: `"keyword"` hoặc `"keyword_0"`, `"keyword_1"` khi một prompt tạo nhiều mask
   - `modal_mask`: mảng uint8 nhị phân full-image — phần đối tượng đang nhìn thấy
   - `bbox`: `(x, y, width, height)` bounding box chặt
   - `segmentation_index`: thứ tự chèn (dùng cho sắp xếp nhóm ổn định ở bước sau)
   → [segmentation.py#L87-L95](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/segmentation.py#L87-L95)
   → Kiểu dữ liệu: [types.py#L37-L91](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/types.py#L37-L91)

### Thoát sớm

Nếu không phát hiện được đối tượng nào, pipeline trả về ảnh gốc làm background
cùng mảng layers rỗng, ghi nhận `decision=return_original_background`.
→ [orchestrator.py#L236-L266](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L236-L266)

### Kích thước kernel hình thái

Sau segmentation, kernel size cho morphology được tính từ kích thước ảnh:
`kernel_size = _calc_kernel_size(image_np, 0.0075)`. Kernel này được tái sử dụng
trong tách lớp (Stage 8a) và inpaint nền (Stage 8b).
→ [orchestrator.py#L268](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L268)

---

## Bước 2 — Phát hiện chồng lấn & Hoàn thiện Amodal (Amodal Completion)

| Thông tin | Chi tiết |
|-----------|----------|
| **Module** | [completion.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/completion/completion.py), [validate_completion.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/completion/validate_completion.py) |
| **Các hàm** | `link_overlap_partners`, `get_completion_candidates`, `complete_objects`, `filter_pairs_by_amodal_overlap` |
| **Model** | SDAmodal (`backend/models/completion/checkpoints/SDAmodal.pth`) |
| **GPU giải phóng** | Có — [orchestrator.py#L290-L291](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L290-L291) |
| **Điều phối tại** | [orchestrator.py#L269-L320](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L269-L320) |

### Bước 2a — Phát hiện chồng lấn (Overlap Detection)

`link_overlap_partners(objects)` sử dụng `find_cross_class_overlaps` để tìm tất cả
các cặp đối tượng mà **bounding box gốc (original modal bbox) chồng lên nhau theo
diện tích** và **semantic class khác nhau**. Các đối tượng cùng class không bao giờ
được ghép cặp ở đây (chúng sẽ được gom nhóm ở Bước 6). Mỗi đối tượng trong cặp ghi
nhận partner vào tập `overlap_partner_ids`.
→ [completion.py#L26-L57](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/completion/completion.py#L26-L57)

### Bước 2b — Xác định ứng viên Completion

`get_completion_candidates(objects)` trả về mọi `DetectedObject` có ít nhất một
overlap partner (giữ thứ tự ban đầu). Đối tượng không có chồng lấn khác class bị
bỏ qua hoàn toàn.
→ [completion.py#L60-L83](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/completion/completion.py#L60-L83)

### Bước 2c — Dự đoán Amodal Mask

`complete_objects(image, candidates, completion_model, …)` gửi tất cả modal mask
và bounding box của ứng viên tới SDAmodal trong một lần gọi batch:

```python
outputs = completion_model.complete(image, [d.modal_mask for d in candidates], [d.bbox for d in candidates])
```
→ [completion.py#L86-L107](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/completion/completion.py#L86-L107)

Mỗi đầu ra thô được kiểm tra từng đối tượng bởi `_validated_amodal_mask`:

1. **Kiểu & dtype** — phải là numpy array với dtype bool, integer, hoặc floating.
2. **Shape** — phải khớp `modal_mask.shape` (kích thước full-image).
3. **Giá trị hữu hạn** — không có NaN/Inf.
4. **Connected components** — amodal mask được chia thành các thành phần liên thông qua `divide_mask_to_connected_components`; các thành phần không chồng lấn với `modal_mask` gốc bị loại bỏ (ngăn các hình dạng ảo tách rời).
5. **Tăng trưởng diện tích** — `amodal_area / modal_area` không được vượt `max_area_growth_ratio` (mặc định `4.0`).
6. **Tăng trưởng bbox** — `amodal_bbox_area / modal_bbox_area` không được vượt `max_bbox_growth_ratio` (mặc định `9.0`).

→ [validate_completion.py#L39-L155](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/completion/validate_completion.py#L39-L155)

Khi thành công, `_store_valid_completion` ghi:
- `detected.amodal_mask` — toàn bộ hình dạng dự đoán (hiển thị + ẩn)
- `detected.completion_hole_mask` — `amodal_mask & ~modal_mask` — vùng bị ẩn
- `detected.completion_hole_area` — số pixel của vùng lỗ

Khi bất kỳ lỗi nào xảy ra (lỗi suy luận, không khớp hợp đồng, hoặc bị validation
loại), `_store_modal_fallback` ghi `amodal_mask = modal_mask` và
`completion_hole_mask = zeros`, đảm bảo các bước sau luôn có amodal mask hợp lệ.
→ [validate_completion.py#L16-L29](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/completion/validate_completion.py#L16-L29)

### Bước 2d — Xác nhận Amodal Overlap

`filter_pairs_by_amodal_overlap(objects, pairs)` kiểm tra lại từng cặp tiềm năng:
chỉ giữ lại những cặp mà cả hai amodal mask đã được validation **thực sự chồng lấn
nhau ít nhất một pixel**. Các cặp không đạt (ví dụ vì một bên đã fallback về
modal) bị ghi log và loại bỏ.
→ [completion.py#L195-L241](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/completion/completion.py#L195-L241)
→ Điều phối tại: [orchestrator.py#L301-L320](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L301-L320)

---

## Bước 3 — Xếp thứ tự sâu & Chuẩn bị mask khôi phục (Depth Ordering & Reconstruction Mask Preparation)

| Thông tin | Chi tiết |
|-----------|----------|
| **Module** | [mask_preparation.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py) |
| **Hàm chính** | [prepare_raw_reconstruction_masks](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L474-L586) |
| **Model** | Không — chỉ phân tích hình học thuần tuý |
| **Điều phối tại** | [orchestrator.py#L322-L407](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L322-L407) |

Bước này chỉ chạy khi `overlap_pairs` không rỗng.

### Bước 3a — Diện tích lỗ hiệu quả (Effective Hole Area)

Với mọi đối tượng có `completion_hole_area`, hàm `effective_hole_area` áp dụng
hai bộ lọc nhiễu:
- **Tuyệt đối:** lỗ dưới `minimum_hole_area_pixels` (mặc định `16`) → diện tích hiệu quả = 0.
- **Tương đối:** lỗ dưới `minimum_hole_area_ratio × modal_area` (mặc định `0.01`) → diện tích hiệu quả = 0.

→ [mask_preparation.py#L494-L516](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L494-L516)

### Bước 3b — Mask khôi phục có hướng (Directional Reconstruction Masks)

Với mỗi cặp `(A, B)` đã giữ lại, `_directional_reconstruction_mask` được gọi theo
cả hai hướng `(A←B)` và `(B←A)`:

1. `exact_seed = target.completion_hole_mask & occluder.modal_mask` — giao chính xác giữa lỗ khuyết và vật che.
2. Modal mask của occluder được nở (dilate) tuỳ chọn bởi `composition_margin_pixels` (mặc định `4`) để bù sai số ranh giới.
3. `candidate = hole & occluder_support` — giao mở rộng.
4. Lọc connected component: chỉ giữ lại thành phần chạm vào `exact_seed`.
5. Kết quả bị giới hạn bởi `amodal_mask & ~modal_mask` — phải nằm trong vùng ẩn dự đoán và ngoài phần target đang thấy.

Tạo ra hai mảng mỗi hướng: `exact_seed` (giao thuần) và `filtered` (tức `composition_mask` cho cặp occluder này).
→ [mask_preparation.py#L113-L170](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L113-L170)

### Bước 3c — Gán vai trò & Hướng tái tạo (Role & Direction Assignment)

`assign_directional_pair_roles` xử lý hai nhiệm vụ riêng biệt cho mỗi cặp:

1. **Hướng tái tạo (`reconstruction_directions`):**
   - **Miễn là diện tích bị che của hướng nào lớn hơn 0 (`area > 0`), hướng đó ĐỀU ĐƯỢC KHÔI PHỤC.**
   - Nếu cả 2 vật thể đều che một phần của nhau (ví dụ: bàn tay quấn quanh cuốn sách), CẢ HAI hướng đều được thêm vào `reconstruction_directions` và cả 2 vật thể đều được reconstruct.

2. **Vai trò độ sâu hiển thị (`occluded_id` / `occluder_id`):**
   - So sánh diện tích 2 hướng: vật thể có diện tích lỗ hướng lớn hơn sẽ là `occluded_id` (nằm ở dưới/đằng sau).
   - Nếu chênh lệch diện tích nằm trong `tie_tolerance_ratio` (mặc định `0.1`), cặp đó được đánh dấu là hòa/không rõ ràng (`ambiguous = True`). Kết quả này phục vụ việc sắp xếp thứ tự lớp (Depth Ordering Back-to-Front) ở Bước 6b.

`apply_pair_decisions` duyệt qua `reconstruction_directions` và đăng ký tất cả occluder hợp lệ vào `detected.occluder_ids` của từng target.
→ [mask_preparation.py#L173-L211](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L173-L211)
→ [occlusion.py#L173-L231](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/core/occlusion.py#L173-L231)
→ [mask_preparation.py#L545-L566](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L545-L566)

### Bước 3d — Xây dựng toàn bộ mask khôi phục

`build_reconstruction_masks(objects, kernel_size, …)` duyệt các đối tượng có
occluder đã gán và tạo tất cả mask không gian cho từng target:
→ [mask_preparation.py#L214-L471](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L214-L471)

**Các field lưu trên `DetectedObject`:**

| Field | Mô tả | Code |
|-------|-------|------|
| `reconstruction_seed_mask` | Giao pixel chính xác giữa lỗ completion và mỗi occluder modal | [#L412](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L412) |
| `reconstruction_mask` | **composition_mask** — hợp tất cả lỗ hướng đã lọc cho target này | [#L413](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L413) |
| `reconstruction_generation_seed_mask` | `composition_mask \| qualifying_occluders_in_roi` — hạt giống trước morphology | [#L414](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L414) |
| `reconstruction_generation_mask` | **generation_mask** — mở rộng qua closing + dilation, loại trừ vùng bảo vệ; đưa cho HD-Painter | [#L415](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L415) |
| `reconstruction_occluder_mask` | Hợp tất cả modal occluder đã gán, giới hạn trong ROI | [#L416](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L416) |
| `reconstruction_input_roi` | `SquareROI` — khung vuông crop tập trung vào `amodal_mask \| composition_mask` | [#L417](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L417) |
| `reconstruction_target_bbox_mask` | Cửa sổ full-image bao phủ bounding box chặt của target amodal | [#L418](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L418) |
| `reconstruction_foreign_modal_inside_bbox` | Modal mask các đối tượng khác nằm trong amodal bbox của target (vật che khuất) | [#L419-L421](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L419-L421) |
| `reconstruction_foreign_modal_outside_bbox` | Modal mask các đối tượng khác nằm ngoài amodal bbox của target (người ngoài cuộc) — **bảo vệ cứng** | [#L422-L424](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L422-L424) |
| `reconstruction_replacement_domain_mask` | Vùng amodal+composition mở rộng trong target bbox nơi occluder có thể bị thay thế | [#L425](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L425) |
| `reconstruction_replaceable_foreign_inside` | Pixel occluder đã gán nằm trong replacement domain (có thể bị ghi đè) | [#L426-L428](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L426-L428) |
| `reconstruction_protected_foreign_inside` | Pixel foreign-inside KHÔNG trong replacement domain (vẫn được bảo vệ) | [#L429-L431](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L429-L431) |
| `reconstruction_foreign_protection_mask` | `foreign_outside_bbox \| protected_foreign_inside` sau dilation — ranh giới bảo vệ cuối | [#L432](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L432) |
| `reconstruction_protected_mask` | `target_modal \| foreign_protection` | [#L433](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L433) |
| `reconstruction_accepted_rgb_mask` | `roi_mask & ~protected_mask` — vùng pixel mà đầu ra HD-Painter được phép sử dụng | [#L434](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L434) |

**Quy tắc bảo vệ:**
- `target.modal_mask`: **luôn được bảo vệ** — RGB gốc không bao giờ bị ghi đè tại đây.
- `foreign_modal_outside_bbox`: **bảo vệ cứng** — các đối tượng ngoài cuộc nằm ngoài amodal bbox target.
- `foreign_modal_inside_bbox`: chia thành `replaceable` (trong replacement domain, occluder đã xác nhận) và `protected` (phần còn lại).

**Cách tạo generation mask:**
→ [mask_preparation.py#L391-L410](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/mask_preparation.py#L391-L410)

```python
generation_seed = composition_mask | qualifying_occluders_in_roi
generation_mask = close_and_dilate(generation_seed)
generation_mask &= roi_mask & ~target_modal
generation_mask &= (~foreign_protection | qualifying_occluder_allowance)
generation_mask |= composition_mask  # đảm bảo luôn chứa lỗ cốt lõi
```

---

## Bước 4 — Khôi phục đối tượng (Object Reconstruction)

| Thông tin | Chi tiết |
|-----------|----------|
| **Module** | [reconstruction.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/reconstruction.py) |
| **Hàm chính** | [reconstruct_objects](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/reconstruction.py#L88-L554) |
| **Model** | HD-Painter (`ds8_inp`, 512px gốc + tuỳ chọn 2048px super-resolution) |
| **GPU giải phóng** | Có — [orchestrator.py#L476-L482](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L476-L482) |
| **Điều phối tại** | [orchestrator.py#L414-L491](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L414-L491) |

Bước này chỉ chạy khi ít nhất một đối tượng có `reconstruction_mask` không rỗng.
→ [orchestrator.py#L409-L413](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L409-L413)

### Bước 4a — Chuẩn bị đầu vào

Với mỗi đối tượng có `reconstruction_mask` không rỗng:
→ [reconstruction.py#L110-L200](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/reconstruction.py#L110-L200)

1. `SquareROI` được đọc từ `reconstruction_input_roi` (đã tính ở Bước 3).
2. `source_crop = crop_image(image, roi)` — RGB gốc trong khung vuông crop.
3. `mask_crop = crop_array(generation_mask, roi)` — mask inpainting cho HD-Painter.
4. `mask_image = Image.fromarray(mask_crop * 255, mode="L")` — mask PIL grayscale.
5. Prompt văn bản được tạo từ template:
   ```
   "Reconstruct only the hidden continuation of the {target} behind {occluders}, …"
   ```
   với `style_hint` được nối thêm nếu đã cấu hình (ví dụ: `"Match the source's flat vector illustration style…"`).
   → [reconstruction.py#L180-L188](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/reconstruction.py#L180-L188)

### Bước 4b — Suy luận model

Nếu `reconstruct_many` khả dụng, tất cả item đã chuẩn bị được gọi batch trong một
lần. Nếu không, mỗi item gọi `reconstruct(source_crop, mask_image, prompt)` riêng
lẻ. Exception được bắt riêng từng item và lưu dưới dạng failure.
→ [reconstruction.py#L402-L433](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/reconstruction.py#L402-L433)

### Bước 4c — Validation đầu ra & Phân tách Canvas

Mỗi kết quả `reconstructed` trực tiếp từ HD-Painter đi qua `validate_reconstruction_result`:
→ [validate_reconstruction.py#L181-L268](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/validate_reconstruction.py#L181-L268)
→ Gọi tại: [reconstruction.py#L469-L478](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/reconstruction.py#L469-L478)

1. **Kiểm tra an toàn cấu trúc (Sanity Check):**
   - Kiểm tra kiểu dữ liệu (`PIL.Image`), kích thước crop (`source_crop.size`), mode (`RGB/RGBA/L`).
   - **Phát hiện đầu ra cực đoan:** Nếu vùng completion hole bị toàn đen/toàn trắng bất thường, validation ném lỗi và hủy bỏ đối tượng khôi phục này.

2. **Tạo biến `validated` (Khôi phục pixel ngoài vùng permitted):**
   - Pixel nằm ngoài vùng `permitted` (`accepted_rgb_mask` hoặc `hard_mask` đã mở rộng) được **ghi đè bằng pixel source gốc**.
   - Việc này sửa lỗi Poisson blending / super-resolution của HD-Painter rò rỉ màu ra ngoài mask.

### Bước 4d — Phân tách 2 luồng Canvas (Raw vs Validated)

Sau khi kiểm tra an toàn thành công, đối tượng được lưu 2 phiên bản canvas riêng biệt:
→ [reconstruction.py#L479-L521](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/reconstruction.py#L479-L521)

```python
raw_reconstructed = reconstructed.convert("RGB")
detected.raw_reconstruction_canvas = raw_reconstructed   # 1. RGB THÔ trực tiếp từ HD-Painter (CHƯA pixel restoration)
detected.reconstruction_canvas = validated                # 2. Đầu ra đã qua khôi phục pixel ngoài vùng permitted
detected.reconstruction_roi = item.roi
```

> **ĐIỂM MẤU CHỐT VỀ WORKFLOW:**
> - `raw_reconstruction_canvas`: Lưu ảnh RGB **THÔ NGUYÊN BẢN trực tiếp từ model HD-Painter** (chưa qua bước khôi phục pixel / cắt xén cứng nào). **ĐÂY LÀ ĐẦU VÀO TRỰC TIẾP CHO BIREFNET Ở BƯỚC 5**.
> - `reconstruction_canvas`: Chỉ lưu phiên bản đã khôi phục pixel để làm phương án fallback an toàn nếu BiRefNet bị tắt.
> 
> Việc đưa **Raw Canvas trực tiếp vào BiRefNet** giúp BiRefNet có góc nhìn đầy đủ và tự nhiên nhất về toàn bộ phần nét vẽ foreground mà HD-Painter đã sinh ra, từ đó tự bóc tách alpha mềm một cách mượt mà nhất mà không bị viền vỡ do cắt xén cứng từ trước.

### Bước 4e — Chỉ số màu sắc (Color Metrics)

`reconstruction_color_metrics` tính các chỉ số tư vấn (không loại cứng):
→ [validate_reconstruction.py#L61-L116](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/validate_reconstruction.py#L61-L116)
→ Gọi tại: [reconstruction.py#L486-L503](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/reconstruction/reconstruction.py#L486-L503)

- `target_lab_distance`: trung vị khoảng cách CIE-Lab từ pixel sinh đến bảng màu target nhìn thấy.
- `occluder_lab_distance`: trung vị khoảng cách CIE-Lab đến bảng màu occluder.
- `source_persistence_ratio`: tỷ lệ pixel sinh gần như không đổi so với source.
- `generated_detail_ratio`: tỷ lệ năng lượng chi tiết Laplacian giữa vùng sinh và vùng nhìn thấy.

---

## Bước 5 — Tinh chỉnh Support khôi phục (BiRefNet lần 1)

| Thông tin | Chi tiết |
|-----------|----------|
| **Module** | [matting.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py) |
| **Hàm chính** | [refine_reconstruction_supports](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L75-L409) |
| **Model** | BiRefNet (`ZhengPeng7/BiRefNet`) — dùng chung với Bước 7 (tải một lần) |
| **Điều phối tại** | [orchestrator.py#L530-L600](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L530-L600) |

Bước này chạy khi `has_accepted_reconstruction` là true và
`support_refinement_enabled` không bị tắt.

### Mục đích

BiRefNet là bộ tách foreground/background bất khả tri lớp (class-agnostic). Nó nhận
**đầu ra thô HD-Painter** (`raw_reconstruction_canvas`) và tạo ra bản đồ alpha mềm.
Mục tiêu:

1. Tìm thành phần foreground nào thực sự thuộc target.
2. Loại bỏ thành phần lạc (foreign objects, nhiễu nền).
3. Tạo `reconstruction_write_alpha` mềm để blend mượt.

### Luồng xử lý

Với mỗi đối tượng có `reconstruction_mask` không rỗng và `raw_reconstruction_canvas` hợp lệ:

#### 5a — Suy luận BiRefNet trên đầu ra thô HD-Painter

```python
raw_canvas = target.raw_reconstruction_canvas  # KHÔNG phải reconstruction_canvas
alpha_crop = _resize_alpha(matte(raw_canvas.convert("RGB")), roi.size)
```

BiRefNet nhận **canvas thô chưa cắt, chưa khôi phục** để nhìn thấy toàn bộ
foreground mà HD-Painter đã sinh, kể cả phần mà pixel restoration đã ghi đè.
→ [matting.py#L127-L131](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L127-L131)
→ [matting.py#L178-L181](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L178-L181)

#### 5b — Chuẩn bị mask không gian trong toạ độ crop

→ [matting.py#L205-L235](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L205-L235)

- `modal_crop`: modal mask của target trong ROI crop
- `foreign_protection_crop`: `reconstruction_foreign_protection_mask` (foreign outside bbox + protected foreign inside) trong toạ độ crop
- `generation_evidence`: `reconstruction_generation_mask` nở bởi `generation_evidence_margin_pixels`
- `connection_anchor`: target modal nở bởi `connection_margin_pixels` (mặc định `4px`) — "neo" kiểm tra kết nối

#### 5c — Xây dựng candidate foreground

→ [matting.py#L241-L254](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L241-L254)

```python
candidate = (alpha_crop >= alpha_low_threshold)   # mặc định 0.35
             & valid_roi_crop
             & ~foreign_protection_crop
```

Tính thêm:
- `strong_foreground = alpha_crop >= alpha_high_threshold` (mặc định `0.7`)
- `change_distance = mean(|raw_rgb - source_rgb|, axis=2)` — thay đổi RGB per-pixel
- `changed_by_model = change_distance >= change_threshold` (mặc định `8.0`)

#### 5d — Lọc connected component

→ [matting.py#L256-L288](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L256-L288)

`cv2.connectedComponentsWithStats(candidate, connectivity=8)` chia candidate thành
các thành phần cô lập. Mỗi thành phần **chỉ được chấp nhận khi TẤT CẢ điều kiện
đều thoả mãn**:

| # | Điều kiện | Mục đích | Code |
|---|-----------|----------|------|
| 1 | `np.any(component & connection_anchor)` | Thành phần chạm vào target modal đã nở | [#L274](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L274) |
| 2 | `np.any(component & strong_foreground)` | Ít nhất vài pixel có độ tin cậy BiRefNet cao | [#L276](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L276) |
| 3 | `np.any(extension_part & changed_by_model)` | HD-Painter thực sự đã thay đổi pixel ở vùng này | [#L278](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L278) |
| 4 | `np.any(extension_part & generation_evidence)` | Thành phần chồng lấn vùng generation mask (nếu `require_generation_evidence=true`) | [#L280-L284](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L280-L284) |
| 5 | `extension_area <= max_extension_area` | Extension không vượt `modal_area × max_extension_area_ratio` (mặc định `2.0`) | [#L285](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L285) |
| 6 | `component_area >= min_component_area_pixels` | Không phải đốm nhiễu nhỏ (mặc định `8px`) | [#L272](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L272) |

`extension_part = component & ~modal_crop` — chỉ phần nằm ngoài target nhìn thấy.

#### 5e — Tạo soft extension alpha

→ [matting.py#L290-L313](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L290-L313)

```python
extension_support = accepted & ~modal_crop & ~foreign_protection_crop & valid_roi_crop
extension_alpha = alpha_crop * extension_support.astype(float64)
```

Feathering tuỳ chọn: nếu `alpha_feather_pixels > 0`, Gaussian blur được áp dụng
để làm mượt biên. Sau feathering, các mask bảo vệ được áp lại:

```python
extension_alpha[modal_crop] = 0.0
extension_alpha[foreign_protection_crop] = 0.0
extension_alpha[~valid_roi_crop] = 0.0
```

#### 5f — Khôi phục về mảng full-image

→ [matting.py#L315-L322](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L315-L322)

```python
target.reconstruction_evidence_alpha = restore_array(alpha_crop, roi)        # alpha BiRefNet thô
target.reconstruction_write_alpha    = restore_array(extension_alpha, roi)    # trọng số blend mềm
target.reconstruction_write_mask     = extension_full                         # nhị phân: alpha > epsilon
target.reconstruction_extension_mask = extension_full                         # giống write_mask
target.reconstruction_support_mask   = modal | extension_full                 # support tổng hợp
```

#### 5g — Tạo reconstruction_canvas an toàn

→ [matting.py#L324-L334](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L324-L334)

```python
composed_crop = raw_rgb * extension_alpha + source_rgb * (1.0 - extension_alpha)
```

Nếu extension không rỗng, `target.reconstruction_canvas` được đặt thành raw canvas
RGB (phép blend soft-alpha thực tế được trì hoãn tới group composition ở Bước 6).
Nếu extension rỗng, `reconstruction_canvas` đặt là `None` để báo hiệu không có
khôi phục khả dụng.

---

## Bước 6 — Gom nhóm & Hợp thành nguồn (Grouping & Source Composition)

| Thông tin | Chi tiết |
|-----------|----------|
| **Module** | [grouping.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/grouping.py) |
| **Các hàm** | [group_reconstructed_objects](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/grouping.py#L33-L190), [compose_group_sources](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/grouping.py#L193-L418) |
| **Model** | Không |
| **Điều phối tại** | [orchestrator.py#L602-L613](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L602-L613) |

### Bước 6a — Hình thành nhóm (Group Formation)

`group_reconstructed_objects(objects, pair_decisions)` hợp nhất các đối tượng thô
thành các `GroupedObject` bằng **union-find**:
→ [grouping.py#L33-L138](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/grouping.py#L33-L138)

- Hai đối tượng được gộp nếu cùng **`semantic_class`** VÀ **bounding box modal gốc
  chồng lấn nhau**.
- Đối tượng khác class không bao giờ được gộp (chúng chỉ được ghép cặp cho phân
  tích sâu).

Mỗi `GroupedObject` lưu:
- `members`: tuple có thứ tự các `DetectedObject` thành viên
- `modal_mask`: hợp tất cả modal mask thành viên (uint8 × 255)
- `amodal_mask`: hợp tất cả amodal mask thành viên
- `effective_support_mask`: property trả về `modal | reconstruction_support` mỗi thành viên
  → [types.py#L126-L139](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/types.py#L126-L139)

### Bước 6b — Sắp xếp nhóm theo độ sâu (Depth-Aware Group Ordering)

Khi có `pair_decisions`, các nhóm được sắp topo từ sau ra trước (back-to-front)
dùng thứ tự sâu từ Bước 3. Chu trình được xử lý bằng fallback thứ tự segmentation
ổn định.
→ [grouping.py#L138-L190](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/grouping.py#L138-L190)

### Bước 6c — Hợp thành nguồn (Source Composition)

`compose_group_sources(image, groups)` tạo canvas RGB đã giải quyết xung đột cho
mỗi nhóm:
→ [grouping.py#L193-L418](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/grouping.py#L193-L418)

1. ROI được tính từ `group.effective_support_mask` với `context_ratio=0.0`.
2. `composed` được khởi tạo từ crop source gốc.
3. `protected_modal` đánh dấu pixel modal nhìn thấy tổng hợp của nhóm — **không bao
   giờ bị ghi đè**.

Với mỗi thành viên có `reconstruction_canvas`:
→ [grouping.py#L240-L404](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/grouping.py#L240-L404)

4. Vùng chồng lấn giữa ROI nhóm và ROI khôi phục thành viên được tính.
5. `permitted` = `reconstruction_write_mask` — chỉ pixel mà model được phép khôi phục.
6. `permitted_alpha` = `reconstruction_write_alpha` clamp về `[0.0, 1.0]` — trọng số blend mềm.
7. `writable = permitted & ~protected_modal & ~filled_reconstruction` — loại trừ pixel modal-protected và pixel đã ghi bởi thành viên trước.
8. **Blend có alpha:**
   → [grouping.py#L378-L385](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/grouping.py#L378-L385)
   ```python
   effective_alpha = where(writable, permitted_alpha, 0.0)
   blended = candidate * effective_alpha + destination * (1 - effective_alpha)
   destination[writable] = blended[writable]
   ```
9. `filled_reconstruction` được cập nhật để thành viên sau không ghi đè (xác định: khôi phục đầu tiên thắng).
10. Xung đột giữa các khôi phục thành viên chồng nhau được ghi log.

Kết quả:
- `group.composed_source` — ảnh RGB hợp thành cuối cùng cho nhóm
- `group.composed_roi` — ROI dùng để hợp thành
- `group.reconstruction_conflicts` — các cặp xung đột đã ghi

---

## Bước 7 — Matting nhóm cuối cùng (BiRefNet lần 2)

| Thông tin | Chi tiết |
|-----------|----------|
| **Module** | [matting.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py) |
| **Hàm chính** | [refine_objects](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L475-L584) |
| **Model** | BiRefNet (cùng instance với Bước 5) |
| **GPU giải phóng** | Có — [orchestrator.py#L626-L629](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L626-L629) sau cả Bước 5 và 7 |
| **Điều phối tại** | [orchestrator.py#L615-L625](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L615-L625) |

### Mục đích

Lần chạy BiRefNet thứ hai tạo ra alpha RGBA mềm chất lượng cao cuối cùng cho mỗi
nhóm, hoạt động trên **RGB đã hợp thành** (bao gồm cả pixel gốc nhìn thấy và
phần extension đã khôi phục từ Bước 6).

### Luồng xử lý

Với mỗi `GroupedObject`:

1. **Chọn nguồn:**
   → [matting.py#L493-L519](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L493-L519)
   - **Nhóm đã khôi phục**: Dùng `_expand_composed_source` lấy `composed_source`
     từ Bước 6, nhúng nó vào matting ROI lớn hơn, và bổ sung ngữ cảnh xung quanh
     từ ảnh gốc.
     → [matting.py#L412-L444](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L412-L444)
   - **Nhóm không khôi phục**: Dùng crop ảnh source gốc trực tiếp.
   - Support mask: `effective_support_mask` cho nhóm đã khôi phục,
     `modal_mask > 0` cho nhóm đơn giản.

2. **ROI**: `square_roi_from_support(support, context_ratio)` với `context_ratio`
   từ `pipeline.matting.context_ratio` (mặc định `0.025`).

3. **Suy luận BiRefNet**: `alpha = matte(source_crop)` trả về tensor 2D.
   Resize về kích thước ROI nếu cần.
   → [matting.py#L543-L560](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L543-L560)

4. **Lọc alpha:**
   → [matting.py#L561-L569](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L561-L569)
   ```python
   valid_support = cv2.dilate(support_crop, dilation_kernel)  # support_dilation_pixels (mặc định 2)
   alpha_crop[(alpha_crop <= THRESHOLD_ALPHA) | ~valid_support] = 0.0
   ```
   Pixel dưới `THRESHOLD_ALPHA` (`0.005`) hoặc ngoài support đã nở bị zero —
   ngăn BiRefNet chiếm foreground không liên quan.

5. **Đầu ra**: `group.soft_alpha = restore_array(alpha_crop, roi)` — alpha mềm
   full-image, `group.matting_source` và `group.matting_roi` được lưu cho tách lớp.
   → [matting.py#L571-L573](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/matting.py#L571-L573)

---

## Bước 8 — Tách lớp & Inpaint nền

| Thông tin | Chi tiết |
|-----------|----------|
| **Module** | [layers.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/layers.py) (lớp), [background.py](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/background.py) (nền) |
| **Các hàm** | [extract_object_layers](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/layers.py#L22-L181), [generate_final_background](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/background.py#L74-L103) |
| **Model** | LaMa hoặc SDXL (background inpainting) |
| **GPU giải phóng** | Có — [orchestrator.py#L654-L657](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L654-L657) |
| **Điều phối tại** | [orchestrator.py#L631-L657](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/orchestrator.py#L631-L657) |

### Bước 8a — Tách lớp (Layer Extraction)

`extract_object_layers(objects, kernel_size, background_inpaint)`:
→ [layers.py#L22-L181](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/layers.py#L22-L181)

Với mỗi nhóm có `soft_alpha` hợp lệ:

1. `source_rgb` = `matting_source` từ Bước 7 (chính xác RGB mà BiRefNet đã thấy).
2. `hard_mask = support_crop | (alpha > THRESHOLD_ALPHA)` — hợp support và vùng alpha.
3. **Nền tạm per-component**: `build_inpaint_mask` tạo mask, rồi model inpainting
   lấp vùng foreground để tạo `component_background`. `refine_background` sửa màu
   vùng inpaint.
   → [layers.py#L87-L110](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/layers.py#L87-L110)
4. **Tách nền foreground (Unblending)**: `refine_alpha_with_colors(source_rgb, background_rgb, alpha, hard_mask, kernel_size)`:
   → [layers.py#L111-L118](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/layers.py#L111-L118)
   - Dùng nền ước lượng để giải cho màu foreground thực:
     `F = (composite - B×(1-α)) / α`
   - Tinh chỉnh giá trị alpha tại biên bán trong suốt nơi màu foreground và
     background hoà trộn.
5. **Crop về bbox chặt**: Pixel padding ngoài vùng ảnh thực bị zero. Ảnh RGBA
   được crop về bounding box chặt.
   → [layers.py#L120-L136](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/layers.py#L120-L136)
6. **Đầu ra**: `ObjectLayer(keyword, png_base64, x, y, width, height)` với `x` và
   `y` là toạ độ canvas toàn cục.
   → [layers.py#L155-L164](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/layers.py#L155-L164)
   → Kiểu dữ liệu: [types.py#L18-L25](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/types.py#L18-L25)

### Bước 8b — Inpaint nền (Background Inpainting)

`generate_final_background(image, final_groups, kernel_size, background_inpaint)`:
→ [background.py#L74-L103](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/background.py#L74-L103)
→ Gọi nội bộ: [generate_background_from_masks](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/background.py#L30-L71)

1. Hợp tất cả **modal mask nhìn thấy** của nhóm (không phải amodal — phần ẩn không
   được ảnh hưởng nền).
2. Thêm vùng alpha mềm mở rộng nhẹ qua visible modal (qua `_visible_soft_alpha`
   với `VISIBLE_ALPHA_DILATION = (3, 3)`).
   → [background.py#L19-L27](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/background.py#L19-L27)
3. Mở rộng union mask bằng `kernel_size` để xoá bóng viền và rò rỉ màu.
4. Chạy inpainting nền một lần trên union đã mở rộng.
5. `refine_background` sửa màu vùng inpaint.

---

## Phản hồi API

Pipeline trả về `ProcessResult`:
→ [types.py#L28-L34](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/types.py#L28-L34)

| Field | Kiểu | Mô tả |
|-------|------|-------|
| `background_base64` | `str` | PNG nền sạch mã hoá Base64 |
| `original_width` | `int` | Chiều rộng ảnh nguồn |
| `original_height` | `int` | Chiều cao ảnh nguồn |
| `layers` | `list[ObjectLayer]` | Các lớp foreground RGBA |
| `diagnostics` | `PipelineDiagnostics \| None` | Thời gian stage, bộ nhớ GPU đỉnh, quyết định cặp |

Mỗi `ObjectLayer`:
→ [types.py#L18-L25](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/pipeline/types.py#L18-L25)

| Field | Kiểu | Mô tả |
|-------|------|-------|
| `keyword` | `str` | Text prompt (display label) |
| `png_base64` | `str` | PNG RGBA mã hoá Base64 (đã crop) |
| `x`, `y` | `int` | Vị trí canvas toàn cục của crop |
| `width`, `height` | `int` | Kích thước crop |

---

## Cài đặt, Weights & Cấu hình

### Phụ thuộc

```powershell
pip install -e .
pip install -r backend/requirements.txt
```

### Checkpoints model

| Model | Checkpoint / ID | Tải tự động |
|-------|-----------------|-------------|
| SAM3 | `checkpoints/sam3.pt` | HF gated (đặt `load_from_hf: true`) |
| SDAmodal | `backend/models/completion/checkpoints/SDAmodal.pth` | Không |
| HD-Painter | `ds8_inp` (qua HF) | Có (`auto_download: true`) |
| BiRefNet | `ZhengPeng7/BiRefNet` | Có (cache khi dùng lần đầu) |
| SimpleLaMa | Tích hợp sẵn | Có (cache khi dùng lần đầu) |
| SDXL (tuỳ chọn) | `model_id` tuỳ cấu hình | Có |

### Cấu hình chính (`backend/config.yaml`)

→ [config.yaml](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml)

| Section | Các tham số chính |
|---------|-------------------|
| `pipeline.completion` | `max_area_growth_ratio`, `max_bbox_growth_ratio`, `minimum_hole_area_pixels`, `minimum_hole_area_ratio`, `tie_tolerance_ratio` — [#L7-L12](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L7-L12) |
| `pipeline.object_reconstruction` | `context_ratio`, `generation_mask_dilation_pixels`, `generation_mask_closing_pixels`, `composition_margin_pixels`, `blend_allowance_ratio`, `support_alpha_low_threshold`, `support_alpha_high_threshold`, `support_change_threshold`, `prompt_template`, `style_hint`, `diagnostics_directory` — [#L13-L52](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L13-L52) |
| `pipeline.matting` | `context_ratio`, `support_dilation_pixels` — [#L53-L55](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L53-L55) |
| `pipeline.diagnostics` | `enabled`, `log_summary`, `track_peak_gpu_memory` — [#L56-L59](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L56-L59) |
| `models.segmentation.sam3` | `checkpoint_path`, `confidence_threshold`, `load_from_hf` — [#L118-L123](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L118-L123) |
| `models.completion.sdamodal` | `config_path`, `checkpoint_path`, `input_size`, `enlarge_box` — [#L124-L131](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L124-L131) |
| `models.object_reconstruction.hd_painter` | `model_id`, `method`, `input_size`, `num_steps`, `guidance_scale`, `fp16`, `sequential_cpu_offload`, `super_resolution.*` — [#L67-L97](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L67-L97) |
| `models.matting.birefnet` | `model_id`, `trust_remote_code`, `resolution` — [#L61-L66](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L61-L66) |
| `models.background_inpainting.lama` | `generation_mask_expansion`, `composition_mask_expansion`, `feather_radius` — [#L98-L105](file:///d:/Documents/Qikify/Magic_Layer_v1/backend/config.yaml#L98-L105) |

### Chạy

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
