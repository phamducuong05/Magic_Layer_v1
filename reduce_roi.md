# Reduce Reconstruction ROI and Use Full HD-Painter Output

## 1. Mục tiêu

Thay đổi object-reconstruction pipeline theo các nguyên tắc sau:

1. Giảm `object_reconstruction.context_ratio` từ `0.15` xuống `0.025`.
2. Amodal completion chỉ còn đóng vai trò:
   - Phát hiện object nào có khả năng bị che và cần reconstruction.
   - Cung cấp bounding box ước lượng của toàn bộ target.
   - Hỗ trợ xác định directional overlap và depth order.
3. Giữ nguyên hoàn toàn công thức tạo `reconstruction_generation_mask` hiện tại để điều khiển vùng HD-Painter được yêu cầu inpaint.
4. Sau khi HD-Painter hoàn tất, không giới hạn RGB được chấp nhận bằng `reconstruction_generation_mask`, `amodal_mask`, `completion_hole` hoặc `reconstruction_mask`; thay vào đó, lấy toàn bộ output hợp lệ trong ROI rồi loại các modal pixel cần bảo vệ.
5. Modal mask của chính target luôn được giữ nguyên từ ảnh gốc.
6. Modal mask của object khác nằm trong bounding box của target được xem là vùng che target và phải được inpaint.
7. Modal mask của object khác nằm ngoài bounding box target nhưng vẫn xuất hiện trong ROI được bảo vệ và không bị HD-Painter ghi đè.
8. Không ghi hoặc validate các pixel padding nằm ngoài ảnh gốc.

## 2. Quyết định hình học

### 2.1. Target bounding box

`target_bbox` sẽ là tight bounding box tính từ `validated amodal_mask` của target.

Lý do:

- Bounding box modal ban đầu chỉ bao phủ phần target đang nhìn thấy.
- Phần target bị che có thể mở rộng ra ngoài modal bounding box.
- Validated amodal mask vẫn hữu ích để ước lượng phạm vi không gian của target, dù không còn được dùng làm hard write mask.
- Nếu amodal completion thất bại hoặc không được validation, object đó không đủ điều kiện chạy reconstruction theo workflow hiện tại.

`target_bbox_mask` là boolean mask full-image:

```text
target_bbox_mask[pixel] = True
```

khi pixel nằm trong `target_bbox`.

### 2.2. Reconstruction ROI

ROI tiếp tục là square ROI bao quanh validated amodal support, nhưng dùng:

```text
context_ratio = 0.025
```

Kích thước ROI:

```text
roi_size = ceil(max(amodal_bbox_width, amodal_bbox_height) * 1.05)
```

ROI vẫn có thể đi ra ngoài biên ảnh để giữ hình vuông và tâm target. Các pixel đó là padding, không được đưa vào write-back.

### 2.3. Các loại modal mask

Với target đang được reconstruction:

#### A. `target_modal_mask`

Modal mask của chính target.

Quy tắc:

- Không inpaint.
- Luôn giữ RGB từ ảnh gốc.
- Luôn là RGB nguồn được bảo vệ, không qua bước chỉnh màu hậu kỳ.
- Không được HD-Painter hoặc BiRefNet extension ghi đè.

#### B. `foreign_modal_inside_bbox`

Pixel modal của tất cả object khác nằm trong `target_bbox`.

```text
foreign_modal_inside_bbox =
    union(other.modal_mask)
    AND target_bbox_mask
```

Quy tắc:

- Được coi là vùng có khả năng đang che target.
- Không tự động thay đổi `generation_mask`; việc HD-Painter inpaint vùng nào vẫn theo công thức generation mask hiện tại.
- Sau inference, output HD-Painter tại đây được phép trở thành RGB ẩn của target và không bị phục hồi bằng source RGB.
- Không yêu cầu toàn bộ object kia phải nằm trong bounding box; chỉ phần giao với bounding box mới được inpaint.

#### C. `foreign_modal_outside_bbox`

Pixel modal của object khác nằm trong ROI nhưng ở ngoài `target_bbox`.

```text
foreign_modal_outside_bbox =
    union(other.modal_mask)
    AND roi_real_pixels
    AND NOT target_bbox_mask
```

Quy tắc:

- Được xem là visual context hoặc object không liên quan.
- Không làm thay đổi công thức generation mask hiện tại.
- Luôn phục hồi RGB gốc sau inference, kể cả khi dilation hoặc blending của model đã tác động tới vùng này.
- Không được sử dụng làm reconstruction support của target.

#### D. `non_modal_roi_pixels`

Pixel thật trong ROI nhưng không thuộc modal mask của bất kỳ object nào.

Quy tắc:

- Không tự động được thêm vào generation mask.
- Output HD-Painter được giữ lại sau inference, kể cả khi pixel nằm ngoài generation mask, để kết quả cuối không bị giới hạn bởi amodal/completion mask.
- BiRefNet ở downstream sẽ xác định pixel nào thực sự thuộc foreground target.

## 3. Mask contract mới

### 3.1. Detection masks

Các mask sau vẫn được xây và giữ nguyên mục đích:

- `completion_hole_mask`
- `reconstruction_seed_mask`
- directional reconstruction mask
- `reconstruction_mask`
- `reconstruction_occluder_mask`

Chúng chỉ dùng để:

- Chứng minh target có vùng bị che hợp lệ.
- Quyết định có chạy reconstruction hay không.
- Xác định hướng `(occluded, occluder)`.
- Tính depth order.
- Xây prompt.
- Chạy validation trên vùng hidden seed có ý nghĩa.
- Xuất diagnostics.

Chúng không còn giới hạn toàn bộ vùng RGB được lấy từ HD-Painter.

### 3.2. Protected mask

```text
reconstruction_protected_mask =
    target_modal_mask
    OR foreign_modal_outside_bbox
    OR roi_padding_pixels
```

### 3.3. Generation mask giữ nguyên

Không refactor công thức `reconstruction_generation_mask`. Pipeline tiếp tục tạo mask theo logic hiện tại:

```text
composition_mask =
    directional completion-hole components
    đã được neo vào overlap seed

relevant_occluder =
    union(assigned occluder modal masks)
    AND roi_real_pixels
    AND NOT target_modal_mask

generation_seed =
    composition_mask
    OR relevant_occluder

reconstruction_generation_mask =
    close(generation_seed)
    → dilate(...)
    → intersect roi_real_pixels
    → remove target_modal_mask
    → restore composition_mask
```

Các tham số sau vẫn giữ nguyên tác dụng:

- `generation_mask_closing_pixels`
- `generation_mask_dilation_pixels`
- `composition_margin_pixels`
- `support_margin_pixels`

Modal classification theo bounding box không được dùng để mở rộng, thu hẹp hoặc thay thế generation mask.

### 3.4. Accepted model RGB mask

Sau khi output HD-Painter vượt qua validation:

```text
accepted_model_rgb_mask =
    roi_real_pixels
    AND NOT target_modal_mask
    AND NOT foreign_modal_outside_bbox
```

Đây là mask dùng để lấy RGB từ reconstruction canvas sau khi HD-Painter đã hoàn tất. Mask này có thể lớn hơn `reconstruction_generation_mask`.

Không intersect mask này với:

- `amodal_mask`
- `completion_hole_mask`
- `reconstruction_mask`
- `reconstruction_generation_mask`

### 3.5. Reconstruction write mask

```text
reconstruction_write_mask = accepted_model_rgb_mask
```

Mask này được dùng khi compose reconstructed member vào same-class group.

## 4. Data flow mới

```text
SAM3 raw modal masks
    ↓
Cross-class bounding-box overlap
    ↓
Amodal completion cho các object liên quan
    ↓
Validate amodal completion
    ↓
Directional overlap + completion-hole area
    ↓
Depth order và quyết định object cần reconstruction
    ↓
Tạo amodal target_bbox
    ↓
Tạo square ROI với context_ratio = 0.025
    ↓
Tạo generation mask theo đúng công thức hiện tại
    ↓
HD-Painter inference
    ↓
Phân loại modal pixel theo vị trí trong target_bbox
    ↓
Validation trên directional hidden region
    ↓
Khôi phục target modal + protected foreign modal từ source
    ↓
Giữ toàn bộ model RGB trong accepted_model_rgb_mask
    ↓
BiRefNet tìm foreground support từ reconstructed RGB
    ↓
Same-class grouping + conflict resolution
    ↓
Final group matting và layer extraction
```

## 5. Thay đổi theo file

### 5.1. `backend/config.yaml`

Thay đổi:

```yaml
pipeline:
  object_reconstruction:
    context_ratio: 0.025
```

Không thay đổi `matting.context_ratio`, vì đây là ROI riêng của final group matting.

### 5.2. `backend/pipeline/types.py`

Bổ sung các field diagnostics/data-contract cần thiết cho mỗi `DetectedObject`:

- `reconstruction_target_bbox_mask`
- `reconstruction_foreign_modal_inside_bbox`
- `reconstruction_foreign_modal_outside_bbox`
- `reconstruction_protected_mask`

Các field đều là `Optional[np.ndarray]`, full-image boolean mask.

Mục đích:

- Không phải tính lại classification ở nhiều stage.
- Cho validation, support refinement, grouping và diagnostics dùng cùng một kết quả.
- Tránh mỗi stage hiểu “foreign modal” theo một cách khác nhau.

### 5.3. `backend/pipeline/reconstruction/mask_preparation.py`

#### Thay đổi ROI

- Tiếp tục xây ROI từ validated amodal support.
- Nhận `context_ratio=0.025` từ config.
- Tạo `roi_real_pixels` bằng phần giao giữa ROI và ảnh gốc.

#### Thêm modal classification cho post-inference acceptance

Với từng target:

1. Tạo `target_bbox_mask` từ tight bbox của validated amodal mask.
2. Tạo `other_modal_union` từ modal mask của mọi object khác target.
3. Tạo:

```text
foreign_modal_inside_bbox =
    other_modal_union & target_bbox_mask & roi_real_pixels
```

4. Tạo:

```text
foreign_modal_outside_bbox =
    other_modal_union & ~target_bbox_mask & roi_real_pixels
```

5. Tạo protected mask dùng sau HD-Painter:

```text
protected =
    target_modal | foreign_modal_outside_bbox
```

6. Tạo accepted/write region:

```text
accepted_model_rgb_mask =
    roi_real_pixels & ~protected
```

#### Vai trò morphology

- Directional reconstruction mask vẫn có thể dùng closing/dilation để tìm hidden seed ổn định.
- Closing/dilation vẫn quyết định `reconstruction_generation_mask` đúng như code hiện tại.
- Modal classification không tham gia vào quá trình tạo generation mask.
- Sau inference, accepted RGB mask mới là vùng rộng hơn và độc lập với generation mask.

#### Điều kiện skip

Target vẫn bị skip nếu:

- Không có assigned reconstruction direction.
- Không có validated amodal mask.
- Không có meaningful completion hole.
- Không có directional reconstruction seed.
- Generation mask hiện tại rỗng.

### 5.4. `backend/pipeline/reconstruction/reconstruction.py`

#### Input HD-Painter

Giữ:

- `source_crop`: RGB crop của reconstruction ROI.
- `mask_image`: crop của `reconstruction_generation_mask`.
- Prompt dựa trên target class và occluder classes.

Thay đổi:

- `mask_image` giữ nguyên, tiếp tục là crop của generation mask hiện tại.
- `composition_crop` vẫn là directional reconstruction mask và được dùng làm validation seed.
- Thêm `accepted_rgb_crop` lấy từ `accepted_model_rgb_mask`, không lấy từ generation mask.
- `accepted_rgb_crop` chỉ được dùng sau khi model trả kết quả để validation và write-back.

#### Output acceptance

Sau inference:

1. Kiểm tra output type, mode và size.
2. Kiểm tra directional completion seed không rỗng.
3. Kiểm tra output tại hidden seed không phải blank extreme không hợp lệ.
4. Phục hồi source RGB tại:
   - Target modal.
   - Foreign modal ngoài target bbox.
   - Padding ngoài ảnh.
5. Giữ model RGB tại toàn bộ accepted RGB mask, kể cả phần nằm ngoài generation mask.

### 5.5. `backend/pipeline/reconstruction/validate_reconstruction.py`

Đổi tên/ý nghĩa tham số trong nội bộ để phân biệt:

- `hard_mask`: generation mask gửi vào model; giữ nguyên công thức và chỉ mô tả model được yêu cầu inpaint ở đâu.
- `validation_seed`: directional hidden region dùng để kiểm tra chất lượng tối thiểu.
- `protected_mask`: pixel bắt buộc giữ source.
- `accepted_model_rgb_mask`: vùng được phép lấy từ output sau inference, có thể rộng hơn `hard_mask`.

Validation contract:

```text
validated_output[pixel] =
    model_output[pixel], nếu pixel thuộc accepted_model_rgb_mask
    source_crop[pixel], trong các trường hợp còn lại
```

Vẫn giữ:

- Size restoration validation.
- Channel validation.
- Real-image pixel validation.
- Blank black/white extreme validation tại directional hidden seed.

Bỏ sự phụ thuộc giữa write-back area và amodal/completion shape.

### 5.6. Không chỉnh màu RGB sau reconstruction

`validated_output` được dùng trực tiếp làm `reconstruction_canvas`.

- Không gọi `refine_reconstruction_colors()`.
- Không dùng `color_refinement_enabled` hoặc `color_refinement_strength`.
- Không tạo `color_refined_output.png`.
- Color metrics vẫn được giữ để đo và log, nhưng không thay đổi pixel.

### 5.7. `backend/pipeline/matting.py`

`refine_reconstruction_supports()` không được kéo support trở lại giới hạn amodal cũ.

#### Candidate region

```text
birefnet_candidate =
    alpha >= low_threshold
    AND accepted_model_rgb_mask
```

Không intersect candidate với amodal mask.

#### Core

```text
target_core = target_modal_mask
```

Validated amodal mask có thể giữ trong diagnostics nhưng không phải hard support.

#### Accepted foreground support

BiRefNet component được chấp nhận khi:

- Nằm trong accepted model RGB region.
- Có pixel đạt high alpha threshold.
- Có thay đổi RGB thực sự so với source.
- Kết nối trực tiếp hoặc qua configured connection margin với target core.

Không nhận:

- Padding.
- Foreign modal ngoài target bbox.
- Target modal dưới dạng extension mới.

#### Area growth

Không giới hạn extension bằng diện tích amodal mask.

Nếu vẫn cần safety cap, cap phải dựa trên diện tích ROI hoặc target bbox thay vì amodal area. Mặc định kế hoạch này ưu tiên bỏ amodal-based cap để đúng mục tiêu không phụ thuộc vào chất lượng amodal mask.

#### Final masks

```text
reconstruction_write_mask = accepted_model_rgb_mask

reconstruction_support_mask =
    target_modal_mask
    OR accepted_birefnet_foreground
```

Ý nghĩa:

- Write mask quyết định RGB nào có sẵn để compose.
- Support mask quyết định khu vực nào có khả năng thuộc target và được đưa sang final group matting.
- Không coi toàn bộ ROI là foreground target.

### 5.8. `backend/pipeline/grouping.py`

Giữ conflict policy hiện tại:

- Modal source của member cùng group được ưu tiên.
- Reconstruction đã ghi trước thắng khi hai reconstructed RGB overlap.

Thay đổi nguồn mask:

- Compose RGB bằng `reconstruction_write_mask`.
- Group ROI và final matting support dùng `reconstruction_support_mask`.

Kết quả:

- Toàn bộ RGB hợp lệ từ HD-Painter có thể đi vào composed source.
- BiRefNet support ngăn toàn bộ ROI bị coi là object.
- Same-class grouping không phụ thuộc vào amodal shape để lấy RGB reconstructed.

## 6. Diagnostics mới

Trong `outputs/reconstruction_debug/<object_id>/`, giữ các artifact hiện tại và bổ sung:

1. `target_bbox_mask.png`
   - Bounding box dùng để phân loại modal pixel.

2. `target_modal_protected.png`
   - Modal mask của target được giữ nguyên.

3. `foreign_modal_inside_bbox.png`
   - Modal pixel của object khác được đưa vào inpainting.

4. `foreign_modal_outside_bbox.png`
   - Modal pixel của object khác được bảo vệ.

5. `roi_real_pixels.png`
   - Phần ROI ánh xạ vào ảnh thật, không gồm padding.

6. `generation_mask.png`
   - Generation mask cũ, được giữ nguyên và thực sự gửi vào HD-Painter.

7. `accepted_model_rgb_mask.png`
   - Pixel model output được giữ sau validation.

8. `protected_source_pixels.png`
   - Pixel được phục hồi từ source.

9. `birefnet_candidate.png`
   - Candidate foreground trước connected-component filtering.

10. `final_reconstruction_support.png`
    - Support cuối cùng dùng cho grouping/matting.

Log cho từng target cần tóm tắt:

- ROI.
- Target bbox.
- Số target modal pixel được bảo vệ.
- Số foreign modal pixel bên trong bbox được inpaint.
- Số foreign modal pixel ngoài bbox được bảo vệ.
- Số generated pixel được chấp nhận.
- Số foreground support pixel được BiRefNet giữ.

## 7. Kế hoạch test theo TDD

### Task 1: Config và ROI

**Files:**

- Modify: `backend/config.yaml`
- Test: `tests/test_image_processor.py`

Các test:

1. `object_reconstruction.context_ratio == 0.025`.
2. ROI tạo từ amodal bbox có đúng kích thước 5% total expansion.
3. ROI sát biên ảnh vẫn giữ square geometry.
4. Padding không được đánh dấu là real ROI pixel.

### Task 2: Modal classification

**Files:**

- Modify: `backend/pipeline/types.py`
- Modify: `backend/pipeline/reconstruction/mask_preparation.py`
- Test: `tests/test_image_processor.py`

Các test:

1. Target modal luôn nằm trong protected mask.
2. Modal classification không làm thay đổi generation mask hiện tại.
3. Foreign modal trong target bbox nằm trong accepted model RGB mask.
4. Foreign modal ngoài target bbox không nằm trong accepted model RGB mask.
5. Một foreign object đi xuyên qua bbox bị chia theo pixel:
   - Phần trong bbox được inpaint.
   - Phần ngoài bbox được bảo vệ.
6. Non-modal pixel thật trong ROI nằm trong accepted model RGB mask.
7. Pixel ngoài ảnh không nằm trong accepted model RGB mask.
8. Accepted model RGB mask không bị intersect với amodal, completion hoặc generation mask.

### Task 3: Validation và full-output acceptance

**Files:**

- Modify: `backend/pipeline/reconstruction/reconstruction.py`
- Modify: `backend/pipeline/reconstruction/validate_reconstruction.py`
- Test: `tests/test_reconstruction_validation.py`

Các test:

1. Model output ngoài amodal mask và ngoài generation mask nhưng trong accepted model RGB mask được giữ.
2. Target modal RGB được phục hồi từ source.
3. Foreign modal ngoài target bbox được phục hồi từ source.
4. Foreign modal trong target bbox giữ RGB từ model output.
5. Padding không được write-back.
6. Directional hidden seed vẫn bắt blank extreme failure.
7. Model output sai size/mode vẫn bị reject.

### Task 4: Không thay đổi validated RGB

**Files:**

- Modify: `backend/pipeline/reconstruction/reconstruction.py`
- Modify: `backend/pipeline/reconstruction/validate_reconstruction.py`
- Test: `tests/test_reconstruction_validation.py`

Các test:

1. RGB trong accepted generation region giữ nguyên giá trị từ validated output.
2. Target modal không đổi.
3. Protected foreign modal không đổi.
4. Không tạo artifact `color_refined_output.png`.

### Task 5: BiRefNet support

**Files:**

- Modify: `backend/pipeline/matting.py`
- Test: `tests/test_reconstruction_support_refinement.py`

Các test:

1. High-confidence foreground ngoài amodal được chấp nhận.
2. Candidate ngoài accepted model RGB mask bị loại.
3. Foreign modal ngoài bbox bị loại.
4. Foreground component không kết nối target bị loại.
5. Foreground component kết nối target và có RGB change được giữ.
6. Final support không bị cap bởi amodal area.
7. Write mask vẫn là full accepted model RGB mask.

### Task 6: Group composition

**Files:**

- Modify: `backend/pipeline/grouping.py`
- Test: `tests/test_grouping.py`
- Test: `tests/test_group_matting.py`

Các test:

1. Group lấy RGB generated ngoài amodal shape.
2. Group không lấy protected foreign modal RGB.
3. Original modal của same-class member vẫn ưu tiên.
4. First reconstruction wins khi hai write mask overlap.
5. Group support dùng BiRefNet support thay vì toàn ROI.
6. Depth order không thay đổi.

### Task 7: Diagnostics và regression

**Files:**

- Modify: `backend/pipeline/reconstruction/reconstruction.py`
- Modify: `backend/pipeline/matting.py`
- Test: `tests/test_reconstruction_validation.py`
- Test: `tests/test_pipeline_diagnostics.py`

Các test:

1. Tất cả artifact mới được tạo khi bật `diagnostics_directory`.
2. Không tạo artifact khi diagnostics bị tắt.
3. Log summary chứa đủ pixel counts nhưng không dump full arrays.
4. Chạy toàn bộ test suite.

## 8. Tiêu chí nghiệm thu

Thay đổi được coi là hoàn thành khi:

1. Reconstruction ROI dùng `context_ratio=0.025`.
2. HD-Painter vẫn nhận generation mask được tạo bằng công thức cũ.
3. RGB nằm ngoài amodal mask nhưng trong allowed ROI được giữ.
4. Target modal không bị thay đổi.
5. Foreign modal bên trong amodal target bbox được inpaint.
6. Foreign modal bên ngoài target bbox được giữ nguyên.
7. Padding ngoài ảnh không lọt vào output.
8. BiRefNet có thể tìm target support ngoài amodal mask.
9. Final object alpha không mặc định phủ toàn bộ ROI.
10. Same-class grouping và depth order vẫn giữ hành vi hiện tại.
11. Reconstruction failure vẫn fallback an toàn về modal object.
12. Toàn bộ test suite pass.

## 9. Phạm vi không thay đổi

Kế hoạch này không thay đổi:

- SAM3 segmentation.
- Cross-class bounding-box overlap detection.
- Amodal model inference và validation.
- Completion-hole area calculation.
- Directional pair/depth-order logic.
- Same-class grouping rule dựa trên original modal bounding boxes.
- Reconstruction conflict policy.
- Background inpainting model.
- Layer extraction API.
- Model lifecycle và sequential CPU/GPU offload.

## 10. Rủi ro và biện pháp kiểm soát

### Rủi ro 1: Output ngoài generation mask không chắc là nội dung được inpaint trực tiếp

Biện pháp:

- Chấp nhận đây là chủ đích của refactor: giữ toàn bộ kết quả HD-Painter/SR trong ROI thay vì khôi phục source ngoài generation mask.
- Dùng BiRefNet để chỉ lấy foreground support có confidence, RGB change và kết nối với target.
- Lưu riêng `generation_mask.png` và `accepted_model_rgb_mask.png` để so sánh vùng model được yêu cầu sinh với vùng RGB cuối cùng được giữ.

### Rủi ro 2: HD-Painter sinh foreground không thuộc target

Biện pháp:

- Prompt vẫn nêu rõ target class.
- BiRefNet support yêu cầu confidence, RGB change và kết nối target.
- Foreign modal ngoài target bbox được bảo vệ.

### Rủi ro 3: Amodal bbox quá rộng

Biện pháp:

- ROI context chỉ còn `0.025`.
- Final foreground support do BiRefNet quyết định.
- Diagnostics hiển thị riêng target bbox và accepted support.

### Rủi ro 4: Amodal bbox quá nhỏ

Biện pháp:

- Full square ROI vẫn có 2.5% context mỗi phía.
- Output không bị cắt theo amodal shape bên trong ROI.
- Đây vẫn là giới hạn hình học còn lại; nếu target bị cắt tại ROI edge, cần đánh giá riêng chiến lược mở rộng bbox thay vì tăng lại context cho mọi object.

### Rủi ro 5: Modal object khác nằm trong target bbox nhưng không thực sự che target

Biện pháp:

- Đây là trade-off được chấp nhận theo quy tắc hình học mới.
- ROI nhỏ giảm số object không liên quan lọt vào target bbox.
- Directional completion seed vẫn phải tồn tại trước khi target được chạy reconstruction.
- Diagnostics cho phép nhận biết trường hợp classification sai.

## 11. Thứ tự triển khai sau khi được phê duyệt

1. Viết test failing cho config và modal classification.
2. Implement ROI/config và post-inference modal classification mà không thay đổi generation-mask construction.
3. Chạy targeted tests.
4. Viết test failing cho full-output validation.
5. Implement validation và accepted RGB mask.
6. Chạy targeted tests.
7. Viết test xác nhận validated RGB không bị chỉnh màu.
8. Viết test failing cho BiRefNet support ngoài amodal.
9. Refactor support refinement.
11. Viết và chạy grouping regression tests.
12. Bổ sung diagnostics.
13. Chạy toàn bộ test suite.
14. Chỉ sau khi tất cả test pass mới yêu cầu kiểm thử CUDA end-to-end bằng ảnh thật.
