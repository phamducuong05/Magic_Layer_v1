# Kế hoạch refactor BiRefNet chạy trực tiếp trên đầu ra HD-Painter

## 1. Mục tiêu

Refactor lần này thay đổi điểm chạy BiRefNet trong reconstruction pipeline:

```text
Luồng hiện tại

HD-Painter model_output
→ validate và khôi phục các pixel không được phép sửa
→ validated_output / reconstruction_canvas
→ BiRefNet
→ hard candidate mask
→ reconstruction support
```

thành:

```text
Luồng mới

HD-Painter model_output nguyên bản
→ kiểm tra cấu trúc output
→ BiRefNet trực tiếp
→ lọc foreground thuộc target
→ tạo soft reconstruction alpha
→ bảo vệ modal target, foreign outside và foreign inside không liên quan
→ alpha blend RGB từ HD-Painter với source
→ group composition
→ BiRefNet lần cuối cho toàn group
→ RGBA layer
```

Mục tiêu chính:

1. Không khôi phục pixel trước khi BiRefNet có cơ hội nhìn thấy toàn bộ foreground mà HD-Painter đã sinh.
2. Không dùng `accepted_rgb_mask` làm hard mask để cắt foreground BiRefNet.
3. Giữ biên foreground dưới dạng soft alpha để giảm đường viền cứng và các mảnh RGB nhỏ.
4. Chỉ cho phép phần foreign inside đã được xác nhận là occluder trong reconstruction domain bị thay thế.
5. Bảo vệ RGB gốc của target đang nhìn thấy, foreign outside và foreign inside không liên quan.
6. Vẫn loại foreground rời rạc hoặc không thuộc target vì BiRefNet là model class-agnostic.
7. Có fallback an toàn nếu HD-Painter output hoặc BiRefNet output không hợp lệ.

---

## 2. Quy tắc mask bắt buộc

### 2.1. `target.modal_mask`

Đây là phần target đang nhìn thấy trong ảnh nguồn.

Quy tắc:

- Dùng làm anchor để xác định component BiRefNet nào thuộc target.
- Luôn giữ RGB ảnh nguồn tại vùng này.
- Không ghi RGB HD-Painter đè lên vùng này.
- Vẫn có thể dùng alpha BiRefNet ở đây để kiểm tra component có nối với target hay không.

### 2.2. `foreign_modal_inside_bbox`

Đây là modal của mọi object khác nằm trong bbox amodal target. Quan hệ “nằm trong bbox” chưa đủ chứng minh toàn bộ vùng này đang che target, vì nó có thể chứa:

- occluder thật;
- phần khác của occluder nhưng không che target;
- object không liên quan tình cờ nằm trong bbox;
- lỗi hoặc phần dư segmentation.

Vì vậy không cho phép toàn bộ `foreign_modal_inside_bbox` bị thay thế. Mask này phải được chia thành hai phần.

#### `assigned_occluder`

Union modal mask của những object đã được depth ordering xác định đang che target:

```python
assigned_occluder = union(
    other.modal_mask
    for other in objects
    if other.object_id in target.occluder_ids
) & foreign_modal_inside_bbox
```

Mask này loại các foreign object không được xác định là occluder và chỉ giữ phần assigned occluder nằm trong bbox target. Vì vậy nó không thể làm thủng protection của `foreign_modal_outside_bbox`.

#### `replacement_domain`

Vùng hình học mà target có khả năng cần được khôi phục:

```python
replacement_domain = dilate(
    target.amodal_mask | composition_mask,
    replacement_domain_margin_pixels,
) & target_bbox_mask & roi_mask
```

Margin nhỏ cho phép xử lý biên, nhưng domain được clip lại theo bbox và ROI để tránh xóa toàn bộ occluder hoặc tràn sang foreign outside.

#### `replaceable_foreign_inside`

Phần foreign inside duy nhất được phép HD-Painter thay thế:

```python
replaceable_foreign_inside = (
    assigned_occluder
    & replacement_domain
    & roi_mask
    & ~target_modal
)
```

Một pixel chỉ được phép thay khi đồng thời:

- thuộc occluder đã được xác nhận;
- nằm trong vùng target có thể cần khôi phục;
- nằm trong ROI;
- không thuộc phần target đang nhìn thấy.

#### `protected_foreign_inside`

Phần foreign inside còn lại phải giữ RGB nguồn:

```python
protected_foreign_inside = (
    foreign_modal_inside_bbox
    & ~replaceable_foreign_inside
)
```

Tóm tắt:

```text
foreign_modal_inside_bbox
├── replaceable_foreign_inside
│   └── được phép reconstruction
└── protected_foreign_inside
    └── giữ nguyên RGB nguồn
```

### 2.3. `foreign_modal_outside_bbox`

Đây là foreign object nằm ngoài bbox amodal target.

Quy tắc:

- Là vùng bảo vệ cứng.
- Không cho BiRefNet extension đi vào.
- Không cho RGB HD-Painter ghi đè.
- Giữ RGB ảnh nguồn.

Foreign protection đúng sau refactor:

```python
foreign_protection = (
    foreign_modal_outside_bbox
    | protected_foreign_inside
)
```

Nếu cần dilation bảo vệ nhẹ, phải mở rộng protection rồi loại lại vùng được phép thay:

```python
foreign_protection = (
    dilate(
        foreign_modal_outside_bbox | protected_foreign_inside,
        foreign_protection_dilation_pixels,
    )
    & ~replaceable_foreign_inside
)
```

Phép `& ~replaceable_foreign_inside` ngăn protection dilation lấn ngược vào vùng occluder cần reconstruction.

### 2.4. `accepted_rgb_mask`

Trong code hiện tại:

```python
accepted_rgb_mask = roi_mask & ~protected_mask
```

với:

```python
protected_mask = target_modal | foreign_modal_outside_bbox
```

Đây là hành vi hiện tại và còn quá rộng quyền ghi trong bbox: mọi foreign inside đều nằm trong `accepted_rgb_mask`.

Sau refactor:

```python
protected_mask = (
    target_modal
    | foreign_modal_outside_bbox
    | protected_foreign_inside
)

accepted_rgb_mask = roi_mask & ~protected_mask
```

Sau refactor, mask này vẫn có ích cho:

- diagnostics;
- fallback validation;
- kiểm tra quyền ghi RGB cuối cùng;
- bảo đảm target modal, foreign outside và protected foreign inside không bị sửa.

Tuy nhiên, nó không còn được dùng để cắt trực tiếp BiRefNet candidate bằng:

```python
candidate &= accepted_rgb_mask
```

BiRefNet candidate sẽ được xác định từ alpha và liên kết component với target. Việc bảo vệ pixel được thực hiện rõ ràng ở bước tạo `write_alpha`.

### 2.5. `generation_mask`

`generation_mask` chỉ là đầu vào điều khiển HD-Painter:

```text
composition_mask | replaceable_foreign_inside
→ generation_seed
→ closing
→ dilation
→ giới hạn trong ROI
→ loại target modal
→ loại foreign protection
→ thêm lại composition_mask
→ generation_mask
```

Thay vì đưa toàn bộ modal occluder trong ROI vào seed:

```python
generation_seed = composition_mask | replaceable_foreign_inside
```

Sau morphology:

```python
generation_mask &= roi_mask
generation_mask &= ~target_modal
generation_mask &= ~foreign_protection
generation_mask |= composition_mask
```

Nếu `composition_mask` có giao với protection do dữ liệu không nhất quán, cần log conflict và chỉ thêm lại phần composition được occlusion logic xác nhận, không được âm thầm mở toàn bộ protection.

Sau khi HD-Painter chạy xong:

- không dùng `generation_mask` làm biên foreground cuối;
- có thể dùng nó làm bằng chứng rằng component chứa pixel do model được yêu cầu sinh;
- cho phép foreground BiRefNet mở rộng mềm ra ngoài biên generation mask nếu vẫn nối với target và không đi vào vùng bảo vệ.

### 2.6. Thay đổi trong mask preparation

File cần sửa:

```text
backend/pipeline/reconstruction/mask_preparation.py
```

Code hiện tại tạo `relevant_occluder` từ gần như toàn bộ modal của assigned occluder trong ROI:

```python
relevant_occluder = (
    occluder_union
    & roi_mask
    & ~target_modal
)
```

Sau refactor:

```python
assigned_occluder = (
    occluder_union
    & foreign_modal_inside_bbox
)

replacement_domain = (
    dilate(
        target.amodal_mask | composition_mask,
        replacement_domain_margin_pixels,
    )
    & target_bbox_mask
    & roi_mask
)

replaceable_foreign_inside = (
    assigned_occluder
    & replacement_domain
    & roi_mask
    & ~target_modal
)

protected_foreign_inside = (
    foreign_modal_inside_bbox
    & ~replaceable_foreign_inside
)

foreign_protection = (
    dilate(
        foreign_modal_outside_bbox | protected_foreign_inside,
        foreign_protection_dilation_pixels,
    )
    & ~replaceable_foreign_inside
    & roi_mask
)
```

Generation seed mới:

```python
generation_seed = (
    composition_mask
    | replaceable_foreign_inside
)
```

Các mask mới phải được lưu vào `DetectedObject` để mask preparation, reconstruction, BiRefNet, grouping và diagnostics dùng cùng một quyết định hình học; không tính lại độc lập ở từng stage.

---

## 3. Kiến trúc dữ liệu mới

File cần sửa:

```text
backend/pipeline/types.py
```

### 3.1. Giữ riêng raw canvas và canvas dùng cho composition

Bổ sung vào `DetectedObject`:

```python
raw_reconstruction_canvas: Optional[Image.Image] = None
reconstruction_canvas: Optional[Image.Image] = None
```

Ý nghĩa:

- `raw_reconstruction_canvas`: crop RGB nguyên bản do HD-Painter trả về.
- `reconstruction_canvas`: crop RGB đã được lọc và blend an toàn để dùng cho group composition.

Không dùng chung một field cho hai ý nghĩa này vì:

- BiRefNet cần raw output.
- Group composition cần output đã bảo vệ pixel.
- Diagnostics cần so sánh rõ raw và accepted result.
- Fallback không được vô tình làm mất raw evidence.

### 3.2. Bổ sung soft write alpha

Bổ sung:

```python
reconstruction_write_alpha: Optional[np.ndarray] = None
```

Đây là array full-image:

- dtype floating point;
- giá trị trong `[0.0, 1.0]`;
- `0.0`: không dùng RGB HD-Painter;
- `1.0`: dùng hoàn toàn RGB HD-Painter;
- giá trị giữa `0` và `1`: alpha blend ở biên.

Giữ `reconstruction_write_mask` để:

- tương thích với logic cũ;
- diagnostics;
- conflict detection;
- fallback.

Mask này được suy ra từ alpha:

```python
reconstruction_write_mask = reconstruction_write_alpha > 0
```

hoặc dùng một epsilon nhỏ:

```python
reconstruction_write_mask = reconstruction_write_alpha > 1e-6
```

### 3.3. Các field evidence vẫn giữ

Các field sau tiếp tục được dùng:

```python
reconstruction_evidence_alpha
reconstruction_extension_mask
reconstruction_support_mask
```

Ý nghĩa sau refactor:

- `reconstruction_evidence_alpha`: raw BiRefNet alpha được restore về full-image.
- `reconstruction_extension_mask`: hard support của phần target được sinh ngoài modal.
- `reconstruction_support_mask`: `target_modal | reconstruction_extension_mask`.

### 3.4. Lưu các mask phân vùng foreign

Bổ sung:

```python
reconstruction_replacement_domain_mask: Optional[np.ndarray] = None
reconstruction_replaceable_foreign_inside: Optional[np.ndarray] = None
reconstruction_protected_foreign_inside: Optional[np.ndarray] = None
reconstruction_foreign_protection_mask: Optional[np.ndarray] = None
```

Tiếp tục dùng `reconstruction_occluder_mask` cho assigned occluder đã giới hạn trong bbox/ROI, hoặc đổi tên field trong một migration riêng nếu cần tránh thay đổi quá rộng.

Các field đều dùng tọa độ full-image và được crop theo cùng `reconstruction_roi` khi đưa vào HD-Painter hoặc BiRefNet.

---

## 4. Refactor bước nhận kết quả HD-Painter

File cần sửa:

```text
backend/pipeline/reconstruction/reconstruction.py
```

### 4.1. Lưu raw output trước validation

Ngay sau khi HD-Painter trả về `reconstructed`:

```python
raw_output = reconstructed.convert("RGB")
detected.raw_reconstruction_canvas = raw_output
detected.reconstruction_roi = item.roi
```

Raw output phải được lưu trước mọi thao tác:

- restore source pixel;
- hard-mask clipping;
- blend;
- composition.

### 4.2. Chia validation thành hai loại

#### Validation cấu trúc

Chạy ngay sau HD-Painter:

- kiểm tra output là `PIL.Image`;
- convert được sang RGB;
- kích thước đúng bằng ROI;
- array có shape hợp lệ;
- không có lỗi dữ liệu;
- không phải output rỗng hoặc bất thường rõ ràng.

Validation này không sửa pixel.

#### Validation nội dung

Các color metrics vẫn được tính để log hoặc quyết định fallback, nhưng không khôi phục pixel trước BiRefNet.

Nếu cần so sánh thay đổi RGB:

```python
change_distance = mean(abs(raw_output - source_crop), axis=2)
```

Kết quả này được dùng ở bước lọc extension.

### 4.3. Chưa tạo final `reconstruction_canvas` tại đây

Trong luồng mới, `reconstruction_canvas` chỉ được tạo sau khi:

1. BiRefNet chạy trên raw output.
2. Target component được xác nhận.
3. `reconstruction_write_alpha` được tạo.
4. Raw RGB được blend với source RGB.

Nếu support refinement bị tắt, code có thể dùng fallback validation hiện tại để tạo `reconstruction_canvas`.

---

## 5. Refactor BiRefNet reconstruction-support pass

File cần sửa:

```text
backend/pipeline/matting.py
```

Hàm chính:

```python
refine_reconstruction_supports(...)
```

### 5.1. Đầu vào mới

Thay:

```python
canvas = target.reconstruction_canvas
alpha_crop = matte(canvas)
```

bằng:

```python
raw_canvas = target.raw_reconstruction_canvas
alpha_crop = matte(raw_canvas)
```

BiRefNet vẫn chạy trên crop ROI của target, không chạy trên toàn bộ ảnh gốc.

### 5.2. Chuẩn hóa alpha

Output BiRefNet:

```python
alpha_crop: np.ndarray[float]
shape = (roi.size, roi.size)
range = [0.0, 1.0]
```

Nếu model trả về kích thước khác:

- resize bilinear về ROI;
- clip về `[0.0, 1.0]`.

### 5.3. Tạo candidate thô

```python
candidate = alpha_crop >= alpha_low_threshold
```

Candidate được giới hạn:

```python
candidate &= roi_valid_crop
candidate &= ~foreign_protection_crop
```

Trong đó:

```python
foreign_protection = (
    foreign_modal_outside_bbox
    | protected_foreign_inside
)
```

`replaceable_foreign_inside` không nằm trong protection nên BiRefNet vẫn được phép nhận target mới tại đó. Phần foreign inside không liên quan bị loại.

Không dùng `accepted_rgb_crop` như một hard foreground mask:

```python
candidate &= accepted_rgb_crop
```

vì đây là phép cắt hard mask có thể làm mất biên nhỏ mà refactor lần này cần giữ.

### 5.4. Tạo target anchor

Target modal được dilation nhẹ:

```python
target_anchor = dilate(
    modal_crop,
    connection_margin_pixels,
)
```

Anchor dùng để trả lời:

> Component foreground này có nối với target đang nhìn thấy hay không?

Không dùng anchor làm write mask.

### 5.5. Tìm connected components

Chạy:

```python
num_labels, labels = cv2.connectedComponents(
    candidate.astype(np.uint8),
    connectivity=8,
)
```

Với từng component, tính:

- diện tích;
- overlap với target anchor;
- số pixel alpha mạnh;
- số pixel thay đổi bởi HD-Painter;
- overlap với generation/completion evidence;
- extension area ngoài modal;
- overlap với `foreign_protection`.

### 5.6. Điều kiện nhận component

Một component được nhận nếu:

1. Có overlap với `target_anchor`.
2. Có ít nhất một lượng pixel đạt `alpha_high_threshold`.
3. Có evidence thay đổi RGB ngoài `target.modal_mask`.
4. Có liên hệ với vùng reconstruction dự kiến.
5. Không phải component quá nhỏ.
6. Không vượt giới hạn extension bất thường.

Pseudocode:

```python
touches_target = np.any(component & target_anchor)

has_strong_alpha = np.any(
    component & (alpha_crop >= alpha_high_threshold)
)

extension_part = component & ~modal_crop

has_model_change = np.any(
    extension_part & changed_by_model
)

has_reconstruction_evidence = np.any(
    extension_part
    & dilated_generation_or_completion_evidence
)

accept = (
    touches_target
    and has_strong_alpha
    and has_model_change
    and has_reconstruction_evidence
    and component_area >= min_component_area_pixels
)
```

`has_reconstruction_evidence` chỉ cần một vùng giao nhỏ. Nó không cắt toàn bộ component theo generation mask. Nhờ vậy phần biên đúng do HD-Painter sinh vẫn có thể vượt nhẹ ra ngoài hard mask.

### 5.7. Vai trò của `support_change_threshold`

Tính:

```python
change_distance = np.mean(
    np.abs(
        raw_rgb.astype(np.int16)
        - source_rgb.astype(np.int16)
    ),
    axis=2,
)

changed_by_model = change_distance >= change_threshold
```

Chỉ kiểm tra thay đổi trên phần ngoài target modal:

```python
changed_extension = changed_by_model & ~modal_crop
```

Không yêu cầu target modal thay đổi vì vùng modal đúng ra phải được giữ nguyên.

Điều kiện change giúp tránh nhận nhầm:

- occluder cũ mà HD-Painter không sửa;
- foreign object còn nguyên trong output;
- foreground BiRefNet phát hiện nhưng không phải nội dung mới.

### 5.8. Áp dụng `support_max_extension_area_ratio`

Tham số này hiện mới được validate và truyền qua nhưng chưa thực sự áp dụng.

Ngân sách extension:

```python
max_extension_area = int(
    modal_area * support_max_extension_area_ratio
)
```

Nếu component có extension lớn hơn ngân sách:

- không crop thành hình chữ nhật;
- không co mask bằng bounding box;
- đánh dấu component là bất thường và loại;
- log đầy đủ diện tích để điều chỉnh config.

Lý do không cắt một phần component:

- dễ tạo cạnh phẳng;
- làm hỏng topology;
- có thể tạo các hình vuông tương tự lỗi mask trước đây.

Nếu về sau cần giữ một phần component lớn, sẽ bổ sung chiến lược geodesic selection riêng thay vì cắt theo bbox.

---

## 6. Tạo soft reconstruction alpha

Sau khi có `accepted_component`:

```python
extension_support = accepted_component & ~modal_crop
```

Giữ alpha gốc của BiRefNet:

```python
extension_alpha = alpha_crop * extension_support.astype(np.float32)
```

Áp dụng bảo vệ:

```python
extension_alpha[modal_crop] = 0.0
extension_alpha[foreign_protection_crop] = 0.0
```

Trong đó protection chứa:

```python
foreign_modal_outside_bbox | protected_foreign_inside
```

Không zero `replaceable_foreign_inside`, vì đây là phần occluder đã được xác nhận nằm trong reconstruction domain.

### 6.1. Feather tùy chọn

BiRefNet vốn đã trả soft alpha. Feather chỉ dùng rất nhẹ nếu diagnostics cho thấy biên bị gãy:

```python
extension_alpha = optional_small_gaussian_blur(extension_alpha)
```

Sau blur phải áp dụng lại protection:

```python
extension_alpha[modal_crop] = 0.0
extension_alpha[foreign_protection_crop] = 0.0
extension_alpha[~roi_valid_crop] = 0.0
```

Không dùng morphology closing/dilation trực tiếp trên alpha float. Nếu cần nối support, morphology chỉ chạy trên binary support trước khi nhân lại với alpha.

### 6.2. Restore về full-image

```python
target.reconstruction_evidence_alpha = restore_array(
    alpha_crop,
    roi,
)

target.reconstruction_write_alpha = restore_array(
    extension_alpha,
    roi,
)

target.reconstruction_extension_mask = (
    target.reconstruction_write_alpha > alpha_write_epsilon
)

target.reconstruction_write_mask = (
    target.reconstruction_write_alpha > alpha_write_epsilon
)

target.reconstruction_support_mask = (
    target_modal
    | target.reconstruction_extension_mask
)
```

---

## 7. Tạo reconstruction canvas bằng alpha blend

Sau khi có `extension_alpha`, tạo canvas an toàn:

```python
source_rgb = np.asarray(source_crop, dtype=np.float32)
raw_rgb = np.asarray(raw_canvas, dtype=np.float32)
alpha = extension_alpha[..., None]

composed_rgb = (
    raw_rgb * alpha
    + source_rgb * (1.0 - alpha)
)
```

Sau đó:

```python
target.reconstruction_canvas = Image.fromarray(
    np.clip(np.rint(composed_rgb), 0, 255).astype(np.uint8),
    mode="RGB",
)
```

Kết quả:

- target modal dùng RGB nguồn vì `write_alpha = 0`;
- foreign outside dùng RGB nguồn vì `write_alpha = 0`;
- `protected_foreign_inside` dùng RGB nguồn vì `write_alpha = 0`;
- chỉ `replaceable_foreign_inside` có thể được thay bằng target mới nếu BiRefNet/component filtering chấp nhận;
- vùng biên dùng soft alpha thay vì ghi đè boolean;
- pixel không thuộc target giữ nguyên source.

---

## 8. Refactor group composition

File cần sửa:

```text
backend/pipeline/grouping.py
```

### 8.1. Thay boolean write bằng alpha-aware write

Logic hiện tại:

```python
destination[writable] = candidate[writable]
```

Luồng mới ưu tiên:

```python
write_alpha = member.reconstruction_write_alpha
```

Trong vùng overlap:

```python
alpha_crop = write_alpha[overlap][..., None]

destination = (
    candidate * alpha_crop
    + destination * (1.0 - alpha_crop)
)
```

Nếu `reconstruction_canvas` đã được blend với source từ bước trước, group composition vẫn cần `write_alpha` để:

- không xem toàn bộ ROI là reconstruction;
- giải quyết conflict giữa nhiều member;
- không block member sau ở pixel alpha bằng zero;
- log đúng số pixel thực sự có reconstruction contribution.

### 8.2. Bảo vệ visible modal

Quy tắc hiện tại vẫn giữ:

```python
effective_alpha = 0
```

tại:

- `group.modal_mask`;
- visible modal của target;
- vùng reconstruction đã thuộc member ưu tiên trước đó nếu có conflict.

### 8.3. Conflict giữa nhiều reconstruction

Giữ quy tắc deterministic:

```text
member reconstruction trước thắng
```

Nhưng conflict được xác định theo:

```python
write_alpha > alpha_write_epsilon
```

thay vì toàn bộ `accepted_rgb_mask`.

Điều này tránh một member chiếm quyền ở những pixel mà alpha thực tế bằng zero.

---

## 9. BiRefNet lần hai cho group cuối

Hàm:

```python
refine_objects(...)
```

vẫn được giữ.

Hai pass có nhiệm vụ khác nhau:

### Pass 1: reconstruction support refinement

Đầu vào:

```text
raw HD-Painter ROI output
```

Mục đích:

- remove background ở output HD-Painter;
- tìm phần target được sinh;
- loại component không thuộc target;
- tạo `reconstruction_write_alpha`.

### Pass 2: final group matting

Đầu vào:

```text
composed RGB gồm visible modal gốc + reconstructed extension
```

Mục đích:

- tạo alpha cuối cho toàn group;
- làm sạch biên sau composition;
- tạo RGBA layer cuối cùng.

Pass 2 tiếp tục được giới hạn bởi:

```python
group.effective_support_mask
```

và:

```python
support_dilation_pixels
```

Do đó nó không được phép tự ý lấy toàn bộ foreground trong ROI.

---

## 10. Thay đổi orchestrator

File cần sửa:

```text
backend/pipeline/orchestrator.py
```

Thứ tự mới:

```text
1. prepare reconstruction masks
2. run HD-Painter
3. lưu raw_reconstruction_canvas
4. load BiRefNet
5. refine_reconstruction_supports trên raw canvas
6. tạo reconstruction_canvas đã alpha blend
7. group objects
8. compose reconstructed members
9. refine_objects bằng BiRefNet lần hai
10. extract RGBA layers
```

Điều kiện `has_accepted_reconstruction` cần dựa trên raw output hợp lệ:

```python
has_raw_reconstruction = any(
    obj.raw_reconstruction_canvas is not None
    for obj in objects
)
```

Sau support refinement, chỉ object có:

```python
reconstruction_canvas is not None
```

mới được coi là reconstruction đã được chấp nhận cho grouping.

---

## 11. Config đề xuất

File:

```text
backend/config.yaml
```

Đề xuất:

```yaml
pipeline:
  object_reconstruction:
    support_refinement_enabled: true
    support_input_source: raw_hd_painter

    support_alpha_low_threshold: 0.2
    support_alpha_high_threshold: 0.7
    support_change_threshold: 8.0
    support_connection_margin_pixels: 4
    support_max_extension_area_ratio: 2.0

    support_min_component_area_pixels: 8
    replacement_domain_margin_pixels: 2
    foreign_protection_dilation_pixels: 1
    support_alpha_write_epsilon: 0.01
    support_alpha_feather_pixels: 0

    support_require_generation_evidence: true
    support_generation_evidence_margin_pixels: 2
    support_fallback_to_validated_output: true
```

### Ý nghĩa tham số

#### `support_alpha_low_threshold`

- Ngưỡng tạo component support.
- Giảm xuống nếu mất các biên alpha mảnh.
- Tăng lên nếu BiRefNet lấy quá nhiều background.

#### `support_alpha_high_threshold`

- Một component phải chứa foreground đủ chắc chắn.
- Tăng lên để lọc component yếu.
- Không dùng ngưỡng này để cắt toàn bộ soft edge.

#### `support_change_threshold`

- Ngưỡng khác biệt RGB giữa raw HD-Painter và source.
- Tăng nếu các thay đổi màu rất nhỏ bị nhận nhầm.
- Giảm nếu HD-Painter chỉ thay đổi nhẹ nhưng đúng.

#### `support_connection_margin_pixels`

- Dilation target modal để tạo anchor.
- Tăng nếu target mới bị tách khỏi modal bởi một khe nhỏ.
- Quá lớn có thể nhận nhầm foreign component gần target.

#### `support_max_extension_area_ratio`

- Giới hạn diện tích extension so với modal target.
- Ngăn BiRefNet lấy một foreground quá lớn trong ROI.
- Lần refactor này sẽ thực sự áp dụng tham số.

#### `support_min_component_area_pixels`

- Loại các component nhỏ, rời rạc.
- Không nên đặt quá lớn vì có thể xóa chi tiết nhỏ đúng.

#### `replacement_domain_margin_pixels`

- Mở rộng nhẹ vùng `amodal_mask | composition_mask`.
- Quyết định phần nào của assigned occluder có thể trở thành `replaceable_foreign_inside`.
- Quá lớn sẽ cho phép sửa quá nhiều occluder; quá nhỏ có thể không cover đủ biên phần bị che.

#### `foreign_protection_dilation_pixels`

- Mở rộng vùng bảo vệ quanh `foreign_modal_outside_bbox | protected_foreign_inside`.
- Sau dilation phải loại lại `replaceable_foreign_inside`.
- Không được biến toàn bộ `foreign_modal_inside_bbox` thành protection.

#### `support_alpha_write_epsilon`

- Ngưỡng rất nhỏ để chuyển soft alpha thành write/conflict mask.
- Không dùng để làm hard alpha cho ảnh cuối.

#### `support_alpha_feather_pixels`

- Blur alpha tùy chọn.
- Mặc định `0` vì BiRefNet đã tạo soft alpha.
- Chỉ tăng nếu diagnostics cho thấy biên còn gãy.

#### `support_generation_evidence_margin_pixels`

- Cho phép component chứng minh liên hệ với generation mask trong một margin nhỏ.
- Không cắt component theo generation mask.

---

## 12. Diagnostics mới

Files cần sửa:

```text
backend/pipeline/reconstruction/artifacts.py
backend/pipeline/reconstruction/reconstruction.py
backend/pipeline/matting.py
```

Thứ tự artifact đề xuất từ đoạn HD-Painter output:

```text
23_model_output.png
24_raw_birefnet_alpha.png
25_assigned_occluder.png
26_replacement_domain.png
27_replaceable_foreign_inside.png
28_protected_foreign_inside.png
29_foreign_protection.png
30_target_connection_anchor.png
31_birefnet_candidate.png
32_changed_by_model.png
33_generation_evidence.png
34_accepted_target_component.png
35_reconstruction_extension_alpha.png
36_reconstruction_extension_mask.png
37_reconstruction_write_alpha.png
38_composed_reconstruction.png
39_final_reconstruction_support.png
```

Ý nghĩa:

### `23_model_output.png`

Raw RGB do HD-Painter trả về, trước mọi restore/blend.

### `24_raw_birefnet_alpha.png`

Alpha remove-background nguyên bản của BiRefNet.

### `25_assigned_occluder.png`

Union modal của các object đã được depth ordering xác định đang che target.

### `26_replacement_domain.png`

Vùng amodal/composition được mở rộng nhẹ, giới hạn nơi occluder có thể bị thay.

### `27_replaceable_foreign_inside.png`

Phần assigned occluder nằm trong replacement domain và được phép reconstruction.

### `28_protected_foreign_inside.png`

Phần foreign inside không được chứng minh là occluder cần thay và phải giữ RGB nguồn.

### `29_foreign_protection.png`

Union foreign outside và protected foreign inside sau protection dilation, nhưng đã loại lại replaceable region.

### `30_target_connection_anchor.png`

Target modal sau dilation dùng để xác định component nào nối với target.

### `31_birefnet_candidate.png`

Binary candidate từ `raw_alpha >= alpha_low_threshold`, đã loại toàn bộ foreign protection.

### `32_changed_by_model.png`

Pixel RGB raw output thay đổi đủ nhiều so với source.

### `33_generation_evidence.png`

Vùng generation/completion được dilation nhẹ chỉ để xác nhận component có liên quan reconstruction.

### `34_accepted_target_component.png`

Connected component được xác nhận thuộc target.

### `35_reconstruction_extension_alpha.png`

Soft alpha phần target mới ngoài modal.

### `36_reconstruction_extension_mask.png`

Binary support suy ra từ extension alpha.

### `37_reconstruction_write_alpha.png`

Alpha thực tế được phép ghi RGB sau khi áp dụng mọi protection.

### `38_composed_reconstruction.png`

Raw HD-Painter RGB đã alpha blend với source.

### `39_final_reconstruction_support.png`

```python
target_modal | reconstruction_extension_mask
```

Artifact numbering phải được cập nhật tập trung trong `artifacts.py` để thứ tự tên file luôn phản ánh đúng thứ tự pipeline.

---

## 13. Fallback và xử lý lỗi

### 13.1. HD-Painter lỗi

Nếu HD-Painter không trả output hợp lệ:

- không tạo raw canvas;
- ghi `reconstruction_failure_stage`;
- object dùng modal gốc;
- pipeline tiếp tục với object khác.

### 13.2. BiRefNet lỗi ở pass reconstruction

Nếu `support_fallback_to_validated_output: true`:

- chạy validation/restore theo luồng cũ;
- chỉ ghi RGB trong vùng được phép;
- support fallback về modal hoặc reconstruction mask hiện tại;
- log rõ `fallback_to_validated_output`.

Nếu fallback bị tắt:

- bỏ reconstruction của object;
- giữ target modal gốc.

### 13.3. Không có component hợp lệ

Nếu raw alpha không tạo component thỏa điều kiện:

- không dùng toàn bộ raw HD-Painter output;
- `reconstruction_support_mask = target_modal`;
- `reconstruction_write_alpha = 0`;
- log `keep_modal_prior`.

### 13.4. Extension vượt area ratio

- loại component bất thường;
- không cắt component theo bbox;
- diagnostics lưu component trước khi loại;
- log area và configured limit.

---

## 14. Kế hoạch test

### 14.1. Test data contract

File:

```text
tests/test_image_processor.py
```

Kiểm tra:

- raw canvas được lưu riêng;
- reconstruction canvas chưa được coi là accepted trước BiRefNet;
- alpha full-image đúng shape;
- reset state xóa đầy đủ field mới;
- assigned occluder được giới hạn trong foreign inside;
- assigned occluder trong replacement domain tạo đúng `replaceable_foreign_inside`;
- foreign object không thuộc `occluder_ids` tạo thành `protected_foreign_inside`;
- phần assigned occluder ngoài replacement domain vẫn được bảo vệ;
- generation seed không còn chứa toàn bộ occluder trong ROI;
- protection dilation không lấn ngược vào replaceable region.

### 14.2. Test reconstruction output

File:

```text
tests/test_reconstruction_validation.py
```

Kiểm tra:

1. HD-Painter output được lưu vào `raw_reconstruction_canvas`.
2. Raw canvas không bị restore pixel trước BiRefNet.
3. Output sai kích thước bị từ chối.
4. Object lỗi không ảnh hưởng object khác.
5. Fallback validation vẫn hoạt động.

### 14.3. Test BiRefNet support refinement

File:

```text
tests/test_reconstruction_support_refinement.py
```

Các case bắt buộc:

1. `matte()` nhận đúng raw HD-Painter canvas.
2. Component nối với target modal được nhận.
3. Component rời target bị loại.
4. Component không có strong alpha bị loại.
5. Component không có RGB change bị loại.
6. Component có evidence trong generation region được phép mở rộng ra ngoài hard generation mask.
7. `replaceable_foreign_inside` không bị loại khỏi candidate.
8. `protected_foreign_inside` bị loại khỏi candidate.
9. `foreign_modal_outside_bbox` bị loại.
10. Target modal có write alpha bằng zero.
11. Soft alpha tại biên được giữ, không bị ép thành `0/1`.
12. Component nhỏ bị loại.
13. Extension vượt `support_max_extension_area_ratio` bị loại.
14. BiRefNet failure kích hoạt fallback.

### 14.4. Test group composition

File:

```text
tests/test_grouping.py
```

Kiểm tra:

1. RGB được alpha blend đúng công thức.
2. Alpha bằng `0` giữ destination/source.
3. Alpha bằng `1` dùng HD-Painter RGB.
4. Alpha giữa `0` và `1` tạo màu blend đúng.
5. Visible modal không bị ghi đè.
6. Foreign outside không bị ghi đè.
7. Protected foreign inside không bị ghi đè.
8. Replaceable foreign inside có thể được thay bởi reconstruction hợp lệ.
9. Conflict giữa nhiều member vẫn deterministic.
10. Pixel alpha zero không bị tính là conflict.

### 14.5. Test diagnostics

Kiểm tra:

- đủ artifact mới;
- tên có số đúng thứ tự;
- tất cả crop artifact có size bằng ROI;
- full-image mask có shape đúng khi restore;
- không còn artifact tên cũ gây hiểu sai semantics.

### 14.6. Regression tests

Kiểm tra:

- object không cần reconstruction vẫn dùng workflow cũ;
- completion và depth ordering không đổi;
- generation mask chỉ thay ở cách giới hạn occluder bằng `replaceable_foreign_inside`;
- background LaMa không bị ảnh hưởng;
- final layer extraction vẫn nhận `GroupedObject.soft_alpha`;
- kết quả không phụ thuộc thứ tự filesystem hoặc diagnostics.

---

## 15. Trình tự triển khai

### Bước 1

Thêm field mới vào `DetectedObject` và cập nhật reset/cleanup logic.

### Bước 2

Refactor `mask_preparation.py` để tạo:

- assigned occluder trong bbox target;
- replacement domain;
- replaceable foreign inside;
- protected foreign inside;
- foreign protection;
- generation seed đã thu hẹp.

### Bước 3

Viết unit test cho việc phân vùng foreign và generation mask mới.

### Bước 4

Sửa reconstruction stage để lưu raw HD-Painter output trước validation.

### Bước 5

Tách structural validation khỏi pixel restoration.

### Bước 6

Viết unit test chứng minh BiRefNet nhận raw output.

### Bước 7

Refactor `refine_reconstruction_supports()`:

- raw alpha;
- target anchor;
- connected components;
- model-change evidence;
- generation evidence;
- dùng foreign protection đã được mask preparation tính sẵn;
- area cap.

### Bước 8

Tạo `reconstruction_write_alpha` và support masks.

### Bước 9

Tạo accepted reconstruction canvas bằng alpha blend.

### Bước 10

Refactor group composition để hỗ trợ soft alpha và conflict theo alpha.

### Bước 11

Cập nhật orchestrator theo trạng thái raw/accepted reconstruction mới.

### Bước 12

Bổ sung config và validation cho các tham số mới.

### Bước 13

Cập nhật artifact order và diagnostics.

### Bước 14

Chạy targeted tests, sau đó chạy toàn bộ test suite.

### Bước 15

Chạy một ảnh thực tế và so sánh:

- raw HD-Painter;
- raw BiRefNet alpha;
- replaceable/protected foreign inside;
- accepted component;
- extension alpha;
- composed reconstruction;
- final RGBA.

---

## 16. Tiêu chí hoàn thành

Refactor được coi là hoàn thành khi:

1. BiRefNet pass đầu nhận trực tiếp raw output của HD-Painter.
2. Raw output không bị hard restore trước BiRefNet.
3. `foreign_modal_inside_bbox` được chia thành replaceable và protected.
4. Chỉ assigned occluder trong replacement domain được phép reconstruction.
5. Foreign outside và protected foreign inside đều là hard protection.
6. Target modal giữ RGB nguồn.
7. Foreground rời target bị loại bằng component filtering.
8. RGB reconstruction được đưa vào composition bằng soft alpha.
9. `support_max_extension_area_ratio` thực sự được áp dụng.
10. BiRefNet pass cuối vẫn tạo alpha cho composed group.
11. Diagnostics thể hiện đúng thứ tự và quyết định của pipeline.
12. Có fallback nếu raw output hoặc BiRefNet thất bại.
13. Toàn bộ targeted tests và regression tests liên quan đều pass.

---

## 17. Luồng cuối cùng sau refactor

```text
modal/amodal/completion masks
→ assigned_occluder
→ replacement_domain
→ replaceable_foreign_inside
→ protected_foreign_inside
→ foreign_protection
→ generation mask
→ HD-Painter
→ raw_reconstruction_canvas
→ structural validation
→ BiRefNet raw alpha
→ candidate components
→ target modal anchor filtering
→ strong-alpha filtering
→ model-change filtering
→ generation-evidence filtering
→ loại foreign_modal_outside_bbox và protected_foreign_inside
→ cho phép replaceable_foreign_inside
→ extension area validation
→ soft reconstruction_write_alpha
→ giữ nguyên target modal RGB
→ alpha blend raw HD-Painter RGB với source
→ reconstruction_canvas
→ group composition
→ final group BiRefNet
→ layer alpha/color refinement
→ RGBA layer cuối cùng
```
