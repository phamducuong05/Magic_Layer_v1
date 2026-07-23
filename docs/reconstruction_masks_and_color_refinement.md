# Thay đổi gần đây: Reconstruction masks và Color refinement

## 1. Mục tiêu của hai task

Hai task gần đây giải quyết hai vấn đề khác nhau nhưng liên quan trực tiếp với nhau:

1. **Tách mask dùng để sinh ảnh khỏi mask dùng để ghi kết quả**
   - HD-Painter cần một vùng đủ rộng để loại bỏ ảnh hưởng của vật thể che.
   - Pipeline chỉ nên lấy RGB mới tại phần thật sự bị thiếu của vật thể đích.
   - Vì vậy `reconstruction_mask` và `reconstruction_generation_mask` không còn có cùng vai trò.

2. **Refine màu của RGB do HD-Painter sinh ra**
   - Ảnh HD-Painter có thể bị nhạt, mờ hoặc lệch màu so với phần modal đang nhìn thấy của vật thể.
   - Pipeline lấy màu phẳng đáng tin cậy từ **modal mask gốc của target** để điều chỉnh nhẹ các pixel được reconstruction.
   - Logic này được dùng chung với nền tảng refine màu đã có trong `layerd_refine.py`.

Luồng tổng quát sau hai task:

```text
completion_hole
    ↓
exact directional seed
    ↓
lọc connected components và tạo reconstruction_mask
    ↓
khóa ROI theo target
    ↓
thêm toàn bộ occluder nằm trong ROI
    ↓
closing + dilation
    ↓
reconstruction_generation_mask
    ↓
HD-Painter
    ↓
validation và khôi phục pixel ngoài vùng cho phép
    ↓
đánh giá màu
    ↓
refine màu trong reconstruction_mask
    ↓
reconstruction_canvas
```

---

## 2. Các loại mask và dữ liệu mới

### File được sửa

`backend/pipeline/types.py`

### Các trường được bổ sung vào `DetectedObject`

| Trường | Ý nghĩa |
|---|---|
| `reconstruction_seed_mask` | Phần giao chính xác giữa `completion_hole` của target và modal mask của occluder. Đây là bằng chứng trực tiếp cho một hướng che khuất. |
| `reconstruction_mask` | Vùng RGB mới được phép sử dụng cho target. Trong code reconstruction, mask này còn được gọi là `composition_mask`. |
| `reconstruction_generation_seed_mask` | Mask trước morphology, bằng `reconstruction_mask` cộng với toàn bộ phần occluder liên quan nằm trong ROI. |
| `reconstruction_generation_mask` | Mask cuối cùng truyền cho HD-Painter sau closing và dilation. |
| `reconstruction_occluder_mask` | Toàn bộ modal mask của các occluder được gán, giới hạn trong ROI của target và loại bỏ modal mask của target. |
| `reconstruction_input_roi` | ROI vuông được khóa theo target trước khi thêm toàn bộ occluder vào generation mask. |

Các trường đã có và vẫn được giữ nguyên:

| Trường | Ý nghĩa |
|---|---|
| `modal_mask` | Phần vật thể thực sự đang nhìn thấy trong ảnh gốc. |
| `amodal_mask` | Hình dạng hoàn chỉnh do completion model dự đoán. |
| `completion_hole_mask` | Phần bị thiếu: `amodal_mask AND NOT modal_mask`. |
| `reconstruction_canvas` | Crop RGB cuối cùng đã qua HD-Painter, validation và color refinement. |
| `reconstruction_roi` | Tọa độ dùng để đặt `reconstruction_canvas` trở lại ảnh toàn cục. |

Việc lưu riêng từng mask giúp diagnostics có thể cho biết lỗi xuất hiện ở bước nào, thay vì chỉ nhìn một `generation_mask` cuối cùng và không biết nó được tạo như thế nào.

---

## 3. Task 1 — Luồng tạo mask mới

## 3.1. Tạo directional seed

### File được sửa

`backend/pipeline/reconstruction/mask_preparation.py`

### Hàm được thêm

`_directional_reconstruction_mask(target, occluder, composition_margin_pixels)`

Với từng cặp có hướng:

```text
target = vật thể cần reconstruction
occluder = vật thể đang che target
```

Code tạo:

```text
completion_hole = target.completion_hole_mask
occluder_modal = occluder.modal_mask

exact_seed = completion_hole AND occluder_modal
```

`exact_seed` trả lời câu hỏi:

> Phần nào trong vùng target bị thiếu thực sự trùng với pixel nhìn thấy của occluder?

Nếu `exact_seed` rỗng, hướng che khuất này không có bằng chứng pixel trực tiếp và không tạo reconstruction mask.

### Tác dụng

- Không dùng toàn bộ `completion_hole` một cách mù quáng.
- Loại bỏ các vùng amodal completion mọc sai nhưng không liên quan tới occluder.
- Giữ được thông tin theo hướng: A bị B che khác với B bị A che.
- Cho phép hai chiều cùng tồn tại nếu cả hai hướng đều có vùng reconstruction hợp lệ.

---

## 3.2. Tạo candidate có margin nhỏ

Modal mask của occluder được mở rộng bằng:

```text
composition_margin_pixels
```

Sau đó:

```text
candidate =
    completion_hole
    AND expanded_occluder_modal
```

Margin này chỉ dùng để lấy thêm một vùng nhỏ sát biên occluder. Nó không phải dilation lớn dùng cho HD-Painter.

### Tác dụng

- Bao phủ sai số nhỏ giữa biên `amodal_mask` và biên `occluder.modal_mask`.
- Tránh `reconstruction_mask` chỉ là một đường giao quá mỏng.
- Vẫn giới hạn vùng RGB được sử dụng gần bằng chứng che khuất thực tế.

Cấu hình hiện tại:

```yaml
composition_margin_pixels: 4
```

---

## 3.3. Lọc connected components trước khi dilation

Code chạy `cv2.connectedComponents()` trên `candidate`.

Chỉ những component có chứa pixel của `exact_seed` mới được giữ:

```text
filtered_component phải giao exact_seed
```

Mask sau lọc tiếp tục bị giới hạn:

```text
filtered &= target.amodal_mask
filtered &= NOT target.modal_mask
```

Kết quả chính là directional `reconstruction_mask`.

Nếu target có nhiều occluder, các directional mask được OR lại:

```text
reconstruction_mask =
    directional_mask_1
    OR directional_mask_2
    OR ...
```

### Tác dụng

- Dilation không còn làm lớn các đường viền hoặc component nhiễu trước đó.
- Chỉ vùng nằm trong amodal target và ngoài phần modal đang nhìn thấy mới được phép nhận RGB reconstruction.
- Hạn chế hiện tượng generation mask vô tình tạo lại hình dạng cuốn sách, điện thoại hoặc một vật thể không liên quan do viền amodal bị nhiễu.

### Lưu ý về phạm vi hiện tại

Code hiện tại lấy **phần của completion hole nằm trong occluder support đã mở rộng**, rồi lọc component dựa trên `exact_seed`.

Nó không lấy toàn bộ một component dài của `completion_hole` chỉ vì một đầu của component chạm occluder. Đây là lựa chọn an toàn nhằm tránh dùng quá nhiều vùng amodal completion không chắc chắn.

---

## 3.4. Dùng directional area để quyết định hướng reconstruction

### File được sửa

`backend/pipeline/reconstruction/mask_preparation.py`

### Hàm liên quan

`prepare_raw_reconstruction_masks(...)`

Trước đây, tổng diện tích `completion_hole` có thể làm sai quyết định depth/reconstruction nếu hole chứa nhiều phần không liên quan.

Code mới tính diện tích sau khi đã áp dụng:

- hướng target–occluder;
- giao với occluder;
- margin nhỏ;
- connected-component filtering;
- ngưỡng noise.

Diện tích này được truyền vào:

```python
assign_directional_pair_roles(...)
```

### Tác dụng

- Một phần completion hole ở xa occluder không còn quyết định sai hướng.
- Nếu hai hướng đều có directional area đủ lớn, pipeline có thể reconstruction cả hai vật thể.
- `occluder_ids` vẫn được lưu riêng cho mỗi target để tạo prompt đúng.
- Quyết định depth vẫn được giữ cho thứ tự hiển thị layer.

---

## 3.5. Khóa ROI theo target

### File được sửa

`backend/pipeline/reconstruction/mask_preparation.py`

Trong `build_reconstruction_masks(...)`, ROI được tạo trước khi thêm toàn bộ occluder:

```text
ROI support =
    target.amodal_mask
    OR reconstruction_mask
```

ROI được lưu vào:

```python
detected.reconstruction_input_roi
```

### Tại sao phải khóa ROI trước?

Nếu đưa toàn bộ occluder vào trước khi tính ROI:

- ROI có thể mở rộng theo toàn bộ kích thước của occluder;
- target bị nhỏ lại tương đối trong crop;
- crop chứa quá nhiều visual context của vật che;
- HD-Painter dễ tập trung vào occluder và tái tạo nó thay vì tiếp tục target;
- VRAM và inference time tăng do ROI lớn hơn.

ROI mới luôn lấy target làm trung tâm, còn occluder chỉ được lấy phần nằm bên trong ROI đó.

---

## 3.6. Lấy toàn bộ occluder trong ROI

Sau khi ROI đã được khóa:

```text
relevant_occluder =
    union(assigned_occluder.modal_mask)
    AND ROI
    AND NOT target.modal_mask
```

Kết quả được lưu vào:

```python
detected.reconstruction_occluder_mask
```

### Tác dụng

Đây là thay đổi giúp giảm visual contamination:

- HD-Painter không chỉ được yêu cầu thay đổi đúng phần giao nhỏ giữa hole và occluder.
- Toàn bộ hình ảnh nhìn thấy của vật che bên trong crop được đưa vào vùng generation.
- Model có cơ hội xóa ngữ cảnh mạnh của cuốn sách, camera, điện thoại hoặc bàn tay khỏi input generation.
- Modal mask của target luôn bị loại ra để bảo vệ phần target đang nhìn thấy.

---

## 3.7. Tách generation seed và generation mask

Generation seed được tạo bằng:

```text
generation_seed =
    reconstruction_mask
    OR relevant_occluder
```

Nó được lưu vào:

```python
detected.reconstruction_generation_seed_mask
```

Sau đó mới chạy morphology:

1. **Closing** để nối các khe nhỏ trong mask.
2. **Dilation** để mở rộng vùng model được phép sinh.
3. Clip lại theo ROI.
4. Loại modal mask của target.
5. OR lại `reconstruction_mask` để bảo đảm vùng cần ghi kết quả không bị morphology làm mất.

Kết quả:

```python
detected.reconstruction_generation_mask
```

Cấu hình hiện tại:

```yaml
generation_mask_closing_pixels: 7
generation_mask_dilation_pixels: 12
```

### Tác dụng

- HD-Painter nhận mask rộng hơn để xóa hoàn toàn occluder và xử lý biên.
- Noise được lọc **trước** dilation nên noise nhỏ không bị phóng đại.
- Target modal được bảo vệ, không bị model sửa lại.
- Mask dùng để sinh ảnh không còn quyết định trực tiếp toàn bộ vùng RGB sẽ được ghép vào target.

---

## 3.8. HD-Painter dùng generation mask, pipeline dùng reconstruction mask

### File được sửa

`backend/pipeline/reconstruction/reconstruction.py`

### Hàm liên quan

`reconstruct_objects(...)`

Code lấy hai mask riêng:

```text
generation_mask =
    reconstruction_generation_mask
    hoặc reconstruction_mask nếu generation mask chưa tồn tại

composition_mask =
    reconstruction_mask
```

Trong ROI:

| Dữ liệu | Được dùng để làm gì? |
|---|---|
| `mask_crop` từ `generation_mask` | Chuyển thành ảnh mask và truyền vào HD-Painter. |
| `composition_crop` từ `reconstruction_mask` | Validation vùng cần có RGB, color metrics, refine và vùng RGB có ý nghĩa cho target. |

### Ý nghĩa quan trọng

HD-Painter có thể sinh trên một vùng rộng chứa toàn bộ occluder, nhưng pipeline không coi toàn bộ vùng đó là phần reconstructed của target.

Ví dụ:

```text
generation mask:
    toàn bộ cuốn sách trong ROI + phần người bị sách che + dilation

reconstruction mask:
    chỉ phần người được xác định bị cuốn sách che
```

Nhờ vậy, hình cuốn sách được loại khỏi visual context trong lúc sinh, nhưng RGB của target chỉ được lấy tại phần người bị thiếu.

---

## 3.9. Validation sau HD-Painter

### File được sửa

`backend/pipeline/reconstruction/validate_reconstruction.py`

### Hàm liên quan

`validate_reconstruction_result(...)`

Validation nhận:

```text
hard_mask = generation_mask
completion_hole = reconstruction_mask
```

`hard_mask` được mở rộng nhẹ bằng `blend_allowance_ratio` để tạo permitted region. Sau đó:

```text
mọi pixel ngoài permitted region
    được khôi phục từ source_crop
```

### Tác dụng

- Không reject toàn bộ kết quả chỉ vì HD-Painter hoặc Poisson blending làm thay đổi nhẹ pixel ngoài mask.
- Bảo vệ source pixel ngoài vùng generation.
- Chỉ `reconstruction_mask` được dùng để xác minh vùng target thật sự cần RGB mới không bị rỗng hoặc trắng/đen bất thường.

Cấu hình hiện tại:

```yaml
blend_allowance_ratio: 0.012
```

---

## 4. Diagnostics được bổ sung

### Các file được sửa

- `backend/pipeline/reconstruction/reconstruction.py`
- `backend/models/object_reconstruction/adapter.py`
- `backend/pipeline/orchestrator.py`

Khi:

```yaml
diagnostics_directory: "outputs/reconstruction_debug"
```

mỗi object có một thư mục riêng.

### Các ảnh diagnostics và ý nghĩa

| Ảnh | Ý nghĩa |
|---|---|
| `source.png` | Crop ảnh gốc được truyền vào reconstruction pipeline. |
| `completion_hole.png` | Toàn bộ `amodal - modal` trong ROI; có thể chứa noise từ completion model. |
| `directional_seed.png` | Giao chính xác giữa completion hole và modal mask của occluder. |
| `filtered_reconstruction_mask.png` | `reconstruction_mask` sau directional filtering và connected-component filtering. |
| `composition_mask.png` | Mask thực tế dùng để validate/refine phần RGB của target; hiện tại giống `filtered_reconstruction_mask.png`. |
| `occluder_mask.png` | Toàn bộ occluder liên quan nằm trong ROI, sau khi loại modal target. |
| `full_occluder_in_roi.png` | Cùng dữ liệu với `occluder_mask.png` ở implementation hiện tại; tên này nhấn mạnh mục tiêu kiểm tra toàn bộ occluder trong ROI. |
| `generation_before_dilation.png` | `reconstruction_mask OR relevant_occluder`, trước closing/dilation. |
| `generation_after_dilation.png` | Generation mask sau morphology và clipping; chính là mask truyền cho HD-Painter. |
| `generation_mask.png` | Mask crop cuối cùng truyền cho HD-Painter; hiện tại giống `generation_after_dilation.png`. |
| `base_output_512.png` | Kết quả phase HD-Painter 512×512 trước Super Resolution. |
| `sr_output.png` | Kết quả trực tiếp của phase Super Resolution, nếu object đủ điều kiện chạy SR. |
| `model_output.png` | Kết quả cuối adapter trả về sau khi resize về kích thước ROI. |
| `validated_output.png` | Kết quả sau validation và sau khi source pixel ngoài permitted region được khôi phục. |
| `color_refined_output.png` | Kết quả cuối sau target-aware color refinement. Đây là ảnh được lưu vào `reconstruction_canvas`. |

### HD-Painter debug artifacts

Trong `backend/models/object_reconstruction/adapter.py`:

- phase 512 lưu `base_output_512`;
- phase SR lưu `sr_output`;
- `consume_debug_artifacts()` trả các ảnh này cho pipeline và xóa danh sách tạm.

Trong `backend/pipeline/orchestrator.py`:

```python
debug_artifacts_provider =
    reconstruction_model.consume_debug_artifacts
```

được truyền vào `reconstruct_objects(...)`.

### Cách đọc diagnostics để tìm lỗi

1. Nếu `completion_hole.png` sai: lỗi bắt đầu từ amodal completion.
2. Nếu completion hole đúng nhưng `directional_seed.png` sai/rỗng: cặp target–occluder hoặc modal mask occluder sai.
3. Nếu directional seed đúng nhưng `filtered_reconstruction_mask.png` mọc vùng lạ: kiểm tra margin và connected-component filtering.
4. Nếu reconstruction mask đúng nhưng `generation_before_dilation.png` sai: kiểm tra `occluder_ids`, occluder mask và ROI.
5. Nếu lỗi chỉ xuất hiện ở `generation_after_dilation.png`: closing/dilation đang quá mạnh.
6. Nếu generation mask đúng nhưng `base_output_512.png` sai: lỗi thuộc prompt, model 512 hoặc visual context.
7. Nếu base output tốt nhưng `sr_output.png` xấu/mờ: lỗi thuộc phase SR.
8. Nếu model output tốt nhưng validated output khác: kiểm tra permitted region và blend allowance.
9. Nếu validated output đúng nhưng refined output sai màu: giảm strength hoặc kiểm tra palette modal target.

---

## 5. Task 2 — Refine màu dựa trên modal mask của target

## 5.1. Tạo hàm refine dùng chung

### File được sửa

`backend/core/layerd_refine.py`

### Hàm được thêm

```python
refine_with_reference_mask(...)
```

Các đầu vào chính:

| Đầu vào | Vai trò |
|---|---|
| `image` | Ảnh cần refine, đối với reconstruction là output HD-Painter đã validate. |
| `edit_mask` | Chỉ pixel trong vùng này được phép đổi. |
| `reference_image` | Ảnh gốc dùng để lấy màu đáng tin cậy. |
| `reference_mask` | Vùng cụ thể trên ảnh gốc được phép dùng để học palette. |
| `max_num_colors` | Số màu phẳng tối đa trong palette. |
| `percentile` | Tham số chọn vùng màu phẳng. |
| `strength` | Mức trộn từ output model sang màu palette, trong khoảng `[0, 1]`. |

### Các bước xử lý

1. Kiểm tra dtype, shape và miền giá trị của `strength`.
2. Không làm gì nếu:
   - strength bằng 0;
   - edit mask rỗng;
   - reference mask rỗng.
3. Loại edit mask khỏi reference mask:

   ```text
   trusted_reference = reference_mask AND NOT edit_mask
   ```

   Điều này ngăn pipeline học lại màu từ chính vùng đang cần sửa.

4. Gọi logic có sẵn:

   ```python
   find_flat_color_region(...)
   ```

   để lấy palette từ các vùng màu phẳng đáng tin cậy.

5. Chuyển pixel cần sửa và palette sang LAB.
6. Tính khoảng cách LAB có trọng số:

   ```text
   L  × 1.0
   a  × 0.5
   b  × 0.5
   ```

7. Với mỗi pixel cần refine, chọn màu palette gần nhất.
8. Trộn:

   ```text
   refined =
       original_generated × (1 - strength)
       + nearest_palette × strength
   ```

9. Chỉ ghi kết quả vào `edit_mask`; mọi pixel khác giữ nguyên.

### Tác dụng

- Không ép toàn bộ vùng reconstructed thành một màu.
- Vẫn giữ một phần texture/detail do HD-Painter sinh.
- Kéo màu nhạt hoặc xám về gần màu thật của target.
- Không lấy màu của occluder hoặc background để refine target.

---

## 5.2. Refactor `refine_background()` dùng chung logic

### File được sửa

`backend/core/layerd_refine.py`

`refine_background()` trước đây tự:

- tìm palette;
- tính LAB distance;
- thay pixel bằng màu gần nhất.

Code trùng lặp này được thay bằng lời gọi:

```python
refine_with_reference_mask(...)
```

Đối với background:

```text
edit_mask = từng connected component của vùng background đã inpaint
reference_mask = vành ngoài component
strength = 1.0
```

### Tác dụng

- Background refine và reconstruction refine dùng cùng một lõi thuật toán.
- Giảm duplicate code.
- Các sửa đổi LAB/palette sau này được áp dụng nhất quán cho cả hai luồng.
- Hành vi background vẫn giữ nguyên về ý nghĩa: dùng vùng xung quanh làm màu tham chiếu và thay hoàn toàn theo palette gần nhất.

---

## 5.3. Reconstruction refine dùng modal target làm reference

### File được sửa

`backend/pipeline/reconstruction/validate_reconstruction.py`

### Hàm được refactor

`refine_reconstruction_colors(...)`

Mapping cụ thể:

```text
image
    = validated HD-Painter output

edit_mask
    = reconstruction_mask/composition_mask

reference_image
    = source crop gốc

reference_mask
    = modal mask gốc của target
```

Đây là điểm khác biệt chính so với refine background:

| Background refine | Reconstruction refine |
|---|---|
| Lấy màu từ vùng nền bao quanh component. | Lấy màu từ phần modal đang nhìn thấy của chính target. |
| Strength cố định bằng `1.0`. | Strength cấu hình được, hiện tại `0.35`. |
| Thay màu mạnh để làm phẳng nền. | Điều chỉnh nhẹ để giữ detail do HD-Painter sinh. |

### Tác dụng

Ví dụ target là người áo đen bị camera che:

- `reference_mask` chỉ chứa modal mask của người áo đen;
- palette được lấy từ áo/quần/phần target đang thấy;
- pixel reconstruction không học màu xám của camera hoặc màu đỏ của background;
- chỉ phần người bị thiếu trong `reconstruction_mask` được điều chỉnh;
- phần modal gốc của người không bị sửa.

---

## 5.4. Vị trí refine trong reconstruction flow

### File được sửa

`backend/pipeline/reconstruction/reconstruction.py`

Thứ tự hiện tại:

1. HD-Painter trả `model_output`.
2. `validate_reconstruction_result(...)`.
3. `reconstruction_color_metrics(...)`.
4. Nếu bật config, gọi `refine_reconstruction_colors(...)`.
5. Gán kết quả vào:

   ```python
   detected.reconstruction_canvas
   ```

6. Lưu:

   ```text
   validated_output.png
   color_refined_output.png
   ```

Refine được chạy sau validation để:

- không che giấu lỗi cấu trúc/output cơ bản của model;
- metrics vẫn mô tả output đã validate trước refine;
- diagnostics có thể so sánh rõ trước và sau refine.

---

## 5.5. Soft color metrics

### File được sửa

`backend/pipeline/reconstruction/validate_reconstruction.py`

### Hàm

`reconstruction_color_metrics(...)`

Metrics hiện tại bao gồm:

- khoảng cách màu từ reconstructed RGB tới palette target;
- khoảng cách màu tới palette occluder;
- tỷ lệ pixel giống source cũ;
- tỷ lệ chi tiết được model thực sự tạo.

Các metrics này được log dưới event:

```text
object_reconstruction / color_assessment
```

### Tác dụng

Đây là validation mềm:

- dùng để quan sát chất lượng và tuning;
- không reject output chỉ vì màu lệch nhẹ;
- tránh làm pipeline fallback quá nhiều trong khi model vẫn tạo đúng cấu trúc.

---

## 6. Cấu hình được thêm hoặc thay đổi

### File được sửa

`backend/config.yaml`

```yaml
pipeline:
  object_reconstruction:
    context_ratio: 0.15
    generation_mask_dilation_pixels: 12
    generation_mask_closing_pixels: 7
    support_margin_pixels: 8
    composition_margin_pixels: 4
    blend_allowance_ratio: 0.012
    style_hint: "Match the source's flat vector illustration style with crisp edges, solid consistent colors, and no blur."
    color_refinement_enabled: true
    color_refinement_strength: 0.35
    diagnostics_directory: null
```

| Cấu hình | Tác dụng |
|---|---|
| `context_ratio` | Context quanh target khi tạo ROI. |
| `composition_margin_pixels` | Margin nhỏ để tạo `reconstruction_mask`. |
| `generation_mask_closing_pixels` | Nối khe nhỏ trong generation seed. |
| `generation_mask_dilation_pixels` | Mở rộng mask đưa vào HD-Painter. |
| `blend_allowance_ratio` | Cho phép biên blending nhỏ khi validation. |
| `style_hint` | Hướng HD-Painter về phong cách vector, màu phẳng, cạnh sắc và không blur. |
| `color_refinement_enabled` | Bật/tắt refine màu reconstruction. |
| `color_refinement_strength` | Độ mạnh refine; `0` là giữ nguyên model, `1` là kéo hoàn toàn về palette gần nhất. |
| `diagnostics_directory` | Nơi lưu ảnh trung gian theo từng object. |

### File nối cấu hình vào pipeline

`backend/pipeline/orchestrator.py`

Orchestrator truyền:

- morphology và margin vào `prepare_raw_reconstruction_masks(...)`;
- style hint, refine config và diagnostics vào `reconstruct_objects(...)`;
- `consume_debug_artifacts` của HD-Painter adapter vào diagnostics flow.

---

## 7. Các test được bổ sung hoặc cập nhật

### `tests/test_image_processor.py`

Kiểm tra:

- reconstruction mask chỉ dùng assigned occluder;
- chỉ component được neo vào occluder mới được giữ;
- generation mask chứa toàn bộ occluder bên trong target-centric ROI;
- nhiều occluder được gộp đúng cho từng raw object;
- hỗ trợ reconstruction hai chiều;
- directional decision dùng filtered reconstruction area thay vì total completion-hole area.

### `tests/test_reconstruction_validation.py`

Kiểm tra:

- HD-Painter nhận generation mask;
- composition/reconstruction mask vẫn được giữ riêng;
- target palette refinement giảm contamination từ màu occluder;
- soft color metrics chứa generated detail ratio;
- toàn bộ tên file diagnostics cần thiết được tạo.

### `tests/test_layerd_refine.py`

Kiểm tra:

- chỉ pixel trong `edit_mask` được đổi;
- modal/reference pixels và pixel ngoài edit mask được giữ nguyên;
- nếu không tìm được palette phẳng đáng tin cậy thì output không bị sửa.

### `tests/test_hd_painter_adapter.py`

Kiểm tra:

- adapter xuất được `base_output_512`;
- adapter xuất được `sr_output`;
- `consume_debug_artifacts()` trả và xóa artifacts đúng cách.

Lần verification gần nhất của toàn bộ suite:

```text
223 passed, 1 deselected
```

Đây là verification bằng unit/integration tests cục bộ. Chất lượng hình ảnh cuối cùng vẫn cần được đánh giá bằng ảnh thật trên CUDA vì test tự động không thể kết luận rằng hình dạng hoặc màu do diffusion model sinh ra đã đẹp.

---

## 8. Các file đã được thay đổi trong hai task

| File | Thay đổi chính |
|---|---|
| `backend/pipeline/types.py` | Thêm các trường lưu seed mask, generation seed, relevant occluder và input ROI. |
| `backend/pipeline/reconstruction/mask_preparation.py` | Tạo directional seed, lọc component, xây reconstruction mask và generation mask riêng. |
| `backend/pipeline/reconstruction/reconstruction.py` | Truyền generation mask vào HD-Painter, giữ composition mask cho validation/refine, lưu diagnostics. |
| `backend/pipeline/reconstruction/validate_reconstruction.py` | Validation theo hai mask, thêm soft color metrics và target-aware refine. |
| `backend/core/layerd_refine.py` | Thêm refiner dùng reference mask và cho background/reconstruction dùng chung lõi thuật toán. |
| `backend/models/object_reconstruction/adapter.py` | Cung cấp output phase 512 và SR cho diagnostics. |
| `backend/pipeline/orchestrator.py` | Nối cấu hình mask/refine/diagnostics vào runtime flow. |
| `backend/config.yaml` | Thêm các tham số mask, style hint, refine và diagnostics. |
| `tests/test_image_processor.py` | Test logic mask và directional reconstruction. |
| `tests/test_reconstruction_validation.py` | Test separation của generation/composition mask, metrics, refine và diagnostics. |
| `tests/test_layerd_refine.py` | Test lõi refine dùng reference mask. |
| `tests/test_hd_painter_adapter.py` | Test intermediate artifacts của HD-Painter. |

---

## 9. Kết quả kiến trúc sau refactor

Trước thay đổi:

```text
một mask vừa điều khiển model
         vừa quyết định vùng RGB được sử dụng
```

Vấn đề:

- mask hẹp thì model vẫn thấy occluder và tái tạo occluder;
- mask rộng thì pipeline có nguy cơ sử dụng RGB sinh ra ở quá nhiều vùng;
- dilation có thể phóng đại noise từ completion hole;
- màu output diffusion có thể lệch khỏi target.

Sau thay đổi:

```text
reconstruction_mask
    = vùng RGB target thật sự cần

generation_mask
    = vùng rộng cho HD-Painter xóa occluder và sinh context

modal target
    = nguồn palette đáng tin cậy cho refine
```

Kết quả:

- trách nhiệm của từng mask rõ ràng;
- model có đủ vùng để xử lý occluder;
- vùng write-back vẫn được kiểm soát;
- modal RGB gốc của target được bảo vệ;
- refinement chỉ sửa phần RGB mới sinh;
- diagnostics cho phép lần theo lỗi từ amodal completion đến phase SR và refine.

---

## 10. Giới hạn hiện tại cần lưu ý

1. Refine màu chỉ sửa **màu**, không sửa được hình dạng sai do HD-Painter sinh.
2. Nếu modal mask target chứa nhiều vùng màu khác nhau, palette toàn target có thể chọn đúng màu gần nhất về LAB nhưng chưa chắc đúng bộ phận ngữ nghĩa.
3. `composition_margin_pixels`, closing và dilation vẫn cần tuning theo độ phân giải và chất lượng mask thực tế.
4. `occluder_mask.png` và `full_occluder_in_roi.png` hiện đang lưu cùng một mask.
5. Style hint hiện phù hợp ảnh vector; nếu pipeline xử lý cả ảnh thật, nên chọn style hint theo loại ảnh.
6. Debug artifacts 512/SR hiện được adapter giữ tạm trong RAM cho batch gần nhất, nên diagnostics ở batch lớn có thêm chi phí CPU RAM.

