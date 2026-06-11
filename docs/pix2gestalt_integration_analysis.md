# Pix2Gestalt Integration Analysis

> Tóm tắt toàn bộ cuộc trò chuyện phân tích pix2gestalt và kế hoạch tích hợp vào Magic_Layer_v1.

---

## 1. Bối cảnh

### Magic Layer (dự án hiện tại)
- Dùng **SAM3** để tách nền → các vật thể kéo thả được
- Dùng **BiRefNet** để alpha matting
- Dùng **LaMa** để inpaint vùng bị che khi objects overlap
- **Vấn đề**: Khi các vật thể đè lên nhau, mask bị khuyết → LaMa chỉ dựa trên màu seed → kết quả không tự nhiên

### Pix2Gestalt (giải pháp)
- Mô hình **amodal completion** dựa trên Stable Diffusion fine-tune
- Input: ảnh bị che + visible mask → Output: vật thể hoàn chỉnh trên nền trắng
- Được train chuyên biệt để **sinh pixel mới** cho vùng bị che

---

## 2. Cấu trúc code pix2gestalt

### File quan trọng

| File | Vai trò |
|---|---|
| `pix2gestalt/inference.py` | Entry point: load model, chạy inference |
| `pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml` | Định nghĩa kiến trúc model |
| `ldm/models/diffusion/ddim.py` | DDIMSampler — thuật toán denoising |
| `ldm/models/diffusion/ddpm.py` | LatentDiffusion — model chính (UNet + VAE + CLIP) |
| `ldm/modules/diffusionmodules/openaimodel.py` | UNetModel — dự đoán noise |
| `ldm/models/autoencoder.py` | AutoencoderKL — VAE encode/decode |
| `ldm/modules/encoders/modules.py` | FrozenCLIPImageEmbedder — CLIP encode |

### Dependencies chính
```
torch, torchvision
omegaconf      ← đọc config YAML
einops         ← rearrange tensor
transformers   ← CLIP
diffusers      ← (tùy chọn, cho VAE)
```

### Checkpoint cần tải
- `epoch=000005.ckpt` hoặc `epoch=000010.ckpt` — fine-tuned Stable Diffusion

---

## 3. Luồng hoạt động chi tiết

### 3.1. inference.py — 6 hàm chính

#### `load_model_from_config(config, ckpt, device)` — Load model
```
Config YAML → instantiate_from_config() → tạo LatentDiffusion model
                                              → Load checkpoint weights
                                              → model.to(device).eval()
```

#### `run_pix2gestalt(model, device, input_im, visible_mask, ...)` — Entry point
```
input_im (256,256,3) numpy → process_input() → tensor (1,3,256,256) [-1,1]
visible_mask (256,256,3)  → process_input() → tensor (1,3,256,256) [-1,1]
                                                    ↓
                                          DDIMSampler(model)
                                                    ↓
                                              sample_model()
                                                    ↓
                                    list of (256,256,3) uint8 [0,255]
```

#### `sample_model(input_im, visible_mask, model, sampler, ...)` — Heart of inference

**Kênh 1 — CLIP cross-attention:**
```
input_im → model.get_learned_conditioning() → (1, 1, 768)
                                    → cc_projection() → (1, 1, 768)
                                    → cond['c_crossattn']
```

**Kênh 2 — VAE concat:**
```
input_im     → encode_first_stage() → (1, 4, 32, 32)
visible_mask → encode_first_stage() → (1, 4, 32, 32)
                              → concat → (1, 8, 32, 32)
                              → cond['c_concat']
```

**DDIM sampling:**
```
shape = [4, 32, 32]  ← latent shape
sampler.sample(S=200, conditioning=cond, ...)
→ samples_ddim: (n_samples, 4, 32, 32)
```

**VAE decode:**
```
samples_ddim → decode_first_stage() → (n_samples, 3, 256, 256) [-1,1]
                            → clamp to [0,1] → CPU
```

**Đầu ra:** `torch.Tensor` shape `(n_samples, 3, 256, 256)`, float32, [0.0, 1.0]

#### `run_pix2gestalt()` — Convert output
```
tensor (N, 3, 256, 256) float32 [0,1]
  → rearrange('chw' → 'hwc')
  → ×255
  → astype(uint8)
  → tách batch thành list
```

**Đầu ra:** `list` of `np.ndarray`, mỗi phần tử `(256, 256, 3)` uint8 [0, 255]

#### `process_input(input_im)` — Tiền xử lý
```
numpy (H,W,3) uint8 [0,255]
  → permute to (3,H,W) → /255 → [0,1]
  → *2-1 → [-1,1]
  → unsqueeze(0) → (1,3,H,W)
```

#### `get_sam_predictor()` + `run_sam()` — Tạo mask từ clicks (không dùng trong inference trực tiếp)

### 3.2. sampler vs model

| | `model` (LatentDiffusion) | `sampler` (DDIMSampler) |
|---|---|---|
| **Là gì** | Mô hình AI (neural network) | Thuật toán (pure code, không có weights) |
| **Chứa gì** | UNet + VAE + CLIP + scheduler params | Schedule arrays + vòng lặp denoising |
| **Học được** | Có, từ training data | Không, logic cố định |
| **Vai trò** | Dự đoán noise `ε` tại mỗi bước | Quyết định bước nào chạy, gọi model thế nào |
| **Thay thế** | Không | Có (DDPM, PLMS, Euler...) |

**Analogie**: Model = người vẽ (có kỹ năng), Sampler = phương pháp vẽ (quy trình từng bước).

### 3.3. DDIMSampler chi tiết

```python
sampler = DDIMSampler(model)
```

**`make_schedule(S, eta)`**: Tính toán `ddim_alphas`, `ddim_sigmas`, `ddim_timesteps` từ 1000 bước DDPM gốc → chọn ra S bước DDIM.

**`sample(S, conditioning, shape, ...)`**: Khởi tạo noise → gọi `ddim_sampling()`.

**`ddim_sampling()`**: Vòng lặp từ t=S về t=0:
```
for step in [S, S-1, ..., 1, 0]:
    → p_sample_ddim(x_t, cond, t)
    → trả về x_{t-1}
```

**`p_sample_ddim()`**: Lõi của mỗi bước:
```
1. Classifier-free guidance:
   - Gọi model 2 lần: có conditioning + không conditioning
   - e_t = e_t_uncond + scale * (e_t - e_t_uncond)

2. Dự đoán x_0:
   - pred_x0 = (x_t - sqrt(1-alpha) * e_t) / sqrt(alpha)

3. Tính hướng đi:
   - dir_xt = sqrt(1 - alpha_prev - sigma^2) * e_t

4. Thêm noise (nếu eta > 0):
   - noise = sigma * randn()

5. Cập nhật:
   - x_{t-1} = sqrt(alpha_prev) * pred_x0 + dir_xt + noise
```

---

## 4. Đầu vào / Đầu ra

### `run_pix2gestalt()` — Hàm chính

| | Input | Output |
|---|---|---|
| **Ảnh** | `(256, 256, 3)` uint8 RGB — ảnh crop chứa vật thể bị che | `list` of `(256, 256, 3)` uint8 RGB — vật thể hoàn chỉnh trên nền trắng |
| **Mask** | `(256, 256, 3)` uint8 — visible mask (0/255, replicate 3 channels) | Trích xuất bằng threshold → `(256, 256)` bool |
| **Model** | LatentDiffusion (SD fine-tune, UNet 12→4 channels) | DDIM 200 steps, CFG scale 2.0 |
| **Device** | CUDA recommended, ~4-6GB VRAM | |

### `sample_model()` — Hàm con

| | Input | Output |
|---|---|---|
| **Tensor** | `(1, 3, 256, 256)` float32 [-1, 1] | `(n_samples, 3, 256, 256)` float32 [0.0, 1.0] |

---

## 5. Giải thích các khái niệm

### Amodal vs Modal

| Khái niệm | Ý nghĩa |
|---|---|
| **Modal (visible)** | Phần vật thể ĐANG NHÌN THẤY |
| **Amodal (full)** | Hình dáng TOÀN VẸN của vật thể (kể cả phần bị che) |
| **Amodal completion** | Phục hồi phần bị che → có hình dáng hoàn chỉnh |

### Classifier-Free Guidance (CFG)

```
e_t = e_t_uncond + scale * (e_t_cond - e_t_uncond)
```

- `scale = 1.0`: Không dùng guidance
- `scale = 2.0`: Cân bằng giữa creativity và fidelity
- `scale > 2.0`: Bám sát conditioning hơn, ít đa dạng hơn

### DDIM (Denoising Diffusion Implicit Models)

- DDPM gốc: 1000 bước, chậm
- DDIM: Có thể dùng ít bước hơn (50-200), nhanh hơn, chất lượng tương đương
- `eta = 1.0`: Fully stochastic (mỗi lần chạy cho kết quả khác nhau)
- `eta = 0.0`: Deterministic (cùng input → cùng output)

---

## 6. Kế hoạch tích hợp vào Magic_Layer

### Vấn đề hiện tại

Khi object A đè lên object B:
1. `_estimate_depth_order()` — heuristic đơn giản, không chính xác
2. `_build_source_image_for_seed()` — seed màu trung bình → LaMa inpaint → kết quả không tự nhiên
3. Vùng overlap trong mask B bị khuyết → BiRefNet cũng không giúp được

### Giải pháp với pix2gestalt

```
Bước 1: Crop vùng object B từ ảnh gốc
        → crop_B: (crop_h, crop_w, 3)

Bước 2: Resize crop_B → 256×256

Bước 3: Tạo visible mask cho object B
        → visible_mask = mask_B & ~mask_A (phần B không bị A đè)
        → Resize mask → 256×256

Bước 4: Gọi run_pix2gestalt(crop_256, mask_256)
        → output: list of (256, 256, 3) — vật thể B hoàn chỉnh

Bước 5: Resize output về kích thước crop gốc
        → output_resized = resize(output, (crop_h, crop_w))

Bước 6: Trích xuất amodal mask
        → gray = mean(output, axis=-1)
        → amodal_mask = gray < 250 (nền trắng)
        → resize mask về (crop_h, crop_w)

Bước 7: Tạo layer RGBA hoàn chỉnh
        → Vùng visible: lấy RGB từ ẢNH GỐC (chính xác 100%)
        → Vùng khuyết: lấy RGB từ PIX2GESTALT OUTPUT
        → Alpha: amodal_mask

Bước 8: Composite hoặc tạo layer kéo thả
```

### Tại sao cần amodal mask?

Pix2Gestalt output là **vật thể trên nền trắng**, không phải trong suốt:

```
OUTPUT:                CẦN TRÍCH XUẤT:
┌──────────────┐       ┌──────────────┐
│ ⬜⬜🐱🐱⬜⬜  │  →    │ ⬜⬜███⬜⬜   │  ← amodal mask
│ ⬜🐱🐱🐱⬜⬜  │  →    │ ⬜█████⬜⬜  │
│ (nền trắng)  │       │ (vùng vật)   │
└──────────────┘       └──────────────┘
```

- **Không thể dán thẳng RGB** → nền trắng sẽ đè lên ảnh gốc
- **Cần amodal mask** để biết vùng nào là vật thể, vùng nào là nền
- **Composite**: chỉ thay thế vùng `amodal_mask & ~visible_mask` (phần khuyết)
- **Layer RGBA**: dùng amodal mask làm alpha channel → vật thể trong suốt xung quanh

### Code mẫu compositing

```python
def composite_amodal_only_recovered(original_rgb, amodal_rgb, amodal_mask, visible_mask, feather_radius=5):
    """
    Chỉ ghép phần ĐƯỢC PHỤC HỒI (vùng bị che) lên ảnh gốc.
    Giữ nguyên vùng visible, chỉ thay thế vùng bị che.
    """
    original = original_rgb.astype(np.float32) / 255.0
    amodal = amodal_rgb.astype(np.float32) / 255.0

    # Vùng được phục hồi = amodal mask TRỪ visible mask
    recovered_mask = ((amodal_mask > 0) & (visible_mask == 0)).astype(np.float32)

    # Feather edge
    if feather_radius > 0:
        recovered_255 = (recovered_mask * 255).astype(np.uint8)
        recovered_blurred = cv2.GaussianBlur(recovered_255, (feather_radius*2+1, feather_radius*2+1), 0)
        recovered_mask = recovered_blurred.astype(np.float32) / 255.0

    recovered_3ch = np.stack([recovered_mask] * 3, axis=-1)
    result = amodal * recovered_3ch + original * (1 - recovered_3ch)

    return (result * 255).astype(np.uint8)
```

### Code mẫu tạo layer RGBA kéo thả

```python
def create_complete_layer(original_image, object_bbox, visible_mask, pix2gestalt_output, amodal_mask):
    """
    Tạo layer RGBA hoàn chỉnh có thể kéo thả.
    - Vùng visible: RGB từ ảnh gốc (chính xác)
    - Vùng khuyết: RGB từ pix2gestalt (phục hồi)
    - Alpha: amodal mask (toàn bộ vật thể)
    """
    x, y, w, h = object_bbox
    layer_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Vùng visible: giữ nguyên từ ảnh gốc
    visible_crop = visible_mask[y:y+h, x:x+w]
    layer_rgba[visible_crop > 0, :3] = original_image[y:y+h, x:x+w][visible_crop > 0]
    layer_rgba[visible_crop > 0, 3] = 255

    # Vùng khuyết: từ pix2gestalt
    recovered = (amodal_mask > 0) & (visible_crop == 0)
    layer_rgba[recovered, :3] = pix2gestalt_output[recovered]
    layer_rgba[recovered, 3] = 255

    return layer_rgba
```

---

## 7. Cấu trúc file notebook

File: `examples/pix2gestalt_demo.ipynb`

| Cell | Nội dung |
|---|---|
| 0 | Cấu hình đường dẫn (input image, mask, checkpoint) |
| 1 | Import & setup device |
| 2 | Load ảnh RGB + visible mask, resize 256×256, visualize |
| 3 | Load pix2gestalt model từ checkpoint + config YAML |
| 4 | Tiền xử lý: numpy → tensor [-1,1] |
| 5 | Encode conditioning: CLIP (768-dim) + VAE (4+4=8 channels) |
| 6 | Chuẩn bị DDIM sampler, visualize timestep schedule |
| 7 | Chạy DDIM sampling (inference chính) |
| 8 | Trích xuất amodal mask, visualize tất cả samples |
| 9 | So sánh chi tiết: visible vs amodal (8-panel) |
| 9b | **Compositing**: ghép vật thể đã complete lên ảnh gốc |
| 10 | Ensemble nhiều samples |
| 11 | Lưu kết quả |

---

## 8. Kết luận

### Điểm mấu chốt

1. **Pix2Gestalt output ≠ ảnh gốc** — là vật thể SINH RA trên nền trắng
2. **Không thể chồng lên nhau** để tái tạo ảnh gốc — background đã mất
3. **Cần amodal mask** để tách vật thể khỏi nền trắng
4. **Cách đúng**: Crop → resize 256 → pix2gestalt → resize về → composite tại bbox gốc
5. **Layer RGBA**: visible (từ ảnh gốc) + recovered (từ pix2gestalt) + alpha (amodal mask)

### Bước tiếp theo

1. Chạy notebook với ảnh + mask thật để kiểm tra kết quả
2. Nếu kết quả tốt → tích hợp vào `image_processor.py`
3. Thay thế bước `_remove_occluder_with_seed()` + LaMa bằng pix2gestalt
4. Tạo layer RGBA hoàn chỉnh cho mỗi object → kéo thả được, không bị khuyết