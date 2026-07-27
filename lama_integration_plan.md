# Kế hoạch tích hợp Original LaMa song song với SimpleLama

> **Phạm vi tài liệu:** chỉ phân tích và lập kế hoạch. Chưa thay đổi code tích
> hợp, chưa thay đổi model đang active, chưa chuyển đổi checkpoint.
>
> **Khi triển khai:** thực hiện theo TDD, từng phase có kiểm thử và điểm dừng
> riêng. `simple_lama` hiện tại phải luôn là đường rollback an toàn.

## 1. Mục tiêu

Tích hợp source LaMa gốc trong `lama/` và checkpoint
`backend/models/background_inpainting/original_lama/checkpoints/best.ckpt`
thành một strategy inpaint background mới,
trong khi:

- giữ nguyên adapter SimpleLama đang chạy;
- chọn backend bằng `backend/config.yaml`;
- không để Orchestrator phụ thuộc vào implementation cụ thể;
- dùng đúng chuẩn input/output hiện tại: `PIL.Image RGB + PIL.Image L ->
  PIL.Image RGB`;
- lazy-load model, tái sử dụng trong toàn bộ background stage và giải phóng
  VRAM ở cuối stage;
- có integration test với checkpoint thật để xác nhận Original LaMa chạy
  end-to-end thành công.

Tên backend đề xuất:

- `lama`: SimpleLama cũ, giữ nguyên để tương thích ngược;
- `original_lama`: source LaMa gốc và checkpoint trong dự án;
- `sdxl`: giữ nguyên như hiện tại.

Chuyển đổi chỉ cần:

```yaml
models:
  background_inpainting:
    active: original_lama  # hoặc lama để rollback
```

Việc đổi `active` có hiệu lực khi tiến trình backend khởi động lại. Không đưa
hot-switch giữa request vào scope đầu tiên vì `ConfigManager` hiện đọc config
một lần và `ModelManager` cache một instance theo category.

## 2. Kết quả phân tích codebase

### 2.1 Phạm vi đã quét

Thư mục `lama/` hiện có 250 file, trong đó:

- 98 file Python;
- 59 file `.yaml` và 1 file `.yml`;
- 24 shell script;
- source inference, training, evaluation, refinement và mask generation;
- một nested `.git/`, một số `__pycache__/` và file tài nguyên/model đánh giá.

Toàn bộ 98 file Python đã được parse cú pháp và toàn bộ 60 YAML/YML đã được
parse cấu trúc thành công. Phần được đọc sâu là dependency chain của inference:

- `lama/bin/predict.py`;
- `lama/configs/prediction/default.yaml`;
- `lama/saicinpainting/evaluation/data.py`;
- `lama/saicinpainting/evaluation/refinement.py`;
- `lama/saicinpainting/evaluation/utils.py`;
- `lama/saicinpainting/training/trainers/__init__.py`;
- `lama/saicinpainting/training/trainers/base.py`;
- `lama/saicinpainting/training/trainers/default.py`;
- `lama/saicinpainting/training/modules/__init__.py`;
- `lama/saicinpainting/training/modules/ffc.py`;
- các module phụ thuộc trực tiếp của FFC;
- `lama/bin/make_checkpoint.py`, `lama/bin/to_jit.py`;
- `lama/LaMa_inpainting.ipynb`;
- checkpoint config trong
  `backend/models/background_inpainting/original_lama/checkpoints/config.yaml`.

Các script training, dataset generation và evaluator không cần được gọi bởi
runtime backend; chúng chỉ được giữ làm source tham chiếu.

### 2.2 Luồng SimpleLama hiện tại

Luồng hiện tại đã có sẵn Strategy + Registry/Factory:

1. `BaseBackgroundInpaintingModel.process()` định nghĩa interface chung.
2. `ModelRegistry` ánh xạ tên backend sang adapter.
3. `ModelManager.get_background_inpainting_model()` đọc `active` từ config,
   lazy-load đúng adapter và cache nó trong background stage.
4. `orchestrator.py` chỉ truyền `background_model.process` vào:
   - `extract_object_layers()`;
   - `generate_final_background()`.
5. `LamaBackgroundInpaintingModel` hiện tại:
   - chuyển image sang RGB, mask sang L;
   - gọi `_prepare_inpaint_masks()` để tạo:
     - generation mask giãn rộng;
     - composition mask nhỏ hơn và feather;
   - gọi `SimpleLama(source, generation_mask)`;
   - gọi `_preserve_unmasked_pixels()` để giữ pixel nguồn ngoài composition
     mask.
6. Trong `finally`, Orchestrator gọi
   `release_model("background_inpainting")`; manager xóa reference, chạy GC và
   `torch.cuda.empty_cache()`.

Do kiến trúc chung đã tồn tại, không nên tạo thêm một tầng
`inpainters/factory.py` song song. Tầng đó sẽ trùng trách nhiệm với
`BaseBackgroundInpaintingModel`, `ModelRegistry` và `ModelManager`.

### 2.3 Luồng inference của Original LaMa

`lama/bin/predict.py` thực hiện:

1. Đọc checkpoint config của Big-LaMa.
2. Đặt `training_model.predict_only = True` và visualizer thành `noop`.
3. Tạo model từ training config.
4. Load `models/best.ckpt`, trong đó weights nằm dưới `state_dict`.
5. Đọc image/mask từ dataset:
   - image RGB thành `float32`, range `[0, 1]`, shape `(3, H, W)`;
   - mask grayscale thành `float32`, shape `(1, H, W)`;
   - pad cạnh phải và cạnh dưới đến bội số 8 bằng symmetric padding;
   - ghi lại `(H, W)` ban đầu để unpad.
6. Binarize mask:

   ```text
   mask = (mask > 0)
   ```

7. Trong `DefaultInpaintingTrainingModule.forward()`:

   ```text
   masked_image = image * (1 - mask)
   network_input = concat(masked_image, mask)  # B x 4 x Hpad x Wpad
   predicted_image = FFCResNetGenerator(network_input)
   inpainted = mask * predicted_image + (1 - mask) * image
   ```

8. Lấy output `inpainted`, bỏ padding, clamp `[0, 1]`, đổi về uint8 RGB.
9. Nếu `refine=True`, chạy tối ưu đa tỉ lệ trong
   `evaluation/refinement.py`; đây là luồng nặng hơn đáng kể và có giả định
   riêng về GPU/Kornia.

Lưu ý: `predict.py` hiện gán `device = torch.device("cpu")` dù
`default.yaml` có `device: cuda`. Adapter mới không được sao chép lỗi này; nó
phải dùng `self.device` do project truyền vào.

### 2.4 Rủi ro tương thích đã xác định

Source LaMa là code đời cũ:

- `lama/requirements.txt` pin Hydra 1.1, PyTorch Lightning 1.2.9, Kornia 0.5;
- project hiện yêu cầu Torch >= 2.5, Lightning >= 2.4, OmegaConf >= 2.3;
- environment lock hiện có các version khác nữa;
- `lama/` không có package metadata để import `saicinpainting` trực tiếp;
- `best.ckpt` chứa object metadata của PyTorch Lightning cũ;
- với Torch mới, `torch.load()` mặc định `weights_only=True` và checkpoint
  hiện không load an toàn trực tiếp vì chứa
  `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`;
- thử `weights_only=False` trong interpreter hiện tại cũng không thể mở vì
  thiếu module `pytorch_lightning`.

Không nên đưa toàn bộ dependency training cũ vào runtime production. Mục tiêu
runtime chỉ cần FFC generator và generator weights.

### 2.5 Tiêu chí tích hợp thành công

Không thực hiện so sánh chất lượng với SimpleLama trong phạm vi này.
`original_lama` được coi là tích hợp thành công khi:

- load được source và generator weights;
- chạy inference thật trên CPU và GPU khả dụng;
- nhận/trả đúng contract PIL hiện tại;
- output đúng mode và kích thước input;
- giữ nguyên pixel ngoài composition mask;
- chọn được bằng `models.background_inpainting.active`;
- được lazy-load, tái sử dụng và giải phóng đúng lifecycle;
- toàn bộ regression test của SimpleLama và pipeline hiện tại vẫn pass.

## 3. Các phương án kiến trúc

### Phương án A — Thêm strategy `original_lama` vào Registry hiện tại

Đây là phương án đề xuất.

- Giữ adapter `lama.py` hiện tại.
- Refactor mỗi backend thành package riêng.
- Tạo adapter `original_lama/adapter.py`.
- Tách loader/inference tensor ra `original_lama/runtime.py`.
- Dùng chung `_prepare_inpaint_masks()` và `_preserve_unmasked_pixels()`.
- Chọn backend bằng `models.background_inpainting.active`.

Ưu điểm:

- thay đổi nhỏ nhất ở pipeline;
- Orchestrator không đổi;
- rollback chỉ là một dòng config;
- lifecycle VRAM hiện tại tiếp tục hoạt động;
- unit test từng lớp dễ dàng.

Nhược điểm:

- cần giải quyết packaging của `saicinpainting`;
- cần một bước chuyển checkpoint cũ sang artifact chỉ chứa tensor.

### Phương án B — Tạo thêm `inpainters/base.py` và factory mới

Không đề xuất vì project đã có đúng các abstraction này. Nó tạo hai hệ thống
registry/lifecycle, khiến config và test bị phân tán.

Chỉ xem xét nếu về sau background inpainting bị tách hoàn toàn khỏi hệ thống
model chung; hiện chưa có nhu cầu đó.

### Phương án C — Chạy `lama/bin/predict.py` bằng subprocess

Ưu điểm:

- cô lập dependency cũ trong một environment riêng;
- ít sửa source LaMa.

Nhược điểm:

- phải ghi/đọc file tạm cho mỗi lần gọi;
- load model lại theo process hoặc phải dựng service phụ;
- latency cao, error handling phức tạp;
- không dùng được lifecycle manager hiện tại;
- `predict.py` đang hardcode CPU.

Chỉ dùng làm fallback cho compatibility spike, không phải kiến trúc production.

## 4. Kiến trúc đích

```text
backend/config.yaml
        |
        v
ConfigManager -> ModelRegistry -> ModelManager (lazy instance)
                                  |
                    +-------------+--------------+
                    |                            |
              active: lama              active: original_lama
                    |                            |
      LamaBackgroundInpaintingModel   OriginalLamaBackgroundInpaintingModel
                    |                            |
              SimpleLama              OriginalLamaRuntime
                                                 |
                                   FFCResNetGenerator từ lama/
                                                 |
                             generator_state.pt từ big-lama checkpoint

Hai adapter cùng dùng:
_prepare_inpaint_masks()
_preserve_unmasked_pixels()
```

### 4.1 Cấu trúc thư mục sau refactor

```text
backend/
  models/
    background_inpainting/
      __init__.py
      common/
        __init__.py
        masks.py
      simple_lama/
        __init__.py
        adapter.py
      original_lama/
        __init__.py
        adapter.py
        runtime.py
        checkpoints/
          config.yaml
          best.ckpt
          generator_state.pt
      sdxl/
        __init__.py
        adapter.py
      lama.py                  # compatibility re-export tạm thời
      sdxl.py                  # compatibility re-export tạm thời

lama/                         # source upstream, không trộn code ứng dụng
  saicinpainting/
  configs/
  ...

scripts/
  convert_lama_checkpoint.py
```

Nguyên tắc refactor:

- mỗi backend là một package độc lập;
- adapter chỉ xử lý contract của hệ thống;
- runtime chỉ xử lý model/tensor của backend đó;
- mask/composition dùng chung nằm trong `common/masks.py`;
- `lama/` được coi là vendored upstream source, không chuyển các module
  `saicinpainting` vào `backend/`;
- config/checkpoint được đặt trong `original_lama/checkpoints/`, còn code load
  model nằm ngoài thư mục artifact;
- file compatibility re-export tồn tại tạm thời để các import cũ không vỡ;
- `ModelRegistry` chỉ đăng ký mỗi strategy một lần tại adapter thật, không
  đăng ký lại trong compatibility module.

### 4.2 Trách nhiệm từng module

#### `backend/models/background_inpainting/simple_lama/adapter.py`

- chứa implementation SimpleLama hiện tại sau khi di chuyển khỏi `lama.py`;
- tiếp tục dùng registry key `lama`;
- không thay đổi behavior hiện tại.

#### `backend/models/background_inpainting/original_lama/adapter.py`

- implement `BaseBackgroundInpaintingModel`;
- đọc/validate config của Original LaMa;
- tạo generation mask và blend mask bằng helper chung;
- gọi runtime bằng PIL/NumPy contract;
- preserve pixel nguồn bằng blend mask;
- empty mask trả lại ảnh nguồn mà không chạy model;
- output luôn RGB và đúng size input;
- `unload()` giải phóng runtime/model rồi gọi lifecycle chung.

#### `backend/models/background_inpainting/original_lama/runtime.py`

Chỉ chịu trách nhiệm source-specific:

- locate và import package `saicinpainting`;
- đọc generator config;
- dựng `FFCResNetGenerator`;
- load pure generator state dict;
- chuyển model sang device, `eval()`, tắt gradient;
- preprocess PIL/NumPy thành tensor;
- pad/unpad theo modulo;
- thực hiện forward;
- postprocess tensor về PIL RGB;
- cung cấp `close()` hoặc xóa model reference rõ ràng.

Adapter không được import PyTorch Lightning, trainer, evaluator, dataset loader
hoặc Hydra entrypoint.

#### `backend/models/background_inpainting/sdxl/adapter.py`

- chứa implementation SDXL hiện tại sau khi di chuyển khỏi `sdxl.py`;
- tiếp tục dùng registry key `sdxl`;
- không thay đổi behavior hiện tại.

#### `backend/models/background_inpainting/common/masks.py`

- nhận hai hàm `_prepare_inpaint_masks()` và
  `_preserve_unmasked_pixels()` từ `backend/core/helpers.py`;
- là dependency chung của cả ba background adapter;
- không import model cụ thể;
- giữ nguyên thuật toán và default hiện tại trong commit di chuyển.

`backend/core/helpers.py` tạm thời re-export hai hàm trên để import cũ không
vỡ. Sau khi toàn bộ call site/test được migrate, compatibility re-export có
thể được xóa ở một thay đổi riêng.

#### `scripts/convert_lama_checkpoint.py`

Utility chạy một lần trong environment tin cậy:

- nhận `best.ckpt`;
- xác minh checkpoint path nằm đúng phạm vi người dùng truyền;
- load checkpoint cũ với cảnh báo trust rõ ràng;
- chỉ lấy các key bắt đầu bằng `generator.`;
- bỏ prefix `generator.`;
- ghi artifact chỉ chứa tensor, ví dụ
  `backend/models/background_inpainting/original_lama/checkpoints/generator_state.pt`;
- load lại artifact bằng `weights_only=True`;
- ghi metadata tối thiểu: source checkpoint hash, generator config hash và số
  tensor.

Runtime production không dùng `weights_only=False`.

#### `lama/pyproject.toml` hoặc bootstrap import có kiểm soát

Ưu tiên thêm packaging metadata tối thiểu cho vendored package:

- expose package `saicinpainting`;
- không kéo toàn bộ dependency training cũ;
- runtime dependency chỉ gồm các package thực sự cần bởi FFC path.

Sau đó cài editable/local package cùng project environment.

Nếu packaging làm xung đột build hiện tại, fallback là một bootstrap helper
chèn absolute `lama/` source root vào `sys.path` đúng một lần trước khi import.
Không đặt `PYTHONPATH` bằng side effect toàn hệ thống và không đổi working
directory trong request.

### 4.3 Contract dữ liệu

Input adapter:

```text
image: PIL.Image, mọi mode -> convert("RGB"), size (W, H)
mask:  PIL.Image, mọi mode -> convert("L"), resize NEAREST về (W, H)
```

Mask contract:

- pixel `> 127` được coi là vùng cần inpaint;
- generation mask vẫn do `_prepare_inpaint_masks()` tạo để behavior giữa hai
  backend dùng chung một contract;
- `1` nghĩa là vùng cần model sinh lại;
- `0` nghĩa là vùng đã biết.

Tensor contract:

```text
image_np: float32, H x W x 3, [0, 1]
image:    float32, 1 x 3 x Hpad x Wpad
mask:     float32, 1 x 1 x Hpad x Wpad, {0, 1}
input:    cat(image * (1 - mask), mask), 1 x 4 x Hpad x Wpad
output:   float32, 1 x 3 x Hpad x Wpad, clamp [0, 1]
```

Padding:

- pad cạnh phải/dưới đến `pad_out_to_modulo`, mặc định 8;
- dùng symmetric/reflect behavior tương đương source gốc;
- ghi lại `H, W`, unpad trước khi chuyển về uint8;
- nếu dimension quá nhỏ cho reflect padding thì dùng nhánh an toàn đã được
  test, không để Torch báo lỗi khó hiểu.

Composition:

- output raw từ Original LaMa chỉ là đề xuất generated RGB;
- adapter vẫn gọi `_preserve_unmasked_pixels(source, generated, blend_mask)`;
- nhờ vậy pixel ngoài composition mask giữ nguyên như SimpleLama;
- Orchestrator và layer/background refinement không cần biết backend nào chạy.

### 4.4 Config đề xuất

Không đổi default ngay trong lần tích hợp đầu. Cấu hình mục tiêu:

```yaml
models:
  background_inpainting:
    active: lama

    lama:
      generation_mask_expansion: 21
      composition_mask_expansion: 15
      feather_radius: 2.0

    original_lama:
      source_root: "lama"
      checkpoint_config_path: "backend/models/background_inpainting/original_lama/checkpoints/config.yaml"
      generator_weights_path: "backend/models/background_inpainting/original_lama/checkpoints/generator_state.pt"
      legacy_checkpoint_path: "backend/models/background_inpainting/original_lama/checkpoints/best.ckpt"
      pad_out_to_modulo: 8
      generation_mask_expansion: 21
      composition_mask_expansion: 15
      feather_radius: 2.0
      precision: "fp32"
      refinement:
        enabled: false
        n_iters: 15
        lr: 0.002
        min_side: 512
        max_scales: 3
        px_budget: 1800000

    sdxl:
      # giữ nguyên cấu hình hiện tại
```

Quy tắc validate:

- mọi path phải tồn tại và error message ghi rõ path thiếu;
- `pad_out_to_modulo >= 1`;
- generation/composition expansion là số dương lẻ;
- generation expansion không nhỏ hơn composition expansion;
- feather radius không âm;
- `precision` phase đầu chỉ chấp nhận `fp32`;
- nếu `refinement.enabled=true` nhưng device/config không đáp ứng, fail-fast
  lúc load model, không âm thầm chạy một chế độ khác.

`fp32` là mặc định an toàn. `fp16`/`bf16` không nằm trong phạm vi tích hợp đầu
tiên.

### 4.5 Lifecycle CPU/VRAM

Luồng mong muốn:

1. Server startup chỉ warm-up segmentation như hiện tại.
2. Original LaMa chỉ load khi background stage bắt đầu.
3. Một instance được dùng cho tất cả lần inpaint layer và final background
   trong request.
4. `finally` của Orchestrator gọi
   `release_model("background_inpainting")`.
5. `OriginalLamaBackgroundInpaintingModel.unload()`:
   - xóa runtime/generator reference;
   - không giữ closure chứa CUDA tensor;
   - gọi cleanup chung;
   - manager chạy GC, synchronize và empty CUDA cache.
6. Request sau load lại backend đang được cấu hình.

Phase đầu không giữ Original LaMa trên CPU sau request, vì model khoảng hàng
trăm MB và manager chưa có offload contract cho background inpainting.
CPU offload có thể được thiết kế ở một thay đổi riêng nếu production cần.

Đổi `active` giữa `lama` và `original_lama` khi server đang chạy không nằm
trong phase đầu. Muốn hot-switch an toàn cần thêm cache theo `(category,
strategy_name)` và cơ chế invalidate config; không nên trộn việc đó vào
integration inference.

## 5. Kế hoạch triển khai theo phase

## Phase 0 — Contract và fixture tích hợp

**Không đổi production behavior.**

Files:

- tạo `tests/fixtures/lama_integration/` với vài ảnh/mask nhỏ có license rõ
  ràng.

Các bước:

1. Chuẩn bị các case:
   - nền phẳng;
   - texture tuần hoàn;
   - đường thẳng đi xuyên vùng mask;
   - mask nhỏ;
   - mask lớn;
   - mask chạm biên ảnh;
   - ảnh không chia hết cho 8;
   - ảnh rất nhỏ;
   - mask rỗng;
   - mask toàn ảnh.
2. Chốt tiêu chí bắt buộc:
   - model và weights load thành công;
   - inference thật hoàn tất không exception;
   - output đúng kích thước;
   - không thay pixel ngoài composition mask vượt sai số do blend;
   - không leak model qua stage;
   - SimpleLama tests vẫn pass.
3. Giữ `active: lama` trong lúc xây dựng để code chưa hoàn chỉnh không ảnh
   hưởng đường chạy hiện tại.

## Phase 1 — Compatibility spike và artifact checkpoint an toàn

Files dự kiến:

- tạo `scripts/convert_lama_checkpoint.py`;
- tạo `tests/test_original_lama_checkpoint_conversion.py`;
- có thể tạo `lama/pyproject.toml`;
- không sửa `best.ckpt`.

Các bước TDD:

1. Viết test với checkpoint giả:
   - chỉ key `generator.*` được xuất;
   - prefix bị bỏ đúng;
   - discriminator/loss/optimizer state không đi vào artifact;
   - artifact load lại được với `weights_only=True`.
2. Chạy test và xác nhận fail vì converter chưa tồn tại.
3. Implement converter tối thiểu.
4. Chạy unit test.
5. Tạo environment chuyển đổi được kiểm soát. Không pin downgrade toàn bộ
   runtime app theo `lama/requirements.txt`.
6. Chạy converter trên
   `backend/models/background_inpainting/original_lama/checkpoints/best.ckpt`.
7. Xác minh:
   - số tensor generator khác 0;
   - không còn key `generator.` trong artifact;
   - strict-load được vào generator dựng từ config;
   - một forward smoke test trên CPU chạy được.
8. Ghi SHA-256 của checkpoint nguồn và artifact.
9. Quyết định cách phân phối artifact lớn:
   - Git LFS;
   - release asset;
   - volume deployment;
   - không commit binary trực tiếp vào Git thường.

Điểm dừng: nếu generator không strict-load hoặc forward không chạy trên Torch
hiện tại, không tiếp tục adapter production. Khi đó dùng subprocess legacy chỉ
để đối chiếu và lập danh sách patch tương thích.

## Phase 2 — Refactor package background inpainting

Files dự kiến:

- tạo `backend/models/background_inpainting/common/__init__.py`;
- tạo `backend/models/background_inpainting/common/masks.py`;
- tạo `backend/models/background_inpainting/simple_lama/__init__.py`;
- tạo `backend/models/background_inpainting/simple_lama/adapter.py`;
- tạo `backend/models/background_inpainting/sdxl/__init__.py`;
- tạo `backend/models/background_inpainting/sdxl/adapter.py`;
- sửa `backend/models/background_inpainting/lama.py` thành compatibility
  re-export;
- sửa `backend/models/background_inpainting/sdxl.py` thành compatibility
  re-export;
- sửa `backend/core/helpers.py` thành compatibility re-export cho hai helper
  mask;
- sửa import trong `backend/models/manager.py`;
- cập nhật `tests/test_image_processor.py` và `tests/test_logging_config.py`.

Các bước TDD:

1. Ghi test xác nhận registry vẫn có `lama` và `sdxl` sau khi đổi import.
2. Ghi test xác nhận import cũ từ `background_inpainting.lama`,
   `background_inpainting.sdxl` và `backend.core.helpers` vẫn hoạt động.
3. Chạy test và xác nhận fail trước khi package mới tồn tại.
4. Di chuyển hai helper mask/composition sang `common/masks.py` mà không đổi
   logic.
5. Di chuyển SimpleLama và SDXL adapter sang package riêng, không đổi class,
   registry key hoặc behavior.
6. Thêm compatibility re-export tại đường dẫn cũ.
7. Đổi manager sang import các package adapter thật.
8. Chạy helper, registry, manager, logging và background pipeline tests.
9. Xác minh không có duplicate registry registration.
10. Chạy full suite trước khi bắt đầu thêm Original LaMa.

Commit refactor folder phải độc lập với commit tích hợp model mới. Nếu regression
test fail thì sửa refactor trước, không tiếp tục chồng thêm runtime.

## Phase 3 — Runtime Original LaMa độc lập với pipeline

Files dự kiến:

- tạo
  `backend/models/background_inpainting/original_lama/runtime.py`;
- tạo `backend/models/background_inpainting/original_lama/__init__.py`;
- tạo `tests/test_original_lama_runtime.py`;
- cập nhật dependency runtime tối thiểu trong `backend/requirements.txt`;
- nếu được chọn, tạo packaging metadata cho `lama/`.

Interface mục tiêu:

```text
OriginalLamaRuntime(config: Mapping[str, Any], device: str)
OriginalLamaRuntime.inpaint(image: PIL.Image, mask: PIL.Image) -> PIL.Image
OriginalLamaRuntime.close() -> None
```

Các bước TDD:

1. Test config/path validation bằng temp directory.
2. Test chuyển RGB `HWC uint8` thành `BCHW float32 [0,1]`.
3. Test mask threshold thành `B1HW {0,1}`.
4. Test pad đến modulo 8 và unpad về đúng size ban đầu.
5. Test network input có 4 channel và vùng mask của RGB input bằng 0.
6. Test output composition raw:

   ```text
   inpainted = mask * predicted + (1 - mask) * image
   ```

7. Test output PIL RGB, đúng size và clamp đúng.
8. Test device:
   - CPU dùng CPU;
   - CUDA config chỉ dùng CUDA khi khả dụng;
   - không hardcode CPU như `predict.py`.
9. Test model được `eval()` và inference chạy dưới no-grad/inference mode.
10. Test `close()` xóa reference tới generator.
11. Implement minimal runtime và chạy toàn bộ test.
12. Chạy smoke test thật với artifact generator ở CPU; GPU test đánh dấu
    integration/optional.

Không đưa refinement vào phase này.

## Phase 4 — Adapter strategy và giữ nguyên mask/composition behavior

Files dự kiến:

- tạo `backend/models/background_inpainting/original_lama/adapter.py`;
- cập nhật `backend/models/background_inpainting/original_lama/__init__.py`
  để export adapter;
- cập nhật `backend/models/background_inpainting/__init__.py` nếu cần;
- tạo `tests/test_original_lama_adapter.py`.

Interface giữ nguyên:

```text
process(
    image: PIL.Image,
    mask: PIL.Image,
    prompt: str = "",
) -> PIL.Image
```

Các bước TDD:

1. Mock runtime để test adapter mà không load weights.
2. Xác minh image được convert RGB và mask được convert L.
3. Xác minh mask khác size được resize NEAREST về image size.
4. Xác minh `_prepare_inpaint_masks()` nhận đúng ba config expansion/feather.
5. Xác minh runtime nhận generation mask, không nhận blend mask.
6. Xác minh `_preserve_unmasked_pixels()` dùng blend mask.
7. Xác minh mask rỗng trả source ngay và runtime không được gọi.
8. Xác minh `prompt` được bỏ qua có chủ đích như adapter LaMa hiện tại.
9. Xác minh `unload()` gọi runtime cleanup và không giữ model reference.
10. Implement adapter tối thiểu và chạy test.
11. Chạy regression tests của SimpleLama, common masks và background pipeline.

## Phase 5 — Registry, manager và config toggle

Files dự kiến:

- sửa `backend/models/manager.py` để import module đăng ký strategy mới;
- sửa `backend/config.yaml` để thêm block `original_lama`, nhưng giữ
  `active: lama`;
- sửa `tests/test_completion_manager.py`;
- sửa các config architecture tests liên quan.

Các bước TDD:

1. Test registry trả đúng class cho
   `("background_inpainting", "original_lama")`.
2. Test danh sách config có đủ `lama`, `original_lama`, `sdxl`.
3. Test `active: lama` vẫn tạo adapter cũ với config cũ.
4. Test config giả `active: original_lama` khiến manager tạo adapter mới và
   chỉ truyền config của block đó.
5. Test gọi getter hai lần trong một stage trả cùng instance.
6. Implement import/register/config.
7. Chạy manager/config tests.
8. Chạy pipeline architecture tests để xác nhận Orchestrator không cần đổi.

Không đổi signature của `extract_object_layers()` hoặc
`generate_final_background()`.

## Phase 6 — Lifecycle, lỗi và quan sát runtime

Files dự kiến:

- sửa adapter/runtime nếu lifecycle test chỉ ra thiếu cleanup;
- mở rộng `tests/test_model_lifecycle.py`;
- bổ sung log trong adapter/runtime theo logging convention hiện tại.

Các bước TDD:

1. Test model không load lúc server warm-up.
2. Test model chỉ load lúc background stage.
3. Test cùng instance phục vụ layer extraction và final background.
4. Test success path luôn release background model.
5. Test inference exception vẫn chạy release trong `finally`.
6. Test load exception không để manager cache instance hỏng.
7. Test release bỏ generator reference trước `empty_cache()`.
8. Log các field không nhạy cảm:
   - backend name;
   - device/precision;
   - padded input size;
   - inference duration;
   - checkpoint/artifact identifier, không log toàn bộ config môi trường.
9. Chạy lifecycle tests và pipeline trace tests.

## Phase 7 — Xác minh tích hợp end-to-end

Files dự kiến:

- tạo `tests/test_original_lama_integration.py`, đánh dấu test cần weights.

Các bước:

1. Chạy Original LaMa thật trên fixture không chia hết cho 8.
2. Xác minh:
   - output đúng mode/size;
   - không NaN/Inf;
   - mask rỗng giữ ảnh tuyệt đối;
   - ngoài blend domain giữ ảnh nguồn.
3. Chạy một request pipeline hoàn chỉnh với
   `active: original_lama`.
4. Xác minh adapter được gọi cho cả layer extraction và final background.
5. Xác minh model được release sau request, kể cả khi inference phát sinh
   exception.
6. Chạy full test suite.
7. Sau khi tất cả kiểm thử pass, đặt `active: original_lama`.

Refinement đa tỉ lệ chưa được tích hợp trong phạm vi này. Backend đầu tiên dùng
forward chuẩn với `refinement.enabled: false`; refinement sẽ là thay đổi riêng
nếu được yêu cầu sau.

## Phase 8 — Hoàn tất cấu hình và vệ sinh vendored source

1. Đặt `active: original_lama` sau khi integration test thật pass.
2. Chạy lại smoke test sau khi đổi config mặc định.
3. Theo dõi OOM, load failure và bảo đảm model được release.
4. Nếu có lỗi vận hành, rollback `active: lama`; không cần revert code.
5. Loại nested `lama/.git/`, `__pycache__/` khỏi artifact/repository bằng thao
   tác có kiểm tra phạm vi; giữ `lama/LICENSE`.
6. Ghi rõ provenance/version/commit của source LaMa.
7. Đảm bảo checkpoint lớn dùng cơ chế phân phối phù hợp, không vô tình đưa vào
   Docker layer hoặc Git history nếu không chủ ý.

## 6. Ma trận kiểm thử bắt buộc

| Nhóm | Test |
|---|---|
| Folder refactor | mỗi strategy là package riêng; import tương thích không đăng ký model hai lần |
| Backward compatibility | `active: lama` vẫn chọn SimpleLama và output contract không đổi |
| Registry | `original_lama` đăng ký đúng category |
| Config | path, modulo, expansion, feather, precision được validate |
| Preprocess | RGB/L, threshold, resize mask, BCHW, range `[0,1]` |
| Padding | kích thước chia hết/không chia hết cho 8, ảnh rất nhỏ |
| Inference | input 4 channel, masked RGB bằng 0, eval/no-grad |
| Postprocess | unpad, clamp, uint8, PIL RGB, đúng size |
| Preservation | pixel ngoài composition mask giữ source |
| Edge cases | empty mask, full mask, mask chạm biên, size mismatch |
| Checkpoint | pure tensor artifact, `weights_only=True`, strict-load |
| Lifecycle | lazy-load, reuse trong stage, release khi success/failure |
| Device | CPU smoke; CUDA optional integration; không hardcode CPU |
| Pipeline | layer extraction và final background không đổi interface |
| End-to-end | chạy pipeline thật với `active: original_lama` và checkpoint thật |

Lệnh test dự kiến khi coding:

```powershell
pytest tests/test_original_lama_checkpoint_conversion.py -q
pytest tests/test_original_lama_runtime.py -q
pytest tests/test_original_lama_adapter.py -q
pytest tests/test_completion_manager.py tests/test_model_lifecycle.py -q
pytest tests/test_pipeline_architecture.py tests/test_group_background.py -q
pytest -q
```

Integration test có weights nên cần marker riêng, ví dụ:

```powershell
pytest -m lama_integration tests/test_original_lama_integration.py -q
```

## 7. Thứ tự commit đề xuất khi triển khai

1. `test: preserve background inpainter contracts during package refactor`
2. `refactor: organize background inpainters by strategy`
3. `test: define original lama checkpoint conversion contract`
4. `feat: add safe original lama generator artifact converter`
5. `test: define original lama runtime tensor contract`
6. `feat: add original lama generator runtime`
7. `test: define original lama background adapter behavior`
8. `feat: register configurable original lama background strategy`
9. `test: cover original lama lifecycle and pipeline compatibility`
10. `docs: record original lama setup and rollback procedure`

Mỗi commit phải chạy nhóm test liên quan trước khi chuyển phase. Không commit
checkpoint 410 MB vào Git thường.

## 8. Các quyết định cần duyệt

Đề xuất mặc định:

1. Dùng kiến trúc **Phương án A**: thêm strategy vào Registry hiện tại.
2. Giữ registry key `lama` cho SimpleLama; backend mới tên `original_lama`.
3. Giữ `active: lama` trong commit đầu.
4. Runtime chỉ load FFC generator, không load toàn bộ Lightning training
   module.
5. Chuyển checkpoint một lần thành pure generator state dict an toàn.
6. Runtime đầu tiên dùng `fp32`, `refinement.enabled: false`.
7. Đổi default sang `original_lama` ngay sau khi integration test thật và full
   regression suite pass.
8. Config switch yêu cầu restart backend trong phase đầu.

Nếu các quyết định trên được chấp thuận, bước tiếp theo là coding lần lượt từ
Phase 0 đến Phase 8.
