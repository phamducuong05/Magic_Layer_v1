# HD-Painter — Detailed Guide

This document provides a detailed explanation of each copied folder's purpose, how to run inference, and all required files/dependencies.

---

## 1. Architecture Overview

HD-Painter is a 2-step inpainting pipeline:

1. **Step 1 — Inpainting at 512px**: Uses a Stable Diffusion Inpainting model (with PAIntA + RASG techniques) to generate an inpainted image at low resolution (512×512).
2. **Step 2 — Super-Resolution to 2048px**: Uses Stable Diffusion x4 Upscaler to upscale the inpainted image to high resolution (2048px), combined with Poisson Blending for seamless compositing with the original image.

```
Input (original image + mask + prompt)
        │
        ▼
┌──────────────────────────┐
│  Step 1: Inpainting 512   │  ← src/methods/rasg.py or sd.py
│  (PAIntA + RASG)           │  ← src/smplfusion/ (custom UNet, attention patches)
│  Model: ds8_inp / sd2_inp  │  ← src/models/inpainting.py
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Step 2: Super-Resolution │  ← src/methods/sr.py
│  SD x4 Upscaler → 2048px  │  ← src/models/sd2_sr.py
│  + Poisson Blending       │  ← src/utils/__init__.py (poisson_blend)
└──────────────────────────┘
        │
        ▼
    Output (HD inpainted image)
```

---

## 2. Folder Details

### 2.1. `src/smplfusion/` — Core Diffusion Engine

This is the **heart** of HD-Painter. Instead of using Hugging Face's `diffusers` library, HD-Painter re-implements the entire Stable Diffusion architecture to allow injection of custom attention techniques (PAIntA, RASG).

| File / Directory | Purpose |
|---|---|
| `__init__.py` | Exports `DDIM`, `share`, `scheduler`, `router`, `attentionpatch`, `transformerpatch` |
| `ddim.py` | `DDIM` class — Denoising Diffusion Implicit Model. Contains the main sampling logic and `get_inpainting_condition()` function to create input conditions for the inpainting UNet |
| `scheduler.py` | `Schedule` class — computes the noise schedule (linear beta schedule), manages `sqrt_alphas`, `sqrt_one_minus_alphas` for each timestep |
| `share.py` | **Global state** module — stores masks at multiple resolutions (8, 16, 32, 64), the current timestep, and the `DDIMIterator` class for tracking denoising progress |
| `util.py` | Utility functions |
| **`models/`** | Custom implementations of model components: |
| ├── `unet.py` | **Custom UNet** — the entire UNet architecture (ResBlock, Downsample, Upsample, TimestepEmbedding) rewritten to allow monkey-patching of attention |
| ├── `vae.py` | **AutoencoderKL** — VAE encoder/decoder |
| ├── `util.py` | Utility modules (GroupNorm, timestep embedding, etc.) |
| └── `encoders/` | Text encoders: `clip_embedder.py` (SD 1.x, uses `transformers`), `open_clip_embedder.py` (SD 2.x, uses `open_clip`) |
| **`modules/`** | Sub-module implementations: |
| ├── `attention/` | Cross-attention, self-attention, spatial transformer, feed-forward blocks |
| ├── `autoencoder.py` | Encoder/Decoder blocks for VAE |
| ├── `distributions.py` | Diagonal Gaussian distribution |
| ├── `ema.py` | Exponential Moving Average (not used during inference) |
| └── `util.py` | Checkpoint, timestep embedding |
| **`patches/`** | **HD-Painter's core techniques**: |
| ├── `router.py` | Forward pass routing — allows swapping between default attention and custom attention (PAIntA) at runtime |
| ├── `attentionpatch/` | `default.py`: standard attention + saves similarity maps; `painta.py`: **PAIntA** — Position-aware Inpainting Token Attention, intervenes in cross-attention to improve quality |
| └── `transformerpatch/` | `default.py`: standard transformer block; `painta.py`: transformer block with injected PAIntA logic |
| **`utils/`** | Input processing: |
| ├── `input_mask.py` | `InputMask` class — resizes masks to multiple resolutions (8, 16, 32, 64) for different UNet layers |
| ├── `input_image.py` | Image input processing |
| └── `input_shape.py` | Shape information management |

### 2.2. `config/` — YAML Configurations

Contains configuration files for instantiating model architectures. The `load_obj()` function in `common.py` reads these YAML files and uses the `__class__` field to instantiate the correct Python class.

| File | Purpose |
|---|---|
| `ddpm/v1.yaml` | DDPM config for SD 1.x: `scale_factor=0.18215`, `timesteps=1000`, linear schedule |
| `ddpm/v2-upsample.yaml` | DDPM config for SD 2.0 Upscaler: `scale_factor=0.08333`, v-prediction, noise augmentation |
| `vae.yaml` | VAE configuration for inpainting (SD 1.x): `ch_mult=[1,2,4,4]` |
| `vae-upsample.yaml` | VAE configuration for upscaler (SD 2.0): `ch_mult=[1,2,4]` |
| `encoders/clip.yaml` | Declares `FrozenCLIPEmbedder` class (SD 1.x text encoder) |
| `encoders/openclip.yaml` | Declares `FrozenOpenCLIPEmbedder` class (SD 2.x text encoder, uses penultimate layer) |
| `unet/inpainting/v1.yaml` | UNet config for SD 1.x inpainting (9 input channels: 4 latent + 1 mask + 4 masked_image) |
| `unet/inpainting/v2.yaml` | UNet config for SD 2.0 inpainting |
| `unet/upsample/v2.yaml` | UNet config for SD 2.0 x4 Upscaler (7 input channels: 4 latent + 3 low-res image) |

### 2.3. `src/models/` — Model Loading & Weight Management

Manages downloading and loading model weights.

| File | Purpose |
|---|---|
| `__init__.py` | Exports: `sd2_sr`, `sam`, `load_inpainting_model`, `pre_download_inpainting_models` |
| `common.py` | **Core loader**: defines `MODEL_FOLDER` (`checkpoints/` directory), `download_file()` function for auto-downloading weights from HuggingFace, `load_sd_inpainting_model()` builds the entire DDIM pipeline, `load_state_dict()` reads `.safetensors`/`.ckpt`/`.bin` files |
| `inpainting.py` | **Model registry** — defines 3 inpainting models: `ds8_inp` (DreamShaper 8), `sd15_inp` (SD 1.5), `sd2_inp` (SD 2.0). Each model has a download URL and weight storage path. Includes a cache mechanism to avoid reloading models |
| `sd2_sr.py` | Loads Stable Diffusion 2.0 x4 Upscaler model + `ImageConcatWithNoiseAugmentation` (adds noise to low-res image before concatenating with latent) |
| `sam.py` | Loads SAM (Segment Anything Model) ViT-H — used in super-resolution for mask refinement (optional, OFF by default) |

**3 available inpainting models:**

| Model ID | Base Model | Source | Weights downloaded from |
|---|---|---|---|
| `ds8_inp` (default) | SD 1.5 | DreamShaper 8 Inpainting | `Lykon/dreamshaper-8-inpainting` on HuggingFace |
| `sd15_inp` | SD 1.5 | Runway SD 1.5 Inpainting | `runwayml/stable-diffusion-inpainting` |
| `sd2_inp` | SD 2.0 | Stability AI SD 2.0 Inpainting | `stabilityai/stable-diffusion-2-inpainting` |

### 2.4. `src/methods/` — Inference Pipelines

Contains `run()` functions that perform inference.

| File | Purpose |
|---|---|
| `rasg.py` | **PAIntA + RASG** (highest quality). Uses gradient-guided sampling: computes BCE score on cross-attention similarity maps → backprop → uses gradient to guide denoising. Requires `requires_grad_(True)` on UNet |
| `sd.py` | **Baseline / PAIntA only** (no RASG). Standard inference without gradients. Faster but lower quality |
| `sr.py` | **Super-Resolution**. Takes inpainted 512px image + original HD image + mask → uses SD x4 Upscaler to upscale to 2048px. Has `blend_trick` option (blends latents during denoising) and `blend_output` option (final Poisson blending) |

**4 available methods:**

| Method | Description | Speed | Quality |
|---|---|---|---|
| `baseline` | Standard SD inpainting | Fastest | Lowest |
| `painta` | + PAIntA attention (injects token attention into mask region) | Fast | Medium |
| `rasg` | + RASG gradient guidance (uses gradient from attention scores) | Slow | High |
| `painta+rasg` (default) | PAIntA + RASG combined | Slowest | **Highest** |

### 2.5. `src/utils/` — Utility Functions

| File | Purpose |
|---|---|
| `__init__.py` | Exports `IImage`, `tokenize`, `resize`, `poisson_blend`, `image_from_url_text` |
| `iimage.py` | `IImage` class — image wrapper, supports conversion between `PIL.Image` ↔ `np.ndarray` ↔ `torch.Tensor`, resize, pad, dilate mask, crop |
| `scores.py` | Attention score functions for RASG: `bce()`, `l1()`, `softmax()` — measures how well cross-attention focuses on the mask region |
| `convert_diffusers_to_sd.py` | Converts state_dict from Diffusers format to original Stable Diffusion format (necessary because `smplfusion` uses a different naming convention than `diffusers`) |

---

## 3. Model Weights — Auto-Download

> **You do NOT need to download weights beforehand.** The system automatically downloads from HuggingFace on first run.

Weights are saved in the `checkpoints/` directory (at the project root level):

```
project_root/
├── checkpoints/                     ← auto-created on first run
│   ├── ds-8-inpainting/             ← DreamShaper 8 Inpainting (~1.7GB)
│   │   ├── unet.fp16.safetensors
│   │   ├── encoder.fp16.safetensors
│   │   └── vae.fp16.safetensors
│   ├── sd-1-5-inpainting/           ← SD 1.5 Inpainting (~1.7GB)
│   │   ├── unet.fp16.safetensors
│   │   ├── encoder.fp16.safetensors
│   │   └── vae.fp16.safetensors
│   ├── sd-2-0-inpainting/           ← SD 2.0 Inpainting (~2.0GB)
│   │   └── 512-inpainting-ema.safetensors
│   ├── sd-2-0-upsample/             ← SD 2.0 x4 Upscaler (~3.4GB)
│   │   └── x4-upscaler-ema.safetensors
│   └── sam/                          ← SAM ViT-H (only when using SAM mask, ~2.5GB)
│       └── sam_vit_h_4b8939.pth
├── config/
├── src/
└── hd_inpaint.py
```

**How it works** (in `src/models/common.py`):
```python
def download_file(url, save_path, chunk_size=1024):
    save_path = Path(save_path)
    if save_path.exists():        # ← If file already exists → skip
        print(f'{save_path.name} exists')
        return
    # ... downloads from HuggingFace if not present
```

**`MODEL_FOLDER` path** is determined by:
```python
PROJECT_DIR = dirname(dirname(dirname(__file__)))  # = project root directory
MODEL_FOLDER = f'{PROJECT_DIR}/checkpoints'
```

> **Note**: If you place the `src/` and `config/` folders in a different location (e.g., inside a subfolder `backend/hd_inpaint/`), then `PROJECT_DIR` will change accordingly, and the `checkpoints/` directory will also be in the corresponding location. Keep this in mind when integrating.

---

## 4. Dependencies (Required Libraries)

### 4.1. Full list (original `requirements.txt`)

```
torch==2.1.1                    # PyTorch (requires CUDA)
torchvision==0.16.1
xformers==0.0.23                # Memory-efficient attention
safetensors==0.3.2              # Reading weight files
omegaconf==2.3.0                # Reading YAML configs
open-clip-torch==2.23.0         # Text encoder for SD 2.x
openai-clip==1.0.1              # Tokenizer
transformers==4.28.0            # Text encoder for SD 1.x (FrozenCLIPEmbedder)
pytorch-lightning==2.1.2        # Only uses seed_everything()
einops==0.7.0                   # Tensor rearrange
scipy==1.10.0                   # binary_dilation in IImage
opencv-python==4.7.*            # Poisson blending, image processing
numpy==1.24.1
Pillow==9.4.0
tqdm==4.66.1
PyYAML==6.0.1
```

### 4.2. Minimum dependencies for inference (inpainting + SR only)

If you are **NOT** using SAM mask refinement (disabled by default), you can skip:
- `segment-anything`
- `mmdet`, `mmengine`, `openmim` (used for evaluation/metrics, not related to inference)
- `gradio` (used for demo UI)
- `pandas` (used for metrics)

**Minimum installation:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install xformers safetensors omegaconf open-clip-torch transformers
pip install pytorch-lightning einops scipy opencv-python numpy Pillow tqdm PyYAML
```

### 4.3. Hardware requirements
- **NVIDIA GPU** with CUDA support (required — all models use `.cuda()`)
- **Minimum VRAM**: ~8GB (with fp16)
- **Recommended VRAM**: ~12GB+ (especially when using `painta+rasg` since it requires `requires_grad_(True)`)

---

## 5. How to Run Inference

### 5.1. Run via command line (original `hd_inpaint.py` script)

```bash
python hd_inpaint.py \
    --image-path path/to/image.jpg \
    --mask-path path/to/mask.png \
    --prompt "a beautiful flower" \
    --output-dir path/to/output \
    --model-id ds8_inp \
    --method painta+rasg \
    --num-steps 50 \
    --guidance-scale 7.5 \
    --seed 1
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--image-path` | (required) | Path to the original image |
| `--mask-path` | (required) | Path to the mask (white region = area to inpaint) |
| `--prompt` | (required) | Text prompt describing the desired content |
| `--output-dir` | (required) | Output directory for results |
| `--model-id` | `ds8_inp` | Model: `ds8_inp`, `sd15_inp`, or `sd2_inp` |
| `--method` | `painta+rasg` | Method: `baseline`, `painta`, `rasg`, `painta+rasg` |
| `--sr-method` | `inpainting_specialized` | SR method: `baseline` or `inpainting_specialized` |
| `--guidance-scale` | `7.5` | Classifier-free guidance scale |
| `--rasg-eta` | `0.1` | RASG gradient strength |
| `--num-steps` | `50` | Number of DDIM denoising steps |
| `--seed` | `1` | Random seed |
| `--num-samples` | `1` | Number of output images to generate |

### 5.2. Run via Python code (project integration)

```python
from PIL import Image
from src import models
from src.methods import rasg, sr
from src.utils import IImage, resize

# ===== Step 1: Load models =====
# Inpainting model (auto-downloads weights on first run)
inp_model = models.load_inpainting_model('ds8_inp', device='cuda:0', cache=True)

# Super-resolution model (auto-downloads weights on first run)
sr_model = models.sd2_sr.load_model(device='cuda:0')

# ===== Step 2: Prepare input =====
image = Image.open('input.jpg').convert('RGB')
mask = Image.open('mask.png').convert('RGB')   # White = area to inpaint
prompt = "a beautiful red rose"

# Resize to 512px for the inpainting step
resized_image = resize(image, 512)
resized_mask = resize(mask, 512)

# ===== Step 3: Inpainting at 512px =====
inpainted = rasg.run(
    ddim=inp_model,
    method='painta+rasg',
    prompt=prompt,
    image=IImage(resized_image),
    mask=IImage(resized_mask),
    seed=1,
    eta=0.1,
    negative_prompt="blurry, low quality",
    positive_prompt="high quality, 4K",
    num_steps=50,
    guidance_scale=7.5
).pil()

# ===== Step 4: Super-Resolution to HD =====
hd_result = sr.run(
    sr_model,
    sam_predictor=None,       # Not using SAM
    lr_image=inpainted,       # Inpainted 512px image
    hr_image=image,           # Original HD image
    hr_mask=mask,             # Original HD mask
    prompt=f"{prompt}, high resolution professional photo",
    noise_level=20,
    blend_trick=True,         # Blend latents during denoising
    blend_output=True,        # Final Poisson blending
    seed=1
)

hd_result.save('output_hd.jpg')
```

### 5.3. Inpainting at 512px only (without Super-Resolution)

If you only need inpainting at 512px (lighter, no need to download the additional SD x4 Upscaler ~3.4GB):

```python
from PIL import Image
from src import models
from src.methods import rasg  # or sd (for baseline/painta)
from src.utils import IImage, resize

inp_model = models.load_inpainting_model('ds8_inp', device='cuda:0', cache=True)

image = Image.open('input.jpg').convert('RGB')
mask = Image.open('mask.png').convert('RGB')

result = rasg.run(
    ddim=inp_model,
    method='painta+rasg',
    prompt="a cat sitting on the grass",
    image=IImage(resize(image, 512)),
    mask=IImage(resize(mask, 512)),
    seed=1
).pil()

result.save('output_512.jpg')
```

---

## 6. Complete Folder Structure Required

Below is **every** file and folder needed to run inference:

```
project_root/
│
├── config/                              ← YAML configs (REQUIRED)
│   ├── ddpm/
│   │   ├── v1.yaml                      ← Config for SD 1.x
│   │   └── v2-upsample.yaml            ← Config for SD 2.0 upscaler
│   ├── encoders/
│   │   ├── clip.yaml                    ← CLIP text encoder (SD 1.x)
│   │   └── openclip.yaml               ← OpenCLIP text encoder (SD 2.x)
│   ├── unet/
│   │   ├── inpainting/
│   │   │   ├── v1.yaml                  ← UNet config for SD 1.x inpainting
│   │   │   └── v2.yaml                  ← UNet config for SD 2.0 inpainting
│   │   └── upsample/
│   │       └── v2.yaml                  ← UNet config for SD 2.0 upscaler
│   ├── vae.yaml                         ← VAE config (inpainting)
│   └── vae-upsample.yaml               ← VAE config (upscaler)
│
├── src/
│   ├── __init__.py                      ← (CREATE if missing — empty file)
│   │
│   ├── smplfusion/                      ← Custom diffusion engine (REQUIRED)
│   │   ├── __init__.py
│   │   ├── ddim.py
│   │   ├── scheduler.py
│   │   ├── share.py
│   │   ├── util.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── unet.py
│   │   │   ├── vae.py
│   │   │   ├── util.py
│   │   │   └── encoders/
│   │   │       ├── clip_embedder.py
│   │   │       └── open_clip_embedder.py
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   ├── autoencoder.py
│   │   │   ├── distributions.py
│   │   │   ├── ema.py
│   │   │   ├── util.py
│   │   │   └── attention/
│   │   │       ├── __init__.py
│   │   │       ├── basic_transformer_block.py
│   │   │       ├── cross_attention.py
│   │   │       ├── feed_forward.py
│   │   │       ├── memory_efficient_cross_attention.py
│   │   │       └── spatial_transformer.py
│   │   ├── patches/
│   │   │   ├── __init__.py
│   │   │   ├── router.py
│   │   │   ├── attentionpatch/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── default.py
│   │   │   │   └── painta.py
│   │   │   └── transformerpatch/
│   │   │       ├── __init__.py
│   │   │       ├── default.py
│   │   │       └── painta.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── input_image.py
│   │       ├── input_mask.py
│   │       └── input_shape.py
│   │
│   ├── models/                          ← Model loading (REQUIRED)
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── inpainting.py
│   │   ├── sd2_sr.py
│   │   └── sam.py
│   │
│   ├── methods/                         ← Inference pipelines (REQUIRED)
│   │   ├── __init__.py
│   │   ├── rasg.py
│   │   ├── sd.py
│   │   └── sr.py
│   │
│   └── utils/                           ← Utilities (REQUIRED)
│       ├── __init__.py
│       ├── iimage.py
│       ├── scores.py
│       └── convert_diffusers_to_sd.py
│
├── checkpoints/                         ← AUTO-CREATED on first run
│
└── hd_inpaint.py                        ← Main inference script (optional)
```

---

## 7. Important Notes for Integration

### 7.1. Import paths

All imports in the code use **absolute imports** starting with `src.`:
```python
from src.smplfusion import DDIM, share, scheduler
from src.models.common import MODEL_FOLDER, load_sd_inpainting_model
from src.utils.iimage import IImage
```

→ If you place the code in a subfolder (e.g., `backend/hd_inpaint/src/`), you need to **rewrite all imports** or add the path to `sys.path`.

### 7.2. `PROJECT_DIR` and `MODEL_FOLDER`

The `checkpoints/` and `config/` paths are computed based on the location of `src/models/common.py`:
```python
PROJECT_DIR = dirname(dirname(dirname(__file__)))
# If common.py is at project_root/src/models/common.py
# → PROJECT_DIR = project_root/
```

→ Make sure the `config/` and `checkpoints/` folder structures are in the correct relative position to `src/models/common.py`.

### 7.3. CUDA is mandatory

The code has many hardcoded `.cuda()` and `device='cuda:0'` calls. It cannot run on CPU without modifying the code.

### 7.4. `src/__init__.py`

Ensure that `src/__init__.py` exists (can be empty) so Python recognizes `src` as a package.

---

## 8. Summary: Pre-Run Checklist

- [ ] Copied all 5 folders: `src/smplfusion/`, `config/`, `src/models/`, `src/methods/`, `src/utils/`
- [ ] `src/__init__.py` file exists (create an empty file if missing)
- [ ] All dependencies are installed (see section 4.2)
- [ ] NVIDIA GPU with CUDA is working (`torch.cuda.is_available() == True`)
- [ ] `config/` directory is at the same level as `src/` (both under `PROJECT_DIR`)
- [ ] Internet connection available for first run (to download weights from HuggingFace)
- [ ] **No need** to pre-download any weight files — the system downloads automatically
