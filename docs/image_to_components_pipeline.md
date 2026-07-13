# Image-to-components pipeline

## Data flow

`POST /api/process-image` accepts an image and comma-separated text prompts. The
API validates and optionally downsizes the image, then calls
`backend.image_processor.process_image`.

1. **Text-guided segmentation (SAM3).** The image embedding is computed once.
   Each prompt resets only the prompt state and produces one or more hard object
   masks. Masks are converted to binary 8-bit arrays and resized to the source
   resolution when necessary.
2. **Matting and refinement (BiRefNet + LayerD utilities).** Each SAM mask
   guides BiRefNet, preventing unrelated foreground predictions. A small support
   dilation preserves hair and semi-transparent edge pixels. A temporary LaMa
   background is generated for each visible component and used for foreground
   RGB unblending and alpha refinement. This pass always uses the original image;
   it does not infer depth or reconstruct component overlap. The RGBA result is
   cropped while `x` and `y` retain its canvas position.
3. **Background reconstruction (LaMa or diffusion).** All object masks are
   unioned and expanded to remove edge shadows and color spill. The inpainting
   model runs once on that union mask to create the clean canvas background.
   Component overlap reconstruction is intentionally outside this basic flow.

Models are lazy-loaded by `ModelManager` from `backend/config.yaml`; startup
warm-up makes the first request predictable. The API returns a base64 PNG
background plus base64 RGBA object crops and their canvas coordinates.

## Installation and weights

Install SAM3 itself from this repository, followed by backend dependencies:

```powershell
pip install -e .
pip install -r backend/requirements.txt
```

For local SAM3 weights, place the checkpoint at `checkpoints/sam3.pt`, or change
`models.segmentation.sam3.checkpoint_path`. Alternatively set `load_from_hf` to
`true` and authenticate with Hugging Face when the upstream model is gated.
BiRefNet (`ZhengPeng7/BiRefNet`) and SimpleLaMa download/cache their weights on
first use. Diffusion is optional: select `sdxl` in `config.yaml`, choose a valid
inpainting model ID, and ensure sufficient GPU memory.

Run the service from the repository root:

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
