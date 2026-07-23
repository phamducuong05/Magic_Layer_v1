import time

import numpy as np
import torch
from PIL import Image

from backend.models import model_manager


IMAGE_PATH = "test_data/hd_painter/image_crop.png"
MASK_PATH = "test_data/hd_painter/combined_mask.png"
OUTPUT_PATH = "test_data/hd_painter/hd_painter_result_512.png"


image = Image.open(IMAGE_PATH).convert("RGB")
mask = Image.open(MASK_PATH).convert("L")

assert image.width == image.height, "Input crop must be square"
assert mask.size == image.size, "Mask and image must have the same size"

mask_values = np.unique(np.asarray(mask))
assert set(mask_values).issubset({0, 255}), (
    f"Mask must be binary 0/255, received values: {mask_values}"
)
assert np.any(np.asarray(mask) == 255), "Reconstruction mask is empty"

model = model_manager.get_object_reconstruction_model()
assert model is not None, "No object-reconstruction model is configured"

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

started_at = time.perf_counter()

result = model.reconstruct(
    image,
    mask,
    (
        "Continue the hidden parts of the bicycle, preserving its visible "
        "appearance, shape, materials, and surrounding context."
    ),
)

torch.cuda.synchronize()
elapsed = time.perf_counter() - started_at
peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

assert isinstance(result, Image.Image)
assert result.mode == "RGB"
assert result.size == image.size

result.save(OUTPUT_PATH)

print(f"Output: {OUTPUT_PATH}")
print(f"Size: {result.size}")
print(f"Mode: {result.mode}")
print(f"Elapsed: {elapsed:.2f} seconds")
print(f"Peak allocated GPU memory: {peak_gb:.2f} GB")