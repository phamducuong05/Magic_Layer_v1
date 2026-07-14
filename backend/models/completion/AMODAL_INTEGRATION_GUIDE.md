# Amodal Completion Integration Guide

This guide explains how to integrate the Amodal Completion codebase into your existing segmentation pipeline. Amodal completion predicts the full, unoccluded shape of an object even if it is partially hidden by other objects.

## 1. Codebase Overview

To integrate this module into your project, you need the following core components to be copied over:

*   **`models/`**: Contains the neural network architectures (e.g., `AWSDM`, UNet backbones).
*   **`utils/`**: Utility functions for image processing, cropping, padding, and masks manipulation.
*   **`dift/`**: Contains scripts to extract Diffusion Features (DIFT) from the input image using Stable Diffusion. The amodal model relies heavily on these features.
*   **`inference.py`**: The core logic for running the amodal inference. The most important function here is `infer_amodal_aw_sdm`.
*   **`config_SDAmodal.yaml`**: The configuration file (located in `experiments/.../`) that defines the model architecture parameters.
*   **Weights (`.pth`)**: The pre-trained model weights (e.g., `ckpt_SDAmodal.pth`).

## 2. The Pipeline

The amodal completion process typically runs **after** your primary segmentation model has detected the visible parts of the objects (inmodal masks).

The workflow consists of two main steps:
1.  **Feature Extraction**: Extract deep features from the original RGB image using Stable Diffusion (DIFT).
2.  **Amodal Inference**: Feed the visible mask (inmodal), the bounding box, and the extracted features into the SDAmodal model to predict the full mask.

## 3. Step-by-Step Integration

### Step 3.1: Extract DIFT Features
Before running the amodal model, you must extract features for the image. By default, the code expects these features to be saved in directories named `feature/pth0`, `feature/pth1`, etc.

You can use the provided script to extract features:
```bash
python dift/extract_dift_amodal.py --input <path_to_image> --output_dir feature/
```

### Step 3.2: Prepare Inputs from Your Segmentation Model
Your segmentation model should output the visible masks (`inmodal masks`) and their corresponding bounding boxes. The bounding boxes need to be slightly expanded to give the model context.

```python
import numpy as np

# Assuming your segmentation model gives you a list of binary masks (0 or 1)
# modal_masks = [mask1, mask2, ...] 

def masks_to_inputs(modal_masks):
    inmodal, bboxes = [], []
    for m in modal_masks:
        m = (m > 0).astype(np.uint8)
        ys, xs = np.where(m)
        x0, y0 = int(xs.min()), int(ys.min())
        inmodal.append(m)
        # Bounding box format: [x, y, width, height]
        bboxes.append([x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)])
    return np.stack(inmodal, 0), np.array(bboxes, dtype=np.int32)

inmodal, bboxes = masks_to_inputs(modal_masks)

# Expand bounding boxes for better context (using the utils/logic from run_inference.py)
def expand_bbox(bboxes, enlarge_box=3.0):
    new = []
    for b in bboxes:
        cx, cy = b[0] + b[2] / 2., b[1] + b[3] / 2.
        size = max(np.sqrt(b[2] * b[3] * enlarge_box), b[2] * 1.1, b[3] * 1.1)
        new.append([int(cx - size / 2.), int(cy - size / 2.), int(size), int(size)])
    return np.array(new)

# Config enlarge_box from yaml
bboxes = expand_bbox(bboxes, enlarge_box=3.0) 
category = np.ones((len(modal_masks),), dtype=np.int32)
```

### Step 3.3: Load the Amodal Model
Initialize the model using the configuration file and load the weights.

```python
import yaml
import torch
import models

# 1. Load config
with open('config_SDAmodal.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# 2. Initialize model
model = models.__dict__[cfg['model']['algo']](cfg['model'], dist_model=False)

# 3. Load weights and set to eval mode
model.load_state('path/to/ckpt_SDAmodal.pth')
model.switch_to('eval')
```

### Step 3.4: Run Amodal Inference
Call the inference function. Make sure `image_fn` matches the filename used during the DIFT feature extraction so the model can find the `.pt` files.

```python
import inference as infer

image_fn = "example_image.jpg" # Must match the feature file name in feature/pthX/
H, W = inmodal.shape[1], inmodal.shape[2]

# Run prediction
patches = infer.infer_amodal_aw_sdm(
    model, 
    image_fn, 
    inmodal, 
    category, 
    bboxes,
    use_rgb=cfg['model']['use_rgb'], 
    th=0.5, # Threshold for binary mask
    input_size=cfg['data']['input_size'],
    min_input_size=16, 
    interp='nearest', 
    args=None
)

# Reconstruct full image masks from cropped patches
amodal_masks = infer.patch_to_fullimage(patches, bboxes, H, W, interp='linear')

# `amodal_masks` now contains the completed masks!
# Shape: (N, H, W) where N is the number of instances.
```

## 4. Customization Tips for Production

1. **Avoid Disk I/O (Crucial for Speed)**: 
   In `inference.py` -> `infer_amodal_aw_sdm()`, the code currently loads DIFT features from disk using `torch.load(...)`. For a seamless integration into a production segmentation pipeline, you should modify this function to accept the DIFT features directly as Python arguments (e.g., as PyTorch tensors in GPU memory) to avoid the massive overhead of reading/writing large tensor files to the hard drive.
2. **Device Placement**: Ensure that your tensors (features, masks) and the amodal model are consistently on the same device (e.g., `cuda:0`) to avoid memory transfer bottlenecks.
