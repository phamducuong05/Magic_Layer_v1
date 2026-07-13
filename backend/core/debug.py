"""Opt-in image snapshots for diagnosing the inpainting pipeline."""

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

DebugImage = Union[Image.Image, np.ndarray]


def save_inpaint_debug(
    debug_dir: Optional[str],
    label: str,
    **images: DebugImage,
) -> None:
    """Save named images under one stable folder when debugging is enabled."""
    if not debug_dir or not label or not isinstance(debug_dir, (str, Path)):
        return

    output_root = Path(debug_dir)
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parents[2] / output_root

    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("_")
    output_dir = output_root / (safe_label or "inpaint")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, image in images.items():
        if isinstance(image, np.ndarray):
            array = image
            if array.dtype == bool:
                array = array.astype(np.uint8) * 255
            image = Image.fromarray(array)
        image.save(output_dir / f"{name}.png")
