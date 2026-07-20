"""Prepare aligned square image/mask inputs for an HD-Painter smoke test.

Example:
    python backend/models/object_reconstruction/prepare_hd_painter_data.py \
        --image /path/to/source.png \
        --mask /path/to/reconstruction_mask.png \
        --output-dir test_data/hd_painter

Pass ``--mask`` more than once to retain the component masks and create their
binary union. White output-mask pixels are the region HD-Painter reconstructs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
from PIL import Image


class PreparedData(NamedTuple):
    """Paths generated for one HD-Painter smoke-test sample."""

    image_crop: Path
    combined_mask: Path
    component_masks: tuple[Path, ...]
    preview: Path
    metadata: Path


def _square_box(mask: np.ndarray, padding_ratio: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("The combined reconstruction mask is empty.")

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    object_width = x_max - x_min
    object_height = y_max - y_min
    side = max(1, math.ceil(max(object_width, object_height) * (1 + 2 * padding_ratio)))
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    left = math.floor(center_x - side / 2)
    top = math.floor(center_y - side / 2)
    return left, top, left + side, top + side


def _crop_with_padding(
    array: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    image_padding: bool,
) -> np.ndarray:
    left, top, right, bottom = box
    height, width = array.shape[:2]
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(width, right)
    clipped_bottom = min(height, bottom)
    cropped = array[clipped_top:clipped_bottom, clipped_left:clipped_right]

    pad_width = (
        (max(0, -top), max(0, bottom - height)),
        (max(0, -left), max(0, right - width)),
    )
    if array.ndim == 3:
        pad_width += ((0, 0),)
    if image_padding:
        return np.pad(cropped, pad_width, mode="edge")
    return np.pad(cropped, pad_width, mode="constant", constant_values=0)


def prepare_hd_painter_data(
    *,
    image_path: str | Path,
    mask_paths: Sequence[str | Path],
    output_dir: str | Path,
    padding_ratio: float = 0.25,
) -> PreparedData:
    """Create native-resolution square inputs from a full image and masks."""
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative.")
    if not mask_paths:
        raise ValueError("At least one component mask is required.")

    image_path = Path(image_path).expanduser().resolve()
    resolved_masks = [Path(path).expanduser().resolve() for path in mask_paths]
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    image_array = np.asarray(image)

    masks: list[np.ndarray] = []
    for mask_path in resolved_masks:
        with Image.open(mask_path) as opened_mask:
            mask = np.asarray(opened_mask.convert("L"))
        if mask.shape != image_array.shape[:2]:
            raise ValueError(
                f"Mask dimensions {mask.shape[::-1]} from {mask_path} do not "
                f"match image dimensions {image.size}."
            )
        masks.append(mask > 0)

    combined = np.logical_or.reduce(masks)
    box = _square_box(combined, padding_ratio)
    image_crop = _crop_with_padding(image_array, box, image_padding=True)
    component_crops = [
        _crop_with_padding(mask.astype(np.uint8) * 255, box, image_padding=False)
        for mask in masks
    ]
    combined_crop = np.logical_or.reduce(
        [mask > 0 for mask in component_crops]
    ).astype(np.uint8) * 255

    image_output = output_dir / "image_crop.png"
    combined_output = output_dir / "combined_mask.png"
    preview_output = output_dir / "preview.png"
    metadata_output = output_dir / "metadata.json"
    Image.fromarray(image_crop, mode="RGB").save(image_output)
    Image.fromarray(combined_crop, mode="L").save(combined_output)

    component_outputs: list[Path] = []
    for index, component_crop in enumerate(component_crops):
        component_output = output_dir / f"component_mask_{index:02d}.png"
        Image.fromarray(component_crop, mode="L").save(component_output)
        component_outputs.append(component_output)

    preview = image_crop.copy()
    overlay = np.zeros_like(preview)
    overlay[..., 0] = 255
    selected = combined_crop > 0
    preview[selected] = (
        0.55 * preview[selected] + 0.45 * overlay[selected]
    ).astype(np.uint8)
    Image.fromarray(preview, mode="RGB").save(preview_output)

    metadata = {
        "source_image": str(image_path),
        "source_masks": [str(path) for path in resolved_masks],
        "source_size": list(image.size),
        "square_crop_box_xyxy": list(box),
        "square_size": int(image_crop.shape[0]),
        "padding_ratio": padding_ratio,
        "mask_semantics": "white pixels are reconstructed",
        "image_padding": "edge replication",
        "mask_padding": "zero",
    }
    metadata_output.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return PreparedData(
        image_crop=image_output,
        combined_mask=combined_output,
        component_masks=tuple(component_outputs),
        preview=preview_output,
        metadata=metadata_output,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a full image and component masks to square HD-Painter inputs."
    )
    parser.add_argument("--image", required=True, help="Path to the full RGB image.")
    parser.add_argument(
        "--mask",
        action="append",
        required=True,
        dest="masks",
        help="Path to a full-size component/reconstruction mask; repeat as needed.",
    )
    parser.add_argument(
        "--output-dir",
        default="test_data/hd_painter",
        help="Directory for the generated smoke-test data.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.25,
        help="Context added on every side relative to the largest mask dimension.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = prepare_hd_painter_data(
        image_path=args.image,
        mask_paths=args.masks,
        output_dir=args.output_dir,
        padding_ratio=args.padding_ratio,
    )
    print(f"Square image: {result.image_crop}")
    print(f"Combined mask: {result.combined_mask}")
    print(f"Preview: {result.preview}")
    print(f"Metadata: {result.metadata}")


if __name__ == "__main__":
    main()
