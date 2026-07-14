"""Run real segmentation and amodal completion, then save mask PNGs."""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def save_masks(
    masks: Sequence[np.ndarray], output_dir: Path
) -> list[Path]:
    """Save masks as binary PNGs and return their paths in input order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for index, mask in enumerate(masks):
        path = output_dir / f"mask_{index}.png"
        binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(path)
        paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM3 grouping and conditional SDAmodal completion."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--keywords",
        required=True,
        help="Comma-separated semantic prompts, for example: person,chair",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    keywords = [
        keyword.strip()
        for keyword in args.keywords.split(",")
        if keyword.strip()
    ]
    if not keywords:
        parser.error("--keywords must contain at least one non-empty prompt")

    from backend.image_processor import process_masks

    with Image.open(args.image) as source:
        masks = process_masks(source.convert("RGB"), keywords)
    paths = save_masks(masks, args.output_dir)

    print(f"Saved {len(paths)} masks to {args.output_dir.resolve()}")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
