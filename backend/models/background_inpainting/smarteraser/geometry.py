"""Pure image geometry for SmartEraser single-image inference."""

from dataclasses import dataclass
from typing import Literal

from PIL import Image

from .regions import BoundingBox


TransformMode = Literal["crop", "padding"]


@dataclass(frozen=True)
class TransformMetadata:
    """Information required to restore a square inference result."""

    mode: TransformMode
    original_size: tuple[int, int]
    resolution: int
    scaled_size: tuple[int, int] | None = None
    crop_box: BoundingBox | None = None
    padding: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class PreparedInputs:
    """Square source and mask plus their inverse transform."""

    image: Image.Image
    mask: Image.Image
    metadata: TransformMetadata


def _binary_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    aligned = mask.convert("L")
    if aligned.size != size:
        aligned = aligned.resize(size, Image.Resampling.NEAREST)
    return aligned.point(lambda value: 255 if value > 127 else 0)


def _prepare_crop(
    image: Image.Image,
    mask: Image.Image,
    resolution: int,
    crop_box: BoundingBox,
) -> PreparedInputs:
    width, height = image.size
    left, top, right, bottom = crop_box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("SmartEraser crop_box must be inside the image")
    if right - left != bottom - top:
        raise ValueError("SmartEraser crop_box must be square")

    mask_box = mask.getbbox()
    if mask_box is None:
        raise ValueError("SmartEraser mask is empty")
    if not (
        left <= mask_box[0]
        and top <= mask_box[1]
        and right >= mask_box[2]
        and bottom >= mask_box[3]
    ):
        raise ValueError(
            "SmartEraser crop_box must contain the complete mask"
        )

    output_size = (resolution, resolution)
    prepared_image = image.crop(crop_box).resize(
        output_size,
        Image.Resampling.BILINEAR,
    )
    prepared_mask = mask.crop(crop_box).resize(
        output_size,
        Image.Resampling.NEAREST,
    )
    return PreparedInputs(
        image=prepared_image,
        mask=prepared_mask,
        metadata=TransformMetadata(
            mode="crop",
            original_size=image.size,
            resolution=resolution,
            crop_box=crop_box,
        ),
    )


def _prepare_padding(
    image: Image.Image,
    mask: Image.Image,
    resolution: int,
) -> PreparedInputs:
    width, height = image.size
    scale = resolution / max(width, height)
    scaled_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    scaled_image = image.resize(scaled_size, Image.Resampling.BILINEAR)
    scaled_mask = mask.resize(scaled_size, Image.Resampling.NEAREST)

    left = (resolution - scaled_size[0]) // 2
    top = (resolution - scaled_size[1]) // 2
    right = resolution - scaled_size[0] - left
    bottom = resolution - scaled_size[1] - top
    padded_image = Image.new("RGB", (resolution, resolution), "white")
    padded_image.paste(scaled_image, (left, top))
    padded_mask = Image.new("L", (resolution, resolution), 0)
    padded_mask.paste(scaled_mask, (left, top))
    return PreparedInputs(
        image=padded_image,
        mask=padded_mask,
        metadata=TransformMetadata(
            mode="padding",
            original_size=image.size,
            scaled_size=scaled_size,
            resolution=resolution,
            padding=(left, top, right, bottom),
        ),
    )


def prepare_inputs(
    image: Image.Image,
    mask: Image.Image,
    resolution: int,
    crop_box: BoundingBox | None,
) -> PreparedInputs:
    """Normalize and transform one source/mask pair to a square canvas."""

    if resolution <= 0:
        raise ValueError("SmartEraser resolution must be positive")

    source = image.convert("RGB")
    binary_mask = _binary_mask(mask, source.size)
    mask_box = binary_mask.getbbox()
    if mask_box is None:
        raise ValueError("SmartEraser mask is empty")

    if crop_box is None:
        return _prepare_padding(source, binary_mask, resolution)
    return _prepare_crop(
        source,
        binary_mask,
        resolution,
        crop_box,
    )


def build_guidance_crop(
    image: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Return the masked source content cropped on a white background."""

    source = image.convert("RGB")
    binary_mask = _binary_mask(mask, source.size)
    mask_box = binary_mask.getbbox()
    if mask_box is None:
        raise ValueError("SmartEraser mask is empty")
    guidance = Image.composite(
        source,
        Image.new("RGB", source.size, "white"),
        binary_mask,
    )
    return guidance.crop(mask_box)


def restore_output(
    generated: Image.Image,
    original: Image.Image,
    metadata: TransformMetadata,
) -> Image.Image:
    """Apply the inverse square transform and return original-size RGB."""

    expected_size = (metadata.resolution, metadata.resolution)
    if generated.size != expected_size:
        raise ValueError(
            "SmartEraser generated image must match inference resolution"
        )
    if original.size != metadata.original_size:
        raise ValueError("Original image size does not match transform metadata")

    generated_rgb = generated.convert("RGB")
    if metadata.mode == "padding":
        if metadata.padding is None:
            raise ValueError("Padding metadata is missing")
        left, top, right, bottom = metadata.padding
        unpadded = generated_rgb.crop(
            (
                left,
                top,
                metadata.resolution - right,
                metadata.resolution - bottom,
            )
        )
        return unpadded.resize(
            metadata.original_size,
            Image.Resampling.BILINEAR,
        )

    if metadata.crop_box is None:
        raise ValueError("Crop metadata is missing")
    left, top, right, bottom = metadata.crop_box
    restored = original.convert("RGB").copy()
    restored.paste(
        generated_rgb.resize(
            (right - left, bottom - top),
            Image.Resampling.BILINEAR,
        ),
        (left, top),
    )
    return restored
