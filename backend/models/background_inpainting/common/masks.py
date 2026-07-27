"""Mask preparation and source-preserving composition shared by adapters."""

from PIL import Image, ImageFilter


def preserve_unmasked_pixels(
    original: Image.Image,
    inpainted: Image.Image,
    blend_mask: Image.Image,
) -> Image.Image:
    """Blend generated pixels near the mask and preserve distant source pixels."""
    original = original.convert("RGB")
    if inpainted.size != original.size:
        inpainted = inpainted.resize(original.size, Image.Resampling.LANCZOS)
    inpainted = inpainted.convert("RGB")

    blend_mask = blend_mask.convert("L").resize(
        original.size, Image.Resampling.BILINEAR
    )
    return Image.composite(inpainted, original, blend_mask)


def prepare_inpaint_masks(
    mask: Image.Image,
    generation_expansion: int = 17,
    composition_expansion: int = 11,
    feather_radius: float = 2.0,
) -> tuple[Image.Image, Image.Image]:
    """Create a wide model mask and a smaller feathered composition mask."""
    if generation_expansion < composition_expansion:
        raise ValueError("generation_expansion must cover composition_expansion")
    if generation_expansion < 1 or generation_expansion % 2 == 0:
        raise ValueError("generation_expansion must be a positive odd integer")
    if composition_expansion < 1 or composition_expansion % 2 == 0:
        raise ValueError("composition_expansion must be a positive odd integer")
    if feather_radius < 0:
        raise ValueError("feather_radius must be non-negative")

    binary_mask = mask.convert("L").point(
        lambda value: 255 if value > 127 else 0
    )
    generation_mask = binary_mask.filter(
        ImageFilter.MaxFilter(generation_expansion)
    )
    composition_core = binary_mask.filter(
        ImageFilter.MaxFilter(composition_expansion)
    )
    blend_mask = composition_core.filter(
        ImageFilter.GaussianBlur(feather_radius)
    )
    return generation_mask, blend_mask
