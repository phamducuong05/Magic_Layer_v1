import numpy as np
import pytest
from PIL import Image


def test_prepare_inputs_aligns_and_binarizes_mask():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
    )

    image = Image.new("RGBA", (20, 10), (10, 20, 30, 255))
    mask = Image.fromarray(np.array([[0, 200]], dtype=np.uint8), mode="L")

    prepared = prepare_inputs(
        image,
        mask,
        resolution=8,
        crop_box=(10, 0, 20, 10),
    )

    assert prepared.image.mode == "RGB"
    assert prepared.image.size == (8, 8)
    assert prepared.mask.size == (8, 8)
    assert set(np.unique(np.asarray(prepared.mask))) <= {0, 255}


def test_prepare_inputs_rejects_empty_mask():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
    )

    with pytest.raises(ValueError, match="empty"):
        prepare_inputs(
            Image.new("RGB", (8, 8)),
            Image.new("L", (8, 8), 0),
            resolution=8,
            crop_box=(0, 0, 8, 8),
        )


def test_prepare_inputs_rejects_non_positive_resolution():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
    )

    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((4, 4), 255)

    with pytest.raises(ValueError, match="resolution"):
        prepare_inputs(
            Image.new("RGB", (8, 8)),
            mask,
            resolution=0,
            crop_box=(0, 0, 8, 8),
        )


def test_small_mask_uses_crop_and_restores_original_size():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
        restore_output,
    )

    image = Image.new("RGB", (12, 8), "blue")
    mask = Image.new("L", image.size, 0)
    mask.putpixel((6, 4), 255)

    prepared = prepare_inputs(
        image,
        mask,
        resolution=8,
        crop_box=(2, 0, 10, 8),
    )
    restored = restore_output(
        Image.new("RGB", (8, 8), "red"),
        image,
        prepared.metadata,
    )

    assert prepared.metadata.mode == "crop"
    assert restored.size == image.size
    assert restored.mode == "RGB"


def test_wide_mask_uses_padding_and_restores_original_size():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
        restore_output,
    )

    image = Image.new("RGB", (12, 8), "blue")
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (1, 2, 11, 6))

    prepared = prepare_inputs(
        image,
        mask,
        resolution=8,
        crop_box=None,
    )
    restored = restore_output(
        Image.new("RGB", (8, 8), "red"),
        image,
        prepared.metadata,
    )

    assert prepared.metadata.mode == "padding"
    assert restored.size == image.size
    assert restored.mode == "RGB"


def test_guidance_crop_keeps_masked_object_on_white():
    from backend.models.background_inpainting.smarteraser.geometry import (
        build_guidance_crop,
    )

    image = Image.new("RGB", (5, 5), "green")
    image.putpixel((2, 2), (0, 0, 255))
    mask = Image.new("L", image.size, 0)
    mask.putpixel((2, 2), 255)

    guidance = build_guidance_crop(image, mask)

    assert guidance.size == (1, 1)
    assert guidance.getpixel((0, 0)) == (0, 0, 255)


def test_restore_output_rejects_wrong_generated_size():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
        restore_output,
    )

    image = Image.new("RGB", (12, 8), "blue")
    mask = Image.new("L", image.size, 0)
    mask.putpixel((6, 4), 255)
    prepared = prepare_inputs(
        image,
        mask,
        resolution=8,
        crop_box=(2, 0, 10, 8),
    )

    with pytest.raises(ValueError, match="generated"):
        restore_output(
            Image.new("RGB", (4, 4)),
            image,
            prepared.metadata,
        )


def test_local_crop_restores_to_original_coordinates():
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
        restore_output,
    )

    image = Image.new("RGB", (1000, 600), "green")
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (300, 200, 400, 280))
    crop_box = (200, 90, 500, 390)

    prepared = prepare_inputs(
        image,
        mask,
        resolution=512,
        crop_box=crop_box,
    )
    restored = restore_output(
        Image.new("RGB", (512, 512), "red"),
        image,
        prepared.metadata,
    )

    assert prepared.image.size == (512, 512)
    assert prepared.mask.size == (512, 512)
    assert prepared.metadata.mode == "crop"
    assert prepared.metadata.crop_box == crop_box
    assert restored.size == image.size
    assert restored.getpixel((250, 150)) == (255, 0, 0)
    assert restored.getpixel((0, 0)) == (0, 128, 0)


@pytest.mark.parametrize(
    ("crop_box", "message"),
    [
        ((-1, 0, 7, 8), "inside"),
        ((0, 0, 7, 8), "square"),
        ((0, 0, 4, 4), "contain"),
    ],
)
def test_prepare_inputs_rejects_invalid_crop_box(crop_box, message):
    from backend.models.background_inpainting.smarteraser.geometry import (
        prepare_inputs,
    )

    image = Image.new("RGB", (8, 8))
    mask = Image.new("L", image.size, 0)
    mask.putpixel((6, 6), 255)

    with pytest.raises(ValueError, match=message):
        prepare_inputs(
            image,
            mask,
            resolution=8,
            crop_box=crop_box,
        )
