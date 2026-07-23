import numpy as np

from backend.core.layerd_refine import refine_with_reference_mask


def test_reference_mask_refiner_changes_only_edit_pixels():
    source = np.full((12, 12, 3), 255, dtype=np.uint8)
    reference_mask = np.zeros((12, 12), dtype=bool)
    reference_mask[2:10, 1:5] = True
    source[reference_mask] = (220, 20, 20)

    generated = source.copy()
    edit_mask = np.zeros((12, 12), dtype=bool)
    edit_mask[4:8, 7:10] = True
    generated[edit_mask] = (10, 80, 220)
    untouched = generated.copy()

    refined = refine_with_reference_mask(
        generated,
        edit_mask=edit_mask,
        reference_image=source,
        reference_mask=reference_mask,
        max_num_colors=8,
        strength=1.0,
    )

    assert np.all(refined[edit_mask] == (220, 20, 20))
    assert np.array_equal(refined[~edit_mask], untouched[~edit_mask])
    assert np.array_equal(refined[reference_mask], source[reference_mask])


def test_reference_mask_refiner_keeps_output_when_flat_palette_is_unreliable():
    source = np.zeros((12, 12, 3), dtype=np.uint8)
    yy, xx = np.indices((12, 12))
    source[..., 0] = xx * 17
    source[..., 1] = yy * 17
    source[..., 2] = (xx + yy) * 8
    reference_mask = np.ones((12, 12), dtype=bool)
    edit_mask = np.zeros((12, 12), dtype=bool)
    edit_mask[4:8, 4:8] = True
    generated = source.copy()
    generated[edit_mask] = (30, 40, 50)

    refined = refine_with_reference_mask(
        generated,
        edit_mask=edit_mask,
        reference_image=source,
        reference_mask=reference_mask,
        max_num_colors=1,
        strength=1.0,
    )

    assert np.array_equal(refined, generated)
