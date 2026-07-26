import numpy as np
import torch
from PIL import Image

from backend.pipeline.roi import SquareROI
from backend.pipeline.types import DetectedObject


def _object(
    object_id: str,
    modal_mask: np.ndarray,
    *,
    semantic_class: str,
) -> DetectedObject:
    ys, xs = np.nonzero(modal_mask)
    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )
    return DetectedObject(
        object_id=object_id,
        semantic_class=semantic_class,
        display_label=object_id,
        modal_mask=modal_mask.astype(np.uint8) * 255,
        bbox=bbox,
    )


def test_reconstruction_birefnet_uses_raw_output_and_soft_extension_alpha(
    tmp_path,
):
    from backend.pipeline import matting

    shape = (8, 8)
    target_modal = np.zeros(shape, dtype=bool)
    target_modal[3:5, 1:3] = True
    target = _object("person", target_modal, semantic_class="person")
    target.amodal_mask = target_modal.copy()
    target.amodal_mask[3:5, 3] = True
    target.reconstruction_mask = np.zeros(shape, dtype=bool)
    target.reconstruction_mask[3:5, 3] = True
    target.reconstruction_generation_mask = np.zeros(shape, dtype=bool)
    target.reconstruction_generation_mask[3:5, 3:5] = True
    target.reconstruction_accepted_rgb_mask = np.zeros(shape, dtype=bool)
    target.reconstruction_accepted_rgb_mask[3:5, 3:6] = True
    target.reconstruction_foreign_protection_mask = np.zeros(
        shape, dtype=bool
    )
    target.reconstruction_foreign_protection_mask[3:5, 6] = True
    target.reconstruction_roi = SquareROI(0, 0, 8, 8, 8)
    target.occluder_ids = {"camera"}

    source_array = np.full((*shape, 3), 240, dtype=np.uint8)
    source_array[3:5, 3:6] = (100, 100, 100)
    raw_reconstructed = source_array.copy()
    raw_reconstructed[3:5, 3:7] = (5, 5, 5)
    target.raw_reconstruction_canvas = Image.fromarray(
        raw_reconstructed, mode="RGB"
    )
    # The validated fallback intentionally lacks the generated foreground.
    target.reconstruction_canvas = Image.fromarray(source_array, mode="RGB")

    camera_modal = np.zeros(shape, dtype=bool)
    camera_modal[3:5, 3:5] = True
    camera = _object("camera", camera_modal, semantic_class="camera")

    unrelated_modal = np.zeros(shape, dtype=bool)
    unrelated_modal[3:5, 6] = True
    unrelated = _object("book", unrelated_modal, semantic_class="book")

    alpha = torch.zeros(shape, dtype=torch.float32)
    alpha[3:5, 1:7] = 0.95
    matte_inputs = []

    def matte(image):
        matte_inputs.append(np.asarray(image))
        return alpha

    matting.refine_reconstruction_supports(
        Image.fromarray(source_array, mode="RGB"),
        [target, camera, unrelated],
        matte,
        alpha_low_threshold=0.2,
        alpha_high_threshold=0.7,
        change_threshold=8.0,
        connection_margin_pixels=4,
        max_extension_area_ratio=2.0,
        diagnostics_directory=tmp_path,
    )

    assert np.array_equal(matte_inputs[0], raw_reconstructed)
    assert target.reconstruction_extension_mask is not None
    # The original reconstruction core remains writable below the occluder.
    assert np.all(target.reconstruction_write_mask[3:5, 3])
    # Accepted foreground can extend beyond the old generation/amodal masks.
    assert np.all(target.reconstruction_extension_mask[3:5, 4])
    assert np.all(target.reconstruction_extension_mask[3:5, 5])
    # Protected modal pixels outside the target bbox remain excluded.
    assert not np.any(target.reconstruction_extension_mask[3:5, 6])
    assert np.all(target.reconstruction_write_mask[3:5, 4])
    assert np.all(target.reconstruction_write_mask[3:5, 5])
    assert not np.any(target.reconstruction_write_mask[3:5, 6])
    assert np.all(target.reconstruction_support_mask[3:5, 5])
    assert np.all(target.reconstruction_support_mask[target_modal])
    assert np.allclose(
        target.reconstruction_write_alpha[3:5, 3:6], 0.95
    )
    assert not np.any(target.reconstruction_write_alpha[target_modal])
    assert np.array_equal(
        np.asarray(target.reconstruction_canvas), raw_reconstructed
    )
    assert sorted(path.name for path in (tmp_path / "person").iterdir()) == [
        "29_raw_birefnet_alpha.png",
        "30_target_connection_anchor.png",
        "31_birefnet_candidate.png",
        "32_changed_by_model.png",
        "33_generation_evidence.png",
        "34_accepted_target_component.png",
        "35_reconstruction_extension_alpha.png",
        "36_reconstruction_extension_mask.png",
        "37_reconstruction_write_alpha.png",
        "38_reconstruction_write_mask.png",
        "39_composed_reconstruction.png",
        "40_final_reconstruction_support.png",
    ]


def test_reconstruction_support_rejects_disconnected_alpha_component():
    from backend.pipeline import matting

    shape = (8, 8)
    target_modal = np.zeros(shape, dtype=bool)
    target_modal[2:4, 1:3] = True
    target = _object("person", target_modal, semantic_class="person")
    target.amodal_mask = target_modal.copy()
    target.reconstruction_mask = np.zeros(shape, dtype=bool)
    target.reconstruction_mask[2:4, 3] = True
    target.reconstruction_generation_mask = np.zeros(shape, dtype=bool)
    target.reconstruction_generation_mask[2:4, 3:5] = True
    target.reconstruction_accepted_rgb_mask = np.zeros(shape, dtype=bool)
    target.reconstruction_accepted_rgb_mask[2:7, 3:7] = True
    target.reconstruction_roi = SquareROI(0, 0, 8, 8, 8)

    source = np.full((*shape, 3), 255, dtype=np.uint8)
    reconstructed = source.copy()
    reconstructed[2:4, 3:5] = 0
    reconstructed[6, 6] = 0
    target.raw_reconstruction_canvas = Image.fromarray(
        reconstructed, mode="RGB"
    )
    target.reconstruction_canvas = Image.fromarray(source, mode="RGB")

    alpha = torch.zeros(shape, dtype=torch.float32)
    alpha[2:4, 1:5] = 0.95
    alpha[6, 6] = 0.99

    matting.refine_reconstruction_supports(
        Image.fromarray(source, mode="RGB"),
        [target],
        lambda _image: alpha,
        alpha_low_threshold=0.2,
        alpha_high_threshold=0.7,
        change_threshold=8.0,
        connection_margin_pixels=2,
        max_extension_area_ratio=2.0,
    )

    assert target.reconstruction_extension_mask[2, 4]
    assert not target.reconstruction_extension_mask[6, 6]
    assert not target.reconstruction_write_mask[6, 6]
    assert not target.reconstruction_support_mask[6, 6]


def test_reconstruction_support_applies_max_extension_area_ratio():
    from backend.pipeline import matting

    shape = (8, 8)
    target_modal = np.zeros(shape, dtype=bool)
    target_modal[3:5, 1:3] = True
    target = _object("person", target_modal, semantic_class="person")
    target.reconstruction_mask = np.zeros(shape, dtype=bool)
    target.reconstruction_mask[3:5, 3:5] = True
    target.reconstruction_generation_mask = (
        target.reconstruction_mask.copy()
    )
    target.reconstruction_roi = SquareROI(0, 0, 8, 8, 8)

    source = np.full((*shape, 3), 255, dtype=np.uint8)
    reconstructed = source.copy()
    reconstructed[2:6, 3:7] = 0
    target.raw_reconstruction_canvas = Image.fromarray(
        reconstructed, mode="RGB"
    )
    target.reconstruction_canvas = Image.fromarray(source, mode="RGB")

    alpha = torch.zeros(shape, dtype=torch.float32)
    alpha[2:6, 1:7] = 0.95

    matting.refine_reconstruction_supports(
        Image.fromarray(source, mode="RGB"),
        [target],
        lambda _image: alpha,
        alpha_low_threshold=0.2,
        alpha_high_threshold=0.7,
        change_threshold=8.0,
        connection_margin_pixels=2,
        max_extension_area_ratio=1.0,
    )

    assert not np.any(target.reconstruction_extension_mask)
    assert target.reconstruction_canvas is None
