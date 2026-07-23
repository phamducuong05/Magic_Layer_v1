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


def test_reconstruction_rgb_alpha_extends_incomplete_amodal_support():
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
    target.reconstruction_generation_mask[3:5, 3:7] = True
    target.reconstruction_roi = SquareROI(0, 0, 8, 8, 8)
    target.occluder_ids = {"camera"}

    source_array = np.full((*shape, 3), 240, dtype=np.uint8)
    source_array[3:5, 3:6] = (100, 100, 100)
    reconstructed = source_array.copy()
    reconstructed[3:5, 3:7] = (5, 5, 5)
    target.reconstruction_canvas = Image.fromarray(reconstructed, mode="RGB")

    camera_modal = np.zeros(shape, dtype=bool)
    camera_modal[3:5, 3:6] = True
    camera = _object("camera", camera_modal, semantic_class="camera")

    unrelated_modal = np.zeros(shape, dtype=bool)
    unrelated_modal[3:5, 6] = True
    unrelated = _object("book", unrelated_modal, semantic_class="book")

    alpha = torch.zeros(shape, dtype=torch.float32)
    alpha[3:5, 1:7] = 0.95

    matting.refine_reconstruction_supports(
        Image.fromarray(source_array, mode="RGB"),
        [target, camera, unrelated],
        lambda _image: alpha,
        alpha_low_threshold=0.2,
        alpha_high_threshold=0.7,
        change_threshold=8.0,
        connection_margin_pixels=4,
        max_extension_area_ratio=2.0,
    )

    assert target.reconstruction_extension_mask is not None
    assert np.all(target.reconstruction_extension_mask[3:5, 4:6])
    assert not np.any(target.reconstruction_extension_mask[3:5, 6])
    assert np.all(target.reconstruction_write_mask[3:5, 4:6])
    assert np.all(target.reconstruction_support_mask[3:5, 4:6])
    assert np.all(target.reconstruction_support_mask[target_modal])


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
    target.reconstruction_generation_mask[2:7, 3:7] = True
    target.reconstruction_roi = SquareROI(0, 0, 8, 8, 8)

    source = np.full((*shape, 3), 255, dtype=np.uint8)
    reconstructed = source.copy()
    reconstructed[2:4, 3:5] = 0
    reconstructed[6, 6] = 0
    target.reconstruction_canvas = Image.fromarray(reconstructed, mode="RGB")

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
