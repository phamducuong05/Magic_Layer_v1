from unittest.mock import Mock

import numpy as np
from PIL import Image

from backend.pipeline.grouping import group_reconstructed_objects
from backend.pipeline.roi import SquareROI
from backend.pipeline.types import DetectedObject


def _group(*, reconstructed: bool):
    shape = (10, 10)
    modal = np.zeros(shape, dtype=np.uint8)
    modal[4:6, 4:6] = 255
    detected = DetectedObject(
        object_id="object-0",
        semantic_class="person",
        display_label="completed person",
        modal_mask=modal,
        bbox=(4, 4, 2, 2),
    )
    if reconstructed:
        detected.reconstruction_canvas = Image.new("RGB", (2, 2), "red")
    group = group_reconstructed_objects([detected])[0]
    group.amodal_mask = group.modal_mask > 0
    group.amodal_mask[7, 7] = True
    group.matting_roi = SquareROI(2, 2, 6, 10, 10)
    group.matting_source = Image.new("RGB", (6, 6), (20, 180, 60))
    group.soft_alpha = np.zeros(shape, dtype=np.float64)
    group.soft_alpha[4:6, 4:6] = 0.75
    if reconstructed:
        group.soft_alpha[7, 7] = 0.5
    return group


def test_reconstructed_group_layer_uses_matting_source_and_final_alpha_bbox(
    monkeypatch,
):
    from backend.pipeline import layers as layer_stage

    group = _group(reconstructed=True)
    captured = {}

    def build_mask(source_rgb, hard_mask, _kernel_size):
        captured["build_source"] = source_rgb.copy()
        captured["hard_mask"] = hard_mask.copy()
        return hard_mask

    def inpaint(source, mask):
        captured["inpaint_source"] = source
        captured["inpaint_mask"] = np.asarray(mask).copy()
        return Image.new("RGB", source.size, "black")

    def refine(source_rgb, _background, alpha, hard_mask, _kernel_size):
        captured["refine_source"] = source_rgb.copy()
        captured["refine_hard_mask"] = hard_mask.copy()
        refined = np.zeros_like(alpha)
        refined[2:5, 1:4] = 0.8
        return refined, source_rgb.copy()

    encoded = []
    monkeypatch.setattr(layer_stage, "build_inpaint_mask", build_mask)
    monkeypatch.setattr(
        layer_stage,
        "refine_background",
        Mock(side_effect=lambda background, *_args, **_kwargs: background),
    )
    monkeypatch.setattr(layer_stage, "refine_alpha_with_colors", refine)
    monkeypatch.setattr(
        layer_stage,
        "_image_to_base64",
        Mock(side_effect=lambda image: encoded.append(image.copy()) or "rgba"),
    )

    result = layer_stage.extract_object_layers(
        [group],
        (1, 1),
        inpaint,
    )

    assert len(result) == 1
    layer = result[0]
    assert layer.keyword == "completed person"
    assert (layer.x, layer.y, layer.width, layer.height) == (3, 4, 3, 3)
    assert layer.png_base64 == "rgba"
    assert captured["inpaint_source"] is group.matting_source
    assert set(np.unique(captured["inpaint_mask"])) <= {0, 255}
    assert captured["hard_mask"][5, 5]  # Amodal completion in crop coords.
    assert np.all(captured["build_source"] == (20, 180, 60))
    assert np.array_equal(
        captured["refine_source"], captured["build_source"]
    )
    assert np.array_equal(
        captured["refine_hard_mask"], captured["hard_mask"]
    )
    assert encoded[0].mode == "RGBA"
    assert encoded[0].size == (3, 3)
    assert np.all(np.asarray(encoded[0])[..., :3] == (20, 180, 60))


def test_fallback_group_layer_uses_modal_support_not_unrendered_amodal_hole(
    monkeypatch,
):
    from backend.pipeline import layers as layer_stage

    group = _group(reconstructed=False)
    captured = {}

    def build_mask(_source_rgb, hard_mask, _kernel_size):
        captured["hard_mask"] = hard_mask.copy()
        return hard_mask

    monkeypatch.setattr(layer_stage, "build_inpaint_mask", build_mask)
    monkeypatch.setattr(
        layer_stage,
        "refine_background",
        Mock(side_effect=lambda background, *_args, **_kwargs: background),
    )
    monkeypatch.setattr(
        layer_stage,
        "refine_alpha_with_colors",
        Mock(
            side_effect=lambda source, _background, alpha, *_args: (
                alpha,
                source,
            )
        ),
    )
    monkeypatch.setattr(layer_stage, "_image_to_base64", Mock(return_value="x"))
    inpaint = Mock(return_value=Image.new("RGB", (6, 6), "black"))

    result = layer_stage.extract_object_layers([group], (1, 1), inpaint)

    assert len(result) == 1
    assert captured["hard_mask"][2, 2]
    assert not captured["hard_mask"][5, 5]
    inpaint.assert_called_once()
    assert inpaint.call_args.args[0] is group.matting_source


def test_layer_cleanup_removes_weak_fringe_and_tiny_unanchored_component(
    monkeypatch,
):
    from backend.pipeline import layers as layer_stage

    group = _group(reconstructed=False)
    encoded = []

    monkeypatch.setattr(
        layer_stage,
        "build_inpaint_mask",
        Mock(side_effect=lambda _source, hard_mask, _kernel: hard_mask),
    )
    monkeypatch.setattr(
        layer_stage,
        "refine_background",
        Mock(side_effect=lambda background, *_args, **_kwargs: background),
    )

    def refine(source, _background, alpha, *_args):
        refined = np.zeros_like(alpha)
        refined[2:4, 2:4] = 0.8  # Main component, anchored to support.
        refined[2, 4] = 0.02  # Attached weak fringe.
        refined[0, 5] = 0.8  # Detached one-pixel artifact.
        return refined, source

    monkeypatch.setattr(layer_stage, "refine_alpha_with_colors", refine)
    monkeypatch.setattr(
        layer_stage,
        "_image_to_base64",
        Mock(side_effect=lambda image: encoded.append(image.copy()) or "rgba"),
    )

    result = layer_stage.extract_object_layers(
        [group],
        (1, 1),
        Mock(return_value=Image.new("RGB", (6, 6), "black")),
        final_alpha_threshold=0.03,
        final_min_component_area_pixels=4,
    )

    assert len(result) == 1
    assert (result[0].x, result[0].y, result[0].width, result[0].height) == (
        4,
        4,
        2,
        2,
    )
    assert encoded[0].size == (2, 2)
    assert np.all(np.asarray(encoded[0])[..., 3] == 204)
