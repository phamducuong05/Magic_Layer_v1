from unittest.mock import Mock

import numpy as np
from PIL import Image

from backend.pipeline.types import DetectedObject, GroupedObject


def _reconstructed_group() -> GroupedObject:
    shape = (9, 9)
    modal = np.zeros(shape, dtype=np.uint8)
    modal[2:4, 2:4] = 255
    amodal = modal.astype(bool)
    amodal[7, 7] = True

    member = DetectedObject(
        object_id="object-0",
        semantic_class="person",
        display_label="person",
        modal_mask=modal,
        bbox=(2, 2, 2, 2),
    )
    member.reconstruction_canvas = Image.new("RGB", (5, 5), "red")

    group = GroupedObject(
        group_id="group-0",
        semantic_class="person",
        display_label="person",
        member_ids=(member.object_id,),
        members=(member,),
        modal_mask=modal,
        amodal_mask=amodal,
        bbox=(2, 2, 6, 6),
        segmentation_index=0,
    )
    group.composed_source = Image.new("RGB", (7, 7), "green")
    group.matting_source = Image.new("RGB", (9, 9), "blue")
    group.soft_alpha = np.zeros(shape, dtype=np.float64)
    group.soft_alpha[2:4, 2:4] = 1.0
    group.soft_alpha[4, 3] = 0.25  # Visible anti-aliased edge.
    group.soft_alpha[7, 7] = 0.8  # Reconstructed completion hole.
    return group


def test_final_background_excludes_reconstructed_amodal_alpha(
    monkeypatch,
):
    from backend.pipeline import background as background_stage

    source = Image.new("RGB", (9, 9), "white")
    group = _reconstructed_group()
    captured = {}

    def inpaint(image, mask):
        captured["image"] = image
        captured["mask"] = np.asarray(mask).copy()
        return Image.new("RGB", image.size, "black")

    monkeypatch.setattr(
        background_stage,
        "refine_background",
        Mock(side_effect=lambda background, *_args, **_kwargs: background),
    )

    result = background_stage.generate_final_background(
        source, [group], (1, 1), inpaint
    )

    removal = captured["mask"] > 0
    assert captured["image"] is source
    assert removal[2, 2]
    assert removal[4, 3]
    assert not removal[7, 7]
    assert result.size == source.size


def test_final_background_inpaints_original_image_once(monkeypatch):
    from backend.pipeline import background as background_stage

    source = Image.new("RGB", (9, 9), "white")
    group = _reconstructed_group()
    inpaint = Mock(return_value=Image.new("RGB", source.size, "black"))
    monkeypatch.setattr(
        background_stage,
        "refine_background",
        Mock(side_effect=lambda background, *_args, **_kwargs: background),
    )

    background_stage.generate_final_background(
        source, [group], (1, 1), inpaint
    )

    inpaint.assert_called_once()
    assert inpaint.call_args.args[0] is source
    assert inpaint.call_args.args[0] is not group.composed_source
    assert inpaint.call_args.args[0] is not group.matting_source

