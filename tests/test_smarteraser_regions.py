from PIL import Image
import math
import pytest


def test_tiny_component_uses_minimum_context_ratio():
    from backend.models.background_inpainting.smarteraser.regions import (
        adaptive_context_box,
    )

    box = adaptive_context_box(
        (100, 100, 120, 120),
        image_size=(1000, 600),
        context_scale=3.0,
        minimum_context_ratio=0.25,
    )

    assert box is not None
    assert box[2] - box[0] == 150
    assert box[3] - box[1] == 150


def test_normal_component_uses_context_scale():
    from backend.models.background_inpainting.smarteraser.regions import (
        adaptive_context_box,
    )

    box = adaptive_context_box(
        (300, 200, 400, 280),
        image_size=(1000, 600),
        context_scale=3.0,
        minimum_context_ratio=0.25,
    )

    assert box is not None
    assert box[2] - box[0] == 300


def test_edge_component_shifts_crop_inside_image():
    from backend.models.background_inpainting.smarteraser.regions import (
        adaptive_context_box,
    )

    box = adaptive_context_box(
        (0, 10, 100, 90),
        image_size=(1000, 600),
        context_scale=3.0,
        minimum_context_ratio=0.25,
    )

    assert box == (0, 0, 300, 300)


def test_component_wider_than_short_side_uses_padding_fallback():
    from backend.models.background_inpainting.smarteraser.regions import (
        adaptive_context_box,
    )

    assert (
        adaptive_context_box(
            (50, 100, 750, 200),
            image_size=(800, 400),
            context_scale=3.0,
            minimum_context_ratio=0.25,
        )
        is None
    )


@pytest.mark.parametrize(
    ("context_scale", "minimum_context_ratio", "message"),
    [
        (0.99, 0.25, "context_scale"),
        (math.nan, 0.25, "context_scale"),
        (math.inf, 0.25, "context_scale"),
        (3.0, 0.0, "minimum_context_ratio"),
        (3.0, 1.01, "minimum_context_ratio"),
    ],
)
def test_context_configuration_is_validated(
    context_scale,
    minimum_context_ratio,
    message,
):
    from backend.models.background_inpainting.smarteraser.regions import (
        adaptive_context_box,
    )

    with pytest.raises(ValueError, match=message):
        adaptive_context_box(
            (10, 10, 20, 20),
            image_size=(100, 100),
            context_scale=context_scale,
            minimum_context_ratio=minimum_context_ratio,
        )


def test_diagonal_pixels_form_one_eight_connected_group():
    from backend.models.background_inpainting.smarteraser.regions import (
        group_mask_components,
    )

    mask = Image.new("L", (10, 10), 0)
    mask.putpixel((3, 3), 255)
    mask.putpixel((4, 4), 255)

    groups = group_mask_components(mask, 3.0, 0.25)

    assert len(groups) == 1
    assert groups[0].bounding_box == (3, 3, 5, 5)


def test_distant_components_form_separate_groups():
    from backend.models.background_inpainting.smarteraser.regions import (
        group_mask_components,
    )

    mask = Image.new("L", (100, 100), 0)
    mask.putpixel((10, 10), 255)
    mask.putpixel((90, 90), 255)

    groups = group_mask_components(mask, 3.0, 0.1)

    assert [group.bounding_box for group in groups] == [
        (10, 10, 11, 11),
        (90, 90, 91, 91),
    ]
    first_mask = groups[0].to_image(mask.size)
    second_mask = groups[1].to_image(mask.size)
    assert first_mask.getpixel((10, 10)) == 255
    assert first_mask.getpixel((90, 90)) == 0
    assert second_mask.getpixel((10, 10)) == 0
    assert second_mask.getpixel((90, 90)) == 255


def test_overlapping_context_boxes_merge_transitively():
    from backend.models.background_inpainting.smarteraser.regions import (
        group_mask_components,
    )

    mask = Image.new("L", (120, 40), 0)
    for x in (20, 32, 44):
        mask.paste(255, (x, 15, x + 5, 20))

    groups = group_mask_components(mask, 3.0, 0.1)

    assert len(groups) == 1
    assert groups[0].bounding_box == (20, 15, 49, 20)


def test_empty_mask_returns_no_groups():
    from backend.models.background_inpainting.smarteraser.regions import (
        group_mask_components,
    )

    assert group_mask_components(
        Image.new("L", (10, 10), 0),
        context_scale=3.0,
        minimum_context_ratio=0.25,
    ) == []


def test_fragmented_mask_groups_store_only_compact_local_masks():
    from backend.models.background_inpainting.smarteraser.regions import (
        group_mask_components,
    )

    mask = Image.new("L", (1000, 1000), 0)
    for y in range(5, 1000, 100):
        for x in range(5, 1000, 100):
            mask.putpixel((x, y), 255)

    groups = group_mask_components(
        mask,
        context_scale=1.0,
        minimum_context_ratio=0.001,
    )

    assert len(groups) == 100
    assert all(group.mask.size == (1, 1) for group in groups)


def test_dense_overlapping_contexts_do_not_compare_every_component_pair(
    monkeypatch,
):
    from backend.models.background_inpainting.smarteraser import regions

    comparison_count = 0
    original_intersection = regions._boxes_intersect_or_touch

    def counting_intersection(first, second):
        nonlocal comparison_count
        comparison_count += 1
        return original_intersection(first, second)

    monkeypatch.setattr(
        regions,
        "_boxes_intersect_or_touch",
        counting_intersection,
    )
    mask = Image.new("L", (100, 100), 0)
    for y in range(10, 90, 4):
        for x in range(10, 90, 4):
            mask.putpixel((x, y), 255)

    groups = regions.group_mask_components(
        mask,
        context_scale=3.0,
        minimum_context_ratio=0.25,
    )

    assert len(groups) == 1
    assert comparison_count < 800
