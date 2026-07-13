"""Unit tests for the image-to-components orchestration pipeline.

These tests use small deterministic fakes. They never load SAM3, BiRefNet, LaMa,
or model weights, so failures point to pipeline wiring rather than model quality.
"""

import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="the backend requires PyTorch")
pytest.importorskip("cv2", reason="the image pipeline requires OpenCV")

# Importing the real registry would import every heavyweight model adapter and
# its optional dependencies. image_processor only needs the manager interface,
# which each test configures with deterministic fakes.
models_module = types.ModuleType("backend.models")
models_module.model_manager = Mock()
sys.modules["backend.models"] = models_module

from backend import image_processor as pipeline


@pytest.fixture
def rgb_image() -> Image.Image:
    """Return a small image with a visible square component."""
    array = np.full((6, 8, 3), 20, dtype=np.uint8)
    array[1:5, 2:6] = (180, 80, 40)
    return Image.fromarray(array, mode="RGB")


class FakeSamProcessor:
    """Minimal stateful replacement for the SAM3 image processor."""

    def __init__(self):
        self.prompts = []
        self.reset_count = 0
        self.image = None

    def set_image(self, image):
        self.image = image
        return {}

    def reset_all_prompts(self, state):
        self.reset_count += 1

    def set_text_prompt(self, state, prompt):
        self.prompts.append(prompt)
        if prompt == "missing":
            state.update(masks=[], scores=[])
            return state

        mask = torch.zeros((1, 6, 8), dtype=torch.float32)
        mask[:, 1:5, 2:6] = 1
        state.update(masks=[mask], scores=torch.tensor([0.91]))
        return state


def _install_segmentation_fake(monkeypatch, processor):
    segmentation_model = Mock()
    segmentation_model.get_processor.return_value = processor
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_segmentation_model",
        Mock(return_value=segmentation_model),
    )


def _install_matting_fake(monkeypatch, alpha):
    matting_model = Mock()
    matting_model.process.return_value = alpha
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_matting_model",
        Mock(return_value=matting_model),
    )
    return matting_model.process


def _install_inpainting_fake(monkeypatch, output):
    inpainting_model = Mock()
    inpainting_model.process.return_value = output
    monkeypatch.setattr(
        pipeline.model_manager,
        "get_inpainting_model",
        Mock(return_value=inpainting_model),
    )
    return inpainting_model.process


def test_extract_raw_masks_uses_each_nonempty_prompt(monkeypatch, rgb_image):
    processor = FakeSamProcessor()
    _install_segmentation_fake(monkeypatch, processor)

    masks, labels = pipeline._extract_raw_masks(
        rgb_image, [" component ", "", "missing"]
    )

    assert processor.image is rgb_image
    assert processor.prompts == ["component", "missing"]
    assert processor.reset_count == 2
    assert labels == ["component"]
    assert len(masks) == 1
    assert masks[0].shape == (6, 8)
    assert masks[0].dtype == np.uint8
    assert set(np.unique(masks[0])) == {0, 255}


def test_extract_raw_masks_merges_overlapping_same_keyword(monkeypatch, rgb_image):
    processor = FakeSamProcessor()

    def two_masks(state, prompt):
        first = torch.zeros((1, 6, 8), dtype=torch.float32)
        first[:, 1:4, 1:4] = 1
        second = torch.zeros((1, 6, 8), dtype=torch.float32)
        second[:, 2:5, 3:6] = 1
        state.update(masks=[first, second], scores=torch.tensor([0.9, 0.8]))
        return state

    processor.set_text_prompt = two_masks
    _install_segmentation_fake(monkeypatch, processor)

    masks, labels = pipeline._extract_raw_masks(rgb_image, ["button"])

    assert len(masks) == 1
    assert labels == ["button"]
    assert np.all(masks[0][1:4, 1:4] == 255)
    assert np.all(masks[0][2:5, 3:6] == 255)


def test_extract_raw_masks_keeps_nonoverlapping_same_keyword(
    monkeypatch, rgb_image
):
    processor = FakeSamProcessor()

    def two_masks(state, prompt):
        first = torch.zeros((1, 6, 8), dtype=torch.float32)
        first[:, 0:2, 0:2] = 1
        second = torch.zeros((1, 6, 8), dtype=torch.float32)
        second[:, 4:6, 6:8] = 1
        state.update(masks=[first, second], scores=torch.tensor([0.9, 0.8]))
        return state

    processor.set_text_prompt = two_masks
    _install_segmentation_fake(monkeypatch, processor)

    masks, labels = pipeline._extract_raw_masks(rgb_image, ["button"])

    assert len(masks) == 2
    assert labels == ["button_0", "button_1"]


def test_refine_masks_guides_matting_and_limits_alpha(monkeypatch, rgb_image):
    raw_mask = np.zeros((6, 8), dtype=np.uint8)
    raw_mask[2:4, 3:5] = 255
    predicted_alpha = torch.ones((6, 8), dtype=torch.float32)
    matting = _install_matting_fake(monkeypatch, predicted_alpha)

    result = pipeline._refine_masks(np.asarray(rgb_image), [raw_mask])

    assert len(result) == 1
    assert result[0].shape == (6, 8)
    assert result[0].dtype == np.float64
    assert np.all((0 <= result[0]) & (result[0] <= 1))

    guided_image = np.asarray(matting.call_args.args[0])
    assert np.all(guided_image[raw_mask == 0] == 0)
    assert np.array_equal(
        guided_image[raw_mask > 0], np.asarray(rgb_image)[raw_mask > 0]
    )


def test_extract_object_layers_builds_rgba_crop(monkeypatch, rgb_image):
    image_np = np.asarray(rgb_image)
    alpha = np.zeros((6, 8), dtype=np.float64)
    alpha[1:5, 2:6] = 0.75
    kernel_size = (1, 1)

    inpaint = _install_inpainting_fake(
        monkeypatch, Image.new("RGB", rgb_image.size, (10, 10, 10))
    )
    inpaint_mask = np.zeros((6, 8), dtype=bool)
    inpaint_mask[1:5, 2:6] = True
    build_mask = Mock(return_value=inpaint_mask)
    monkeypatch.setattr(pipeline, "build_inpaint_mask", build_mask)
    monkeypatch.setattr(
        pipeline,
        "refine_background",
        Mock(side_effect=lambda background, *_args, **_kwargs: background),
    )
    foreground = np.full_like(image_np, 200)
    refine_alpha = Mock(return_value=(alpha, foreground))
    monkeypatch.setattr(pipeline, "refine_alpha_with_colors", refine_alpha)

    layers = pipeline._extract_object_layers(
        rgb_image, image_np, [alpha], ["component"], kernel_size
    )

    assert len(layers) == 1
    layer = layers[0]
    assert (layer.x, layer.y, layer.width, layer.height) == (2, 1, 4, 4)
    assert layer.keyword == "component"
    assert layer.png_base64
    build_mask.assert_called_once()
    inpaint.assert_called_once()
    assert inpaint.call_args.args[0] is rgb_image
    assert inpaint.call_args.args[1].mode == "L"
    refine_alpha.assert_called_once()


def test_generate_final_background_unions_masks(monkeypatch, rgb_image):
    first = np.zeros((6, 8), dtype=np.uint8)
    first[1:3, 1:3] = 255
    second = np.zeros((6, 8), dtype=np.uint8)
    second[3:5, 5:7] = 255

    inpainted = Image.new("RGB", rgb_image.size, (30, 40, 50))
    inpaint = _install_inpainting_fake(monkeypatch, inpainted)
    expand = Mock(side_effect=lambda mask, _kernel: mask)
    monkeypatch.setattr(pipeline, "expand_mask", expand)
    refine = Mock(side_effect=lambda background, *_args, **_kwargs: background)
    monkeypatch.setattr(pipeline, "refine_background", refine)

    result = pipeline._generate_final_background(
        rgb_image, [first, second], (1, 1)
    )

    assert result.size == rgb_image.size
    expand.assert_called_once()
    union = expand.call_args.args[0]
    assert np.array_equal(union, (first > 0) | (second > 0))
    inpaint.assert_called_once()
    passed_mask = np.asarray(inpaint.call_args.args[1])
    assert np.array_equal(passed_mask > 0, union)
    refine.assert_called_once()


def test_process_image_coordinates_all_pipeline_stages(monkeypatch, rgb_image):
    raw_mask = np.zeros((6, 8), dtype=np.uint8)
    raw_mask[1:5, 2:6] = 255
    alpha = raw_mask.astype(np.float64) / 255
    expected_layer = pipeline.ObjectLayer("component", "layer-data", 2, 1, 4, 4)
    expected_background = Image.new("RGB", rgb_image.size, (1, 2, 3))

    extract_masks = Mock(return_value=([raw_mask], ["component"]))
    refine_masks = Mock(return_value=[alpha])
    extract_layers = Mock(return_value=[expected_layer])
    generate_background = Mock(return_value=expected_background)
    encode = Mock(side_effect=lambda image, fmt="PNG": f"encoded-{image.size}")
    monkeypatch.setattr(pipeline, "_extract_raw_masks", extract_masks)
    monkeypatch.setattr(pipeline, "_refine_masks", refine_masks)
    monkeypatch.setattr(pipeline, "_extract_object_layers", extract_layers)
    monkeypatch.setattr(
        pipeline, "_generate_final_background", generate_background
    )
    monkeypatch.setattr(pipeline, "_image_to_base64", encode)

    result = pipeline.process_image(rgb_image, ["component"])

    assert result.original_width == 8
    assert result.original_height == 6
    assert result.background_base64 == "encoded-(8, 6)"
    assert result.layers == [expected_layer]
    extract_masks.assert_called_once()
    prepared_image, prepared_keywords = extract_masks.call_args.args
    assert prepared_image.mode == "RGB"
    assert prepared_image.size == rgb_image.size
    assert prepared_keywords == ["component"]
    refine_masks.assert_called_once()
    extract_layers.assert_called_once()
    generate_background.assert_called_once()


def test_process_image_returns_original_when_nothing_detected(
    monkeypatch, rgb_image
):
    monkeypatch.setattr(pipeline, "_extract_raw_masks", Mock(return_value=([], [])))
    refine_masks = Mock()
    monkeypatch.setattr(pipeline, "_refine_masks", refine_masks)
    monkeypatch.setattr(
        pipeline, "_image_to_base64", Mock(return_value="original-image")
    )

    result = pipeline.process_image(rgb_image, ["missing"])

    assert result.background_base64 == "original-image"
    assert result.layers == []
    refine_masks.assert_not_called()
