import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "backend"
    / "models"
    / "object_reconstruction"
    / "prepare_hd_painter_data.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_hd_painter_data", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_multiple_masks_produce_aligned_square_smoke_data(tmp_path):
    module = load_module()
    image_array = np.zeros((60, 100, 3), dtype=np.uint8)
    image_array[..., 0] = np.arange(100, dtype=np.uint8)
    image_path = tmp_path / "scene.png"
    Image.fromarray(image_array, mode="RGB").save(image_path)

    first = np.zeros((60, 100), dtype=np.uint8)
    first[20:40, 30:42] = 1
    second = np.zeros((60, 100), dtype=np.uint8)
    second[24:36, 46:58] = 180
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.fromarray(first, mode="L").save(first_path)
    Image.fromarray(second, mode="L").save(second_path)

    output_dir = tmp_path / "prepared"
    result = module.prepare_hd_painter_data(
        image_path=image_path,
        mask_paths=[first_path, second_path],
        output_dir=output_dir,
        padding_ratio=0.25,
    )

    crop = Image.open(result.image_crop)
    combined = Image.open(result.combined_mask)
    component_masks = [Image.open(path) for path in result.component_masks]
    assert crop.width == crop.height
    assert combined.size == crop.size
    assert all(mask.size == crop.size for mask in component_masks)
    assert set(np.unique(np.asarray(combined))) == {0, 255}
    assert set(np.unique(np.asarray(component_masks[0]))) == {0, 255}
    assert np.array_equal(
        np.asarray(combined) > 0,
        np.logical_or.reduce([np.asarray(mask) > 0 for mask in component_masks]),
    )

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["source_size"] == [100, 60]
    assert metadata["square_size"] == crop.width
    assert metadata["mask_semantics"] == "white pixels are reconstructed"
    assert Path(result.preview).exists()


def test_object_at_image_edge_is_zero_padded_in_mask(tmp_path):
    module = load_module()
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (80, 40), "red").save(image_path)
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[0:8, 0:5] = 255
    mask_path = tmp_path / "edge-mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    result = module.prepare_hd_painter_data(
        image_path=image_path,
        mask_paths=[mask_path],
        output_dir=tmp_path / "prepared",
        padding_ratio=1.0,
    )

    crop = np.asarray(Image.open(result.image_crop))
    prepared_mask = np.asarray(Image.open(result.combined_mask))
    assert crop.shape[0] == crop.shape[1]
    assert np.any(prepared_mask == 255)
    assert prepared_mask[0, 0] == 0
    assert np.all(crop[0, 0] == crop[0, 1])


@pytest.mark.parametrize("case", ["empty", "wrong_size"])
def test_invalid_masks_are_rejected(tmp_path, case):
    module = load_module()
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (40, 30), "white").save(image_path)
    size = (40, 30) if case == "empty" else (20, 15)
    mask_path = tmp_path / "mask.png"
    Image.new("L", size, 0 if case == "empty" else 255).save(mask_path)

    with pytest.raises(ValueError, match="empty|dimensions"):
        module.prepare_hd_painter_data(
            image_path=image_path,
            mask_paths=[mask_path],
            output_dir=tmp_path / "prepared",
        )
