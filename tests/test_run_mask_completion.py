"""Tests for the real-model mask-completion command-line output helper."""

import numpy as np
from PIL import Image

from run_mask_completion import save_masks


def test_save_masks_writes_binary_full_size_pngs(tmp_path):
    first = np.zeros((3, 4), dtype=bool)
    first[1, 2] = True
    second = np.zeros((3, 4), dtype=np.uint8)
    second[0, 0] = 255

    paths = save_masks([first, second], tmp_path / "masks")

    assert [path.name for path in paths] == ["mask_0.png", "mask_1.png"]
    saved_first = np.asarray(Image.open(paths[0]))
    saved_second = np.asarray(Image.open(paths[1]))
    assert saved_first.shape == (3, 4)
    assert set(np.unique(saved_first)) == {0, 255}
    assert set(np.unique(saved_second)) == {0, 255}
