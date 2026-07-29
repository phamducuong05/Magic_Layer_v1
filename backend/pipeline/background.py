"""Final background removal with a background-only inpainter."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from ..core.layerd_refine import expand_mask, refine_background
from ..core.logging import get_logger, log_event
from .layers import BG_REFINE_NUM_COLORS, BG_REFINE_OUTER_RATIO
from .matting import THRESHOLD_ALPHA
from .types import DetectedObject, GroupedObject


VISIBLE_ALPHA_DILATION = (3, 3)
logger = get_logger(__name__)


def _background_artifact_callback(
    diagnostics_directory: str | Path,
) -> Callable[[str, Image.Image], None]:
    """Create a stage callback that writes ordered background diagnostics."""
    directory = Path(diagnostics_directory)
    directory.mkdir(parents=True, exist_ok=True)
    filenames = {
        "after_lama": "01_after_lama.png",
        "after_smarteraser": "01_after_smarteraser.png",
        "after_model": "01_after_model.png",
        "after_composition_blend": "02_after_composition_blend.png",
    }

    def save_artifact(stage: str, image: Image.Image) -> None:
        filename = filenames.get(stage)
        if filename is not None:
            image.convert("RGB").save(directory / filename)

    return save_artifact


def _visible_soft_alpha(
    modal_mask: np.ndarray,
    soft_alpha: np.ndarray,
) -> np.ndarray:
    """Keep alpha coverage only on, or immediately beside, visible pixels."""
    visible_support = expand_mask(
        modal_mask > 0, VISIBLE_ALPHA_DILATION
    ).astype(bool)
    return np.where(visible_support, soft_alpha, 0.0)


def generate_background_from_masks(
    image: Image.Image,
    raw_masks: Sequence[np.ndarray],
    soft_alphas: Sequence[np.ndarray],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
    diagnostics_directory: str | Path | None = None,
) -> Image.Image:
    """Inpaint all components using hard masks and soft-alpha coverage."""
    union_mask = np.logical_or.reduce([mask > 0 for mask in raw_masks])
    for alpha in soft_alphas:
        union_mask |= alpha > THRESHOLD_ALPHA
    union_mask = binary_fill_holes(union_mask)
    union_mask = expand_mask(union_mask, kernel_size).astype(bool)
    log_event(
        logger,
        "background_inpainting",
        "mask_prepared",
        raw_mask_count=len(raw_masks),
        soft_alpha_count=len(soft_alphas),
        inpaint_pixels=int(np.count_nonzero(union_mask)),
        kernel_size=kernel_size,
    )
    final_mask = Image.fromarray(union_mask.astype(np.uint8) * 255, mode="L")
    if diagnostics_directory is not None:
        directory = Path(diagnostics_directory)
        directory.mkdir(parents=True, exist_ok=True)
        final_mask.save(directory / "00_input_mask.png")

    artifact_callback = (
        _background_artifact_callback(diagnostics_directory)
        if diagnostics_directory is not None
        else None
    )
    if artifact_callback is None:
        background = background_inpaint(image, final_mask)
    else:
        background = background_inpaint(
            image,
            final_mask,
            artifact_callback=artifact_callback,
        )
    log_event(
        logger,
        "background_inpainting",
        "model_result",
        decision="accepted",
        output_size=background.size,
    )
    if background.size != image.size:
        background = background.resize(image.size, Image.Resampling.LANCZOS)

    background_np = np.asarray(background.convert("RGB"), dtype=np.uint8)
    background_np = refine_background(
        background_np,
        union_mask,
        n_outer_ratio=BG_REFINE_OUTER_RATIO,
        max_num_colors=BG_REFINE_NUM_COLORS,
    )
    if diagnostics_directory is not None:
        Image.fromarray(background_np, mode="RGB").save(
            Path(diagnostics_directory) / "03_after_palette_refine.png"
        )
    return Image.fromarray(background_np, mode="RGB")


def generate_final_background(
    image: Image.Image,
    objects: Sequence[DetectedObject | GroupedObject],
    kernel_size: tuple[int, int],
    background_inpaint: Callable[[Image.Image, Image.Image], Image.Image],
    diagnostics_directory: str | Path | None = None,
) -> Image.Image:
    """Inpaint visible object coverage once on the original source image."""
    log_event(
        logger,
        "background_inpainting",
        "decision",
        decision="remove_visible_modal_coverage",
        object_count=len(objects),
        reason="hidden_amodal_rgb_must_not_affect_final_background",
    )
    return generate_background_from_masks(
        image,
        [detected.modal_mask for detected in objects],
        [
            _visible_soft_alpha(
                detected.modal_mask,
                detected.soft_alpha,
            )
            for detected in objects
            if detected.soft_alpha is not None
        ],
        kernel_size,
        background_inpaint,
        diagnostics_directory=diagnostics_directory,
    )
