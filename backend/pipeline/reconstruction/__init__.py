"""Occlusion roles, reconstruction masks, and hidden-RGB reconstruction."""

from .mask_preparation import (
    apply_pair_decisions,
    build_reconstruction_masks,
    prepare_raw_reconstruction_masks,
)
from .reconstruction import reconstruct_objects
from .validate_reconstruction import ReconstructionValidationError

__all__ = [
    "ReconstructionValidationError",
    "apply_pair_decisions",
    "build_reconstruction_masks",
    "prepare_raw_reconstruction_masks",
    "reconstruct_objects",
]
