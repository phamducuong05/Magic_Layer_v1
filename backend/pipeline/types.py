"""Shared data contracts for image pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from PIL import Image

from .roi import SquareROI

if TYPE_CHECKING:
    from .diagnostics import PipelineDiagnostics


@dataclass
class ObjectLayer:
    keyword: str
    png_base64: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class ProcessResult:
    background_base64: str
    original_width: int
    original_height: int
    layers: List[ObjectLayer] = field(default_factory=list)
    diagnostics: Optional[PipelineDiagnostics] = None


@dataclass
class DetectedObject:
    """One raw segmentation object in full-image coordinates."""

    object_id: str
    semantic_class: str
    display_label: str
    modal_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    segmentation_index: int = 0
    _original_modal_bbox: tuple[int, int, int, int] = field(init=False, repr=False)
    overlap_partner_ids: set[str] = field(default_factory=set)
    occluder_ids: set[str] = field(default_factory=set)
    occluder_classes: set[str] = field(default_factory=set)
    amodal_mask: Optional[np.ndarray] = None
    completion_hole_mask: Optional[np.ndarray] = None
    completion_hole_area: Optional[int] = None
    effective_completion_hole_area: Optional[int] = None
    reconstruction_seed_mask: Optional[np.ndarray] = None
    reconstruction_mask: Optional[np.ndarray] = None
    reconstruction_generation_seed_mask: Optional[np.ndarray] = None
    reconstruction_generation_mask: Optional[np.ndarray] = None
    reconstruction_occluder_mask: Optional[np.ndarray] = None
    reconstruction_input_roi: Optional[SquareROI] = None
    reconstruction_target_bbox_mask: Optional[np.ndarray] = None
    reconstruction_foreign_modal_inside_bbox: Optional[np.ndarray] = None
    reconstruction_foreign_modal_outside_bbox: Optional[np.ndarray] = None
    reconstruction_protected_mask: Optional[np.ndarray] = None
    reconstruction_accepted_rgb_mask: Optional[np.ndarray] = None
    reconstruction_canvas: Optional[Image.Image] = None
    reconstruction_roi: Optional[SquareROI] = None
    reconstruction_evidence_alpha: Optional[np.ndarray] = None
    reconstruction_extension_mask: Optional[np.ndarray] = None
    reconstruction_write_mask: Optional[np.ndarray] = None
    reconstruction_support_mask: Optional[np.ndarray] = None
    reconstruction_failure_stage: Optional[str] = None
    reconstruction_failure_reason: Optional[str] = None
    completion_failure_stage: Optional[str] = None
    completion_failure_reason: Optional[str] = None
    soft_alpha: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Snapshot immutable raw geometry before later stages change ``bbox``."""
        self._original_modal_bbox = tuple(self.bbox)

    @property
    def original_modal_bbox(self) -> tuple[int, int, int, int]:
        """Return the tight bbox captured from the original raw modal mask."""
        return self._original_modal_bbox


@dataclass
class GroupedObject:
    """One final draggable group with ordered raw-member provenance."""

    group_id: str
    semantic_class: str
    display_label: str
    member_ids: tuple[str, ...]
    members: tuple[DetectedObject, ...]
    modal_mask: np.ndarray
    amodal_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    segmentation_index: int
    composed_source: Optional[Image.Image] = None
    composed_roi: Optional[SquareROI] = None
    matting_source: Optional[Image.Image] = None
    matting_roi: Optional[SquareROI] = None
    soft_alpha: Optional[np.ndarray] = None
    reconstruction_conflicts: tuple[tuple[str, str], ...] = ()

    @property
    def object_id(self) -> str:
        """Expose a common identity attribute for downstream diagnostics."""
        return self.group_id

    @property
    def has_reconstruction(self) -> bool:
        """Report whether any member retained accepted reconstructed RGB."""
        return any(
            member.reconstruction_canvas is not None for member in self.members
        )

    @property
    def effective_support_mask(self) -> np.ndarray:
        """Return support backed by reconstructed or visible RGB per member."""
        support = np.zeros_like(self.modal_mask, dtype=bool)
        for member in self.members:
            modal = member.modal_mask > 0
            if member.reconstruction_canvas is None:
                support |= modal
            elif member.reconstruction_support_mask is not None:
                support |= member.reconstruction_support_mask > 0
            elif member.amodal_mask is not None:
                support |= member.amodal_mask > 0
            else:
                support |= modal
        return support
