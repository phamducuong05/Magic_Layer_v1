"""Shared data contracts for image pipeline stages."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

from .roi import SquareROI


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


@dataclass
class DetectedObject:
    """One raw or grouped object in full-image coordinates."""

    object_id: str
    semantic_class: str
    display_label: str
    modal_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    segmentation_index: int = 0
    _original_modal_bbox: tuple[int, int, int, int] = field(init=False, repr=False)
    overlap_partner_ids: set[str] = field(default_factory=set)
    occluder_ids: set[str] = field(default_factory=set)
    amodal_mask: Optional[np.ndarray] = None
    completion_hole_mask: Optional[np.ndarray] = None
    completion_hole_area: Optional[int] = None
    effective_completion_hole_area: Optional[int] = None
    reconstruction_mask: Optional[np.ndarray] = None
    reconstruction_canvas: Optional[Image.Image] = None
    reconstruction_roi: Optional[SquareROI] = None
    soft_alpha: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Snapshot immutable raw geometry before later stages change ``bbox``."""
        self._original_modal_bbox = tuple(self.bbox)

    @property
    def original_modal_bbox(self) -> tuple[int, int, int, int]:
        """Return the tight bbox captured from the original raw modal mask."""
        return self._original_modal_bbox
