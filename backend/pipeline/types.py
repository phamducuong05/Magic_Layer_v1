"""Shared data contracts for image pipeline stages."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


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
    """One same-class grouped object in full-image coordinates."""

    object_id: str
    semantic_class: str
    display_label: str
    modal_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    overlap_partner_ids: set[str] = field(default_factory=set)
    occluder_ids: set[str] = field(default_factory=set)
    amodal_mask: Optional[np.ndarray] = None
    completion_hole_mask: Optional[np.ndarray] = None
    completion_hole_area: Optional[int] = None
    reconstruction_mask: Optional[np.ndarray] = None
    soft_alpha: Optional[np.ndarray] = None
