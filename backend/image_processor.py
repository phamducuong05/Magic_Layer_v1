"""Public image-processing API."""

from .pipeline.orchestrator import process_image, process_masks
from .pipeline.types import (
    DetectedObject,
    GroupedObject,
    ObjectLayer,
    ProcessResult,
)

__all__ = [
    "DetectedObject",
    "GroupedObject",
    "ObjectLayer",
    "ProcessResult",
    "process_image",
    "process_masks",
]
