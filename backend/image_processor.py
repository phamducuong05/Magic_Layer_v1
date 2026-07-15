"""Public image-processing API."""

from .pipeline.orchestrator import process_image, process_masks
from .pipeline.types import DetectedObject, ObjectLayer, ProcessResult

__all__ = [
    "DetectedObject",
    "ObjectLayer",
    "ProcessResult",
    "process_image",
    "process_masks",
]
