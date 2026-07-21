"""Function-based stages for the image processing pipeline."""

from .diagnostics import PipelineDiagnostics
from .types import DetectedObject, GroupedObject, ObjectLayer, ProcessResult

__all__ = [
    "DetectedObject",
    "GroupedObject",
    "ObjectLayer",
    "PipelineDiagnostics",
    "ProcessResult",
]
