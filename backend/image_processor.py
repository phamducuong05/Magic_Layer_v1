"""Public image-processing API."""

from .pipeline.diagnostics import PipelineDiagnostics
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
    "PipelineDiagnostics",
    "ProcessResult",
    "process_image",
    "process_masks",
]
