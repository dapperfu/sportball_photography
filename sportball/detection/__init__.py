"""
Detection Module

Tool-agnostic detection framework for sportball.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from .base import DetectionTool, DetectionResult, DetectionConfig
from .registry import DetectionRegistry
from .parallel_validator import ParallelJSONValidator
from .rust_sidecar import RustSidecarManager, RustSidecarConfig

__all__ = [
    "DetectionTool",
    "DetectionResult",
    "DetectionConfig",
    "DetectionRegistry",
    "ParallelJSONValidator",
    "RustSidecarManager",
    "RustSidecarConfig",
]
