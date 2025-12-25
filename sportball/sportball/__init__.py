"""
Sportball - Unified Sports Photo Analysis Package

A comprehensive Python package for analyzing and organizing sports photographs
using computer vision, machine learning, and AI techniques.

Features:
- Face detection and recognition
- Object detection and extraction  
- Game boundary detection and splitting
- Jersey color and number detection
- Photo quality assessment
- Parallel processing with GPU support
- Comprehensive CLI interface

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

__version__ = "1.0.0"
__author__ = "Sportball Team"
__email__ = "team@sportball.ai"

from .core import SportballCore  # SportballCore is in core.py, not core/__init__.py
from .sidecar import SidecarManager
from .decorators import (
    gpu_accelerated,
    parallel_processing,
    progress_tracked,
    cached_result
)

__all__ = [
    "SportballCore",
    "SidecarManager", 
    "gpu_accelerated",
    "parallel_processing",
    "progress_tracked",
    "cached_result"
]
