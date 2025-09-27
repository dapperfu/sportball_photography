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

# Configure logging early to suppress verbose output by default
import os
from loguru import logger

# Set default logging level to ERROR unless explicitly overridden
if not os.environ.get('SPORTBALL_VERBOSE'):
    logger.remove()
    logger.add(lambda msg: None, level="ERROR")

__version__ = "1.0.0"
__author__ = "Sportball Team"
__email__ = "team@sportball.ai"

# Lazy imports to avoid heavy dependencies at package import time
def _lazy_import_core():
    """Lazy import SportballCore to avoid heavy dependencies."""
    from .core import SportballCore
    return SportballCore

def _lazy_import_sidecar():
    """Lazy import SidecarManager to avoid heavy dependencies."""
    from .sidecar import SidecarManager
    return SidecarManager

def _lazy_import_decorators():
    """Lazy import decorators to avoid heavy dependencies."""
    from .decorators import (
        gpu_accelerated,
        parallel_processing,
        progress_tracked,
        cached_result
    )
    return gpu_accelerated, parallel_processing, progress_tracked, cached_result

# Create lazy properties for backward compatibility
class LazySportballCore:
    def __getattr__(self, name):
        return getattr(_lazy_import_core(), name)

class LazySidecarManager:
    def __getattr__(self, name):
        return getattr(_lazy_import_sidecar(), name)

class LazyDecorators:
    def __getattr__(self, name):
        decorators = _lazy_import_decorators()
        decorator_map = {
            'gpu_accelerated': decorators[0],
            'parallel_processing': decorators[1], 
            'progress_tracked': decorators[2],
            'cached_result': decorators[3]
        }
        return decorator_map[name]

# Export lazy objects
SportballCore = LazySportballCore()
SidecarManager = LazySidecarManager()
decorators = LazyDecorators()

# Don't access decorators immediately - keep them truly lazy
# gpu_accelerated = decorators.gpu_accelerated
# parallel_processing = decorators.parallel_processing
# progress_tracked = decorators.progress_tracked
# cached_result = decorators.cached_result

__all__ = [
    "SportballCore",
    "SidecarManager", 
    "decorators"
]
