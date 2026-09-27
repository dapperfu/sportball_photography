"""
Sportball - sports photo organization.

EXIF-based game splitting.
Neural-net detection lives in other projects.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

# Configure logging early to suppress verbose output by default
import os
import logging
from loguru import logger

# Set default logging level to ERROR unless explicitly overridden
if not os.environ.get("SPORTBALL_VERBOSE"):
    # Suppress loguru
    logger.remove()
    logger.add(lambda msg: None, level="ERROR")

    # Suppress standard Python logging
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger().handlers = []

    # Suppress specific noisy loggers
    logging.getLogger("PIL").setLevel(logging.ERROR)

__version__ = "1.0.0"
__author__ = "Sportball Team"
__email__ = "team@sportball.ai"


# Lazy imports to avoid heavy dependencies at package import time
def _lazy_import_core():
    """Lazy import SportballCore to avoid heavy dependencies."""
    from .core import SportballCore

    return SportballCore


def _lazy_import_decorators():
    """Lazy import decorators to avoid heavy dependencies."""
    from .decorators import (
        gpu_accelerated,
        parallel_processing,
        progress_tracked,
        cached_result,
    )

    return gpu_accelerated, parallel_processing, progress_tracked, cached_result


# Create lazy properties for backward compatibility
class LazySportballCore:
    def __call__(self, *args, **kwargs):
        return _lazy_import_core()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_lazy_import_core(), name)


class LazyDecorators:
    def __getattr__(self, name):
        decorators = _lazy_import_decorators()
        decorator_map = {
            "gpu_accelerated": decorators[0],
            "parallel_processing": decorators[1],
            "progress_tracked": decorators[2],
            "cached_result": decorators[3],
        }
        return decorator_map[name]


# Export lazy objects
SportballCore = LazySportballCore()
decorators = LazyDecorators()

__all__ = ["SportballCore", "decorators"]

from . import _version

__version__ = _version.get_versions()["version"]
