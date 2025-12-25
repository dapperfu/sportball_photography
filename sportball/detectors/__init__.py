"""
Sportball Detectors Module

Detection modules for faces, objects, games, balls, and quality assessment.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import os
from loguru import logger

def _setup_cuda_library_path():
    """Automatically set up CUDA library path for ONNX Runtime compatibility.
    
    This function detects installed CUDA libraries and adds them to LD_LIBRARY_PATH
    so that ONNX Runtime can find the CUDA execution provider libraries.
    """
    try:
        import glob
        
        # Check if CUDA libraries are already in the library path
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        
        # Common CUDA installation paths to check
        cuda_paths = [
            "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
            "/usr/local/cuda-13/targets/x86_64-linux/lib",
            "/usr/local/cuda-12.0/targets/x86_64-linux/lib",
            "/usr/local/cuda-12/targets/x86_64-linux/lib",
            "/usr/local/cuda-11.0/targets/x86_64-linux/lib",
            "/usr/local/cuda-11/targets/x86_64-linux/lib",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
        ]
        
        # Try to find a valid CUDA installation
        for cuda_path in cuda_paths:
            lib_path = os.path.join(cuda_path, "libcublasLt.so")
            if os.path.exists(lib_path) or glob.glob(f"{cuda_path}/libcublasLt.so.*"):
                if cuda_path not in current_ld_path:
                    # Add to library path
                    new_path = cuda_path
                    if current_ld_path:
                        new_path = f"{cuda_path}:{current_ld_path}"
                    os.environ["LD_LIBRARY_PATH"] = new_path
                    logger.debug(f"Auto-configured CUDA library path: {cuda_path}")
                    break
    except Exception as e:
        logger.debug(f"Could not auto-configure CUDA library path: {e}")

# Set up CUDA library path on module import
_setup_cuda_library_path()

from .face import FaceDetector

# Import other detectors as they become available
try:
    from .object import ObjectDetector
except ImportError:
    ObjectDetector = None

try:
    from .game import GameDetector
except ImportError:
    GameDetector = None

try:
    from .ball import BallDetector
except ImportError:
    BallDetector = None

try:
    from .quality import QualityAssessor
except ImportError:
    QualityAssessor = None

__all__ = [
    "FaceDetector",
    "ObjectDetector",
    "GameDetector",
    "BallDetector",
    "QualityAssessor",
]
