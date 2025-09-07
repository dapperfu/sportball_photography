"""
Soccer Photo Sorter - AI-powered photo organization system.

This package provides tools for automatically sorting soccer game photographs
based on jersey colors, jersey numbers, and player faces using computer vision
and machine learning techniques.
"""

__version__ = "0.1.0"
__author__ = "Soccer Photo Sorter Team"
__email__ = "team@soccerphotosorter.com"

from .core.image_processor import ImageProcessor
from .core.photo_sorter import PhotoSorter
from .detectors.color_detector import ColorDetector
from .detectors.number_detector import NumberDetector
from .detectors.face_detector import FaceDetector
from .utils.cuda_utils import CudaManager
from .config.settings import Settings

__all__ = [
    "ImageProcessor",
    "PhotoSorter", 
    "ColorDetector",
    "NumberDetector",
    "FaceDetector",
    "CudaManager",
    "Settings",
]
