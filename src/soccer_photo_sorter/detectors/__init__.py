"""Detection modules for soccer photo sorter."""

from .color_detector import ColorDetector
from .number_detector import NumberDetector
from .face_detector import FaceDetector

__all__ = [
    "ColorDetector",
    "NumberDetector",
    "FaceDetector",
]
