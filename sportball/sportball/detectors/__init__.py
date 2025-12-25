"""
Sportball Detectors Module

Detection modules for faces, objects, games, balls, quality assessment,
colors, actions, and field analysis.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

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

try:
    from .color import ColorDetector
except ImportError:
    ColorDetector = None

try:
    from .action import ActionClassifier
except ImportError:
    ActionClassifier = None

try:
    from .field import FieldAnalyzer
except ImportError:
    FieldAnalyzer = None

__all__ = [
    "FaceDetector",
    "ObjectDetector", 
    "GameDetector",
    "BallDetector",
    "QualityAssessor",
    "ColorDetector",
    "ActionClassifier",
    "FieldAnalyzer",
]
