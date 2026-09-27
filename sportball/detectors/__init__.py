"""
Sportball detectors.

EXIF game splitting. Neural-net detection lives in other projects.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from .animate import animate_games
from .game import GameDetector
from .pano import find_action_panos

__all__ = [
    "GameDetector",
    "animate_games",
    "find_action_panos",
]
