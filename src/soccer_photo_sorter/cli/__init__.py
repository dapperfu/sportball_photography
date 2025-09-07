"""CLI modules for soccer photo sorter."""

from .main import main, cli
from .color_sorter import color_sorter
from .number_sorter import number_sorter
from .face_sorter import face_sorter

__all__ = [
    "main",
    "cli",
    "color_sorter",
    "number_sorter", 
    "face_sorter",
]
