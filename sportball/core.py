"""
Sportball Core Module

Core entry point for EXIF game splitting.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger

from .decorators import timing_decorator


class SportballCore:
    """
    Unified access to sportball game organization.

    This package splits a photo dump into game folders using EXIF
    capture times.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        enable_gpu: bool = False,
        max_workers: Optional[int] = None,
        cache_enabled: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SportballCore.

        Parameters
        ----------
        base_dir : Path, optional
            Base directory for operations.
        enable_gpu : bool
            Unused. Kept so existing CLI flags still construct a core.
        max_workers : int, optional
            Maximum number of parallel workers.
        cache_enabled : bool
            Unused. Kept so existing CLI flags still construct a core.
        verbose : bool
            Whether to show verbose output.
        """
        self.base_dir = base_dir or Path.cwd()
        self.enable_gpu = enable_gpu
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self.verbose = verbose

        self._game_detector: Any = None

        self.logger = logger.bind(component="core")
        self.logger.info("Initialized SportballCore")

    @property
    def game_detector(self) -> Any:
        """Lazy-loaded EXIF game splitter."""
        if self._game_detector is None:
            from .detectors.game import GameDetector

            self._game_detector = GameDetector(cache_enabled=self.cache_enabled)
        return self._game_detector

    @timing_decorator
    def detect_games(
        self,
        photo_directory: Union[Path, Sequence[Path], List[Path]],
        pattern: str = "*",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Detect game boundaries across one or more directories of photos.

        Capture times are read from each image's EXIF data via
        fast-exif-rs-py. Results are returned in memory only. Games are
        numbered in capture-time order across every input directory.

        Parameters
        ----------
        photo_directory : Path or sequence of Path
            One dump folder or several (month folders, a season, etc.).
        pattern : str
            File pattern to match.
        **kwargs
            Additional arguments for game detection.

        Returns
        -------
        dict
            Game detection results.
        """
        self.logger.info(f"Detecting games in {photo_directory} with pattern {pattern}")

        try:
            return self.game_detector.detect_games(
                photo_directory, pattern=pattern, **kwargs
            )
        except Exception as e:
            self.logger.error(f"Game detection failed: {e}")
            return {"error": str(e), "success": False}

    @timing_decorator
    def analyze_games(
        self,
        photo_directory: Union[Path, Sequence[Path], List[Path]],
        pattern: str = "*",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Explain how photos would be split without creating folders.

        Parameters
        ----------
        photo_directory : Path or sequence of Path
            One dump folder or several.
        pattern : str
            File pattern to match.
        **kwargs
            Detection knobs plus optional ``bin_minutes``.

        Returns
        -------
        dict
            Timeline, histogram, games, and unsorted clusters.
        """
        self.logger.info(f"Analyzing games in {photo_directory} with pattern {pattern}")
        try:
            return self.game_detector.analyze_games(
                photo_directory, pattern=pattern, **kwargs
            )
        except Exception as e:
            self.logger.error(f"Game analysis failed: {e}")
            return {"error": str(e), "success": False}

    def cleanup_cache(self) -> None:
        """No-op. Sportball does not persist analysis beside images."""
        self.logger.info("No persist cache to clear")
