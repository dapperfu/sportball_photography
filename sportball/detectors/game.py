"""
Game Detection Module

Automated game boundary detection based on photo timestamps with support
for manual splits and comprehensive game session management.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger


@dataclass
class GameDetectionConfig:
    """Configuration for game detection."""

    min_game_duration_minutes: int = 30
    min_gap_minutes: int = 10
    min_photos_per_game: int = 50


@dataclass
class GameSession:
    """Represents a detected game session."""

    game_id: int
    start_time: datetime
    end_time: datetime
    photo_count: int
    photo_files: List[Path]
    gap_before: Optional[int] = None  # seconds
    gap_after: Optional[int] = None  # seconds


class GameDetector:
    """
    Game boundary detection based on photo timestamps.

    This class provides automated game detection capabilities by analyzing
    photo timestamps to identify game boundaries, with support for manual
    splits and comprehensive game session management.
    """

    def __init__(
        self, config: Optional[GameDetectionConfig] = None, cache_enabled: bool = True
    ):
        """
        Initialize the GameDetector.

        Args:
            config: Optional configuration for game detection
            cache_enabled: Whether to enable result caching
        """
        self.config = config or GameDetectionConfig()
        self.cache_enabled = cache_enabled
        self.games: List[GameSession] = []
        self.logger = logger.bind(component="game_detector")
        self.logger.info("Initialized GameDetector")

    def detect_games(
        self,
        photo_directory: Path,
        pattern: str = "*_*",
        save_sidecar: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Detect game boundaries in a directory of photos.

        Args:
            photo_directory: Directory containing photos
            pattern: File pattern to match
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for game detection

        Returns:
            Dictionary containing game detection results
        """
        self.logger.info(f"Detecting games in {photo_directory} with pattern {pattern}")

        try:
            # Find photos matching the pattern
            photo_paths = list(photo_directory.rglob(pattern))
            photo_paths = [
                p
                for p in photo_paths
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            if not photo_paths:
                return {"error": "No photos found", "success": False}

            self.logger.info(f"Found {len(photo_paths)} photos")

            # Analyze timestamps
            photo_metadata = self._analyze_timestamps(photo_paths)

            if not photo_metadata:
                return {"error": "No valid timestamps found", "success": False}

            # Detect game boundaries
            boundaries = self._detect_game_boundaries(photo_metadata)

            if not boundaries:
                return {"error": "No games detected", "success": False}

            # Create game sessions
            games = self._create_game_sessions(photo_metadata, boundaries)

            # Store games for later use
            self.games = games

            # Format results
            results = {
                "success": True,
                "games": self._format_games_for_output(games),
                "summary": {
                    "total_games": len(games),
                    "total_photos": sum(game.photo_count for game in games),
                    "detected_dates": self._get_detected_dates(photo_metadata),
                },
            }

            # Save to sidecar if requested
            if save_sidecar:
                sidecar_path = photo_directory / "game_detection.json"
                with open(sidecar_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)

            return results

        except Exception as e:
            self.logger.error(f"Game detection failed: {e}")
            return {"error": str(e), "success": False}

    def _analyze_timestamps(self, photo_paths: List[Path]) -> List[Dict]:
        """Analyze timestamps from photo filenames."""
        photo_metadata = []

        for photo_path in photo_paths:
            timestamp = self.extract_timestamp_from_filename(photo_path.name)
            if timestamp:
                photo_metadata.append(
                    {
                        "path": photo_path,
                        "filename": photo_path.name,
                        "timestamp": timestamp,
                        "time_str": timestamp.strftime("%H%M%S"),
                    }
                )

        # Sort by timestamp
        photo_metadata.sort(key=lambda x: x["timestamp"])
        return photo_metadata

    def extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Extract timestamp from filename.

        Supports formats like:
        - 20250920_143022.jpg
        - 20250920_143022_001.jpg
        - IMG_20250920_143022.jpg

        Args:
            filename: The filename to parse

        Returns:
            Parsed datetime or None if parsing fails
        """
        try:
            # Remove extension
            name = Path(filename).stem

            # Try different patterns
            patterns = [
                "%Y%m%d_%H%M%S",  # 20250920_143022
                "%Y%m%d_%H%M%S_%f",  # 20250920_143022_001
                "IMG_%Y%m%d_%H%M%S",  # IMG_20250920_143022
            ]

            for pattern in patterns:
                try:
                    return datetime.strptime(name, pattern)
                except ValueError:
                    continue

            # Try to extract date and time parts separately
            if "_" in name:
                parts = name.split("_")
                if len(parts) >= 2:
                    date_part = parts[0]
                    time_part = parts[1]

                    if len(date_part) == 8 and len(time_part) >= 6:
                        try:
                            date_str = f"{date_part}_{time_part[:6]}"
                            return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        except ValueError:
                            pass

            return None

        except Exception as e:
            self.logger.debug(f"Failed to parse timestamp from {filename}: {e}")
            return None

    def _detect_game_boundaries(
        self, photo_metadata: List[Dict]
    ) -> List[Tuple[int, int]]:
        """Detect game boundaries based on timestamp gaps."""
        if len(photo_metadata) < self.config.min_photos_per_game:
            return []

        boundaries = []
        current_start = 0

        for i in range(1, len(photo_metadata)):
            prev_time = photo_metadata[i - 1]["timestamp"]
            curr_time = photo_metadata[i]["timestamp"]

            # Calculate gap in minutes
            gap_minutes = (curr_time - prev_time).total_seconds() / 60

            # Use adaptive gap threshold based on context
            adaptive_gap_threshold = self._calculate_adaptive_gap_threshold(
                photo_metadata, current_start, i, gap_minutes
            )

            # If gap is large enough, end current game and start new one
            if gap_minutes >= adaptive_gap_threshold:
                # Check if current game meets minimum duration
                game_duration_minutes = (
                    prev_time - photo_metadata[current_start]["timestamp"]
                ).total_seconds() / 60
                photo_count = i - current_start

                if (
                    game_duration_minutes >= self.config.min_game_duration_minutes
                    and photo_count >= self.config.min_photos_per_game
                ):
                    boundaries.append((current_start, i - 1))

                current_start = i

        # Add the last game if it meets criteria
        if current_start < len(photo_metadata) - 1:
            last_time = photo_metadata[-1]["timestamp"]
            first_time = photo_metadata[current_start]["timestamp"]
            game_duration_minutes = (last_time - first_time).total_seconds() / 60
            photo_count = len(photo_metadata) - current_start

            if (
                game_duration_minutes >= self.config.min_game_duration_minutes
                and photo_count >= self.config.min_photos_per_game
            ):
                boundaries.append((current_start, len(photo_metadata) - 1))

        return boundaries

    def _calculate_adaptive_gap_threshold(
        self, photo_metadata: List[Dict], current_start: int, current_index: int, gap_minutes: float
    ) -> float:
        """
        Calculate an adaptive gap threshold based on context.
        
        This helps distinguish between:
        - Small breaks within a game (halftime, timeouts) - should NOT split
        - Actual game boundaries - should split
        """
        # Base threshold from config
        base_threshold = self.config.min_gap_minutes
        
        # Calculate current segment stats
        current_duration = (
            photo_metadata[current_index - 1]["timestamp"] - 
            photo_metadata[current_start]["timestamp"]
        ).total_seconds() / 60
        photo_count = current_index - current_start
        
        # If this would create a very short game segment, be more lenient
        if photo_count < self.config.min_photos_per_game:
            # Increase threshold to avoid splitting short segments
            return max(base_threshold * 2, 30)  # At least 30 minutes
        
        # If the current potential game is already long enough, be more strict
        if current_duration >= self.config.min_game_duration_minutes:
            # Game is already long enough, use normal threshold
            return base_threshold
        
        # If we're in the middle of what could be a game, be more lenient
        # Look ahead to see if there's a much larger gap coming
        look_ahead_distance = min(100, len(photo_metadata) - current_index)  # Increased look-ahead
        max_future_gap = 0
        
        for j in range(current_index + 1, min(current_index + look_ahead_distance, len(photo_metadata))):
            if j < len(photo_metadata):
                future_gap = (
                    photo_metadata[j]["timestamp"] - photo_metadata[j-1]["timestamp"]
                ).total_seconds() / 60
                max_future_gap = max(max_future_gap, future_gap)
        
        # If there's a much larger gap coming, this might be a small break within a game
        if max_future_gap > gap_minutes * 2 and gap_minutes < 30:  # Reduced multiplier
            self.logger.debug(f"Adaptive threshold: Found larger gap {max_future_gap:.1f} min ahead, increasing threshold from {base_threshold} to {max(base_threshold * 1.5, 25)}")
            return max(base_threshold * 1.5, 25)  # Be more lenient
        
        # Special case: if we're early in the sequence and there's a big gap coming,
        # be more lenient with small gaps
        if current_start < 200 and max_future_gap > 50:  # More lenient conditions
            self.logger.debug(f"Adaptive threshold: Early sequence with big gap ahead {max_future_gap:.1f} min, increasing threshold from {base_threshold} to {max(base_threshold * 2, 25)}")
            return max(base_threshold * 2, 25)  # Be more lenient
        
        return base_threshold

    def _merge_split_games(
        self, photo_metadata: List[Dict], boundaries: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Merge games that were split by small gaps but should be considered one game.
        
        This handles cases where there are small breaks within a game (like halftime)
        that shouldn't split the game into separate sessions.
        """
        if len(boundaries) <= 1:
            return boundaries

        merged_boundaries = []
        i = 0
        
        while i < len(boundaries):
            current_start, current_end = boundaries[i]
            
            # Check if we should merge with the next game
            if i + 1 < len(boundaries):
                next_start, next_end = boundaries[i + 1]
                
                # Calculate the gap between current and next game
                current_end_time = photo_metadata[current_end]["timestamp"]
                next_start_time = photo_metadata[next_start]["timestamp"]
                gap_minutes = (next_start_time - current_end_time).total_seconds() / 60
                
                # Calculate total duration if merged
                total_start_time = photo_metadata[current_start]["timestamp"]
                total_end_time = photo_metadata[next_end]["timestamp"]
                total_duration_minutes = (total_end_time - total_start_time).total_seconds() / 60
                total_photos = next_end - current_start + 1
                
                # Merge if:
                # 1. Gap is relatively small (less than 30 minutes)
                # 2. Total duration would be reasonable (less than 4 hours)
                # 3. Both games individually meet minimum requirements
                current_duration = (current_end_time - total_start_time).total_seconds() / 60
                next_duration = (total_end_time - next_start_time).total_seconds() / 60
                
                should_merge = (
                    gap_minutes < 30 and  # Small gap
                    total_duration_minutes < 240 and  # Less than 4 hours total
                    current_duration >= self.config.min_game_duration_minutes and
                    next_duration >= self.config.min_game_duration_minutes and
                    total_photos >= self.config.min_photos_per_game
                )
                
                if should_merge:
                    # Merge the games
                    merged_boundaries.append((current_start, next_end))
                    i += 2  # Skip the next game since we merged it
                    continue
            
            # Don't merge, keep current game
            merged_boundaries.append((current_start, current_end))
            i += 1
        
        return merged_boundaries

    def _create_game_sessions(
        self, photo_metadata: List[Dict], boundaries: List[Tuple[int, int]]
    ) -> List[GameSession]:
        """Create GameSession objects from boundaries."""
        games = []

        for i, (start_idx, end_idx) in enumerate(boundaries):
            game_photos = [
                meta["path"] for meta in photo_metadata[start_idx : end_idx + 1]
            ]
            start_time = photo_metadata[start_idx]["timestamp"]
            end_time = photo_metadata[end_idx]["timestamp"]

            game = GameSession(
                game_id=i + 1,
                start_time=start_time,
                end_time=end_time,
                photo_count=len(game_photos),
                photo_files=game_photos,
            )
            games.append(game)

        # Calculate gaps between games
        for i, game in enumerate(games):
            if i > 0:
                prev_game = games[i - 1]
                game.gap_before = int(
                    (game.start_time - prev_game.end_time).total_seconds()
                )

            if i < len(games) - 1:
                next_game = games[i + 1]
                game.gap_after = int(
                    (next_game.start_time - game.end_time).total_seconds()
                )

        return games

    def _format_games_for_output(self, games: List[GameSession]) -> List[Dict]:
        """Format games for output."""
        formatted_games = []

        for game in games:
            duration_minutes = (game.end_time - game.start_time).total_seconds() / 60

            formatted_game = {
                "game_id": game.game_id,
                "start_time": game.start_time.isoformat(),
                "end_time": game.end_time.isoformat(),
                "start_time_formatted": game.start_time.strftime("%H:%M:%S"),
                "end_time_formatted": game.end_time.strftime("%H:%M:%S"),
                "duration_minutes": round(duration_minutes, 1),
                "photo_count": game.photo_count,
                "gap_before_minutes": round(game.gap_before / 60, 1)
                if game.gap_before
                else None,
                "gap_after_minutes": round(game.gap_after / 60, 1)
                if game.gap_after
                else None,
                "photo_files": [str(photo_path) for photo_path in game.photo_files],
            }

            formatted_games.append(formatted_game)

        return formatted_games

    def _get_detected_dates(self, photo_metadata: List[Dict]) -> List[str]:
        """Get list of unique dates detected in the photos."""
        if not photo_metadata:
            return []

        dates = set()
        for meta in photo_metadata:
            dates.add(meta["timestamp"].strftime("%Y-%m-%d"))

        return sorted(list(dates))

    def load_split_file(self, split_file_path: Path) -> List[datetime]:
        """
        Load manual splits from a text file.

        Args:
            split_file_path: Path to the split file

        Returns:
            List of parsed timestamps
        """
        manual_splits = []

        try:
            with open(split_file_path, "r") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                try:
                    # Parse timestamp - handle both formats
                    if " " in line:
                        # Format: "YYYY-MM-DD HH:MM:SS"
                        timestamp = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
                    elif ":" in line:
                        # Format: "HH:MM:SS" - assume current detected date
                        time_part = datetime.strptime(line, "%H:%M:%S")
                        # Use the first detected date from the games
                        if self.games:
                            first_game_date = self.games[0].start_time.date()
                            timestamp = datetime.combine(
                                first_game_date, time_part.time()
                            )
                        else:
                            # Fallback to September 20th, 2025
                            timestamp = datetime(
                                2025,
                                9,
                                20,
                                time_part.hour,
                                time_part.minute,
                                time_part.second,
                            )
                    else:
                        # Format: "HHMMSS" - assume current detected date
                        time_part = datetime.strptime(line, "%H%M%S")
                        if self.games:
                            first_game_date = self.games[0].start_time.date()
                            timestamp = datetime.combine(
                                first_game_date, time_part.time()
                            )
                        else:
                            # Fallback to September 20th, 2025
                            timestamp = datetime(
                                2025,
                                9,
                                20,
                                time_part.hour,
                                time_part.minute,
                                time_part.second,
                            )

                    manual_splits.append(timestamp)

                except ValueError as e:
                    self.logger.warning(
                        f"Invalid timestamp format on line {line_num}: '{line}' - {e}"
                    )
                    continue

            # Sort splits
            manual_splits.sort()

            self.logger.info(
                f"Loaded {len(manual_splits)} manual splits from {split_file_path}"
            )
            return manual_splits

        except FileNotFoundError:
            self.logger.error(f"Split file not found: {split_file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading split file: {e}")
            return []

    def apply_manual_splits(self, manual_splits: List[datetime]) -> List[GameSession]:
        """
        Apply manual splits to the detected games.

        Args:
            manual_splits: List of manual split timestamps

        Returns:
            List of GameSession objects with manual splits applied
        """
        if not manual_splits or not self.games:
            return self.games

        self.logger.info(f"Applying {len(manual_splits)} manual splits")

        final_games = []

        for game in self.games:
            # Find manual splits that fall within this game's time range
            game_splits = [
                split
                for split in manual_splits
                if game.start_time <= split <= game.end_time
            ]

            if not game_splits:
                # No splits within this game, keep it as is
                final_games.append(game)
                continue

            # Sort splits and add start/end times
            game_splits.sort()
            split_points = [game.start_time] + game_splits + [game.end_time]

            # Create sub-games for each segment
            for i in range(len(split_points) - 1):
                segment_start = split_points[i]
                segment_end = split_points[i + 1]

                # Find photos in this time segment
                segment_photos = [
                    photo
                    for photo in game.photo_files
                    if segment_start
                    <= self.extract_timestamp_from_filename(photo.name)
                    <= segment_end
                ]

                if len(segment_photos) >= self.config.min_photos_per_game:
                    # Create new game session for this segment
                    segment_game = GameSession(
                        game_id=len(final_games) + 1,
                        start_time=segment_start,
                        end_time=segment_end,
                        photo_count=len(segment_photos),
                        photo_files=segment_photos,
                    )
                    final_games.append(segment_game)

        # Reassign game IDs and calculate gaps
        for i, game in enumerate(final_games, 1):
            game.game_id = i

        # Calculate gaps between games
        for i, game in enumerate(final_games):
            if i > 0:
                prev_game = final_games[i - 1]
                game.gap_before = int(
                    (game.start_time - prev_game.end_time).total_seconds()
                )

            if i < len(final_games) - 1:
                next_game = final_games[i + 1]
                game.gap_after = int(
                    (next_game.start_time - game.end_time).total_seconds()
                )

        self.logger.info(f"Applied manual splits. Final games: {len(final_games)}")
        return final_games
