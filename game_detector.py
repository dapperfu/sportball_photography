#!/usr/bin/env python3
"""
Game Detection System for Soccer Photo Sorting

This module provides functionality to automatically detect game boundaries
in a collection of soccer photos and organize them into separate game folders.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import os
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GameSession:
    """Represents a single game session with its time boundaries and photos."""
    
    game_id: int
    start_time: datetime
    end_time: datetime
    photo_count: int
    photo_files: List[Path]
    gap_before: Optional[int] = None  # Gap in seconds before this game
    gap_after: Optional[int] = None   # Gap in seconds after this game


@dataclass
class GameDetectionConfig:
    """Configuration for game detection algorithm."""
    
    min_game_duration_minutes: int = 30  # Minimum game duration
    max_gap_minutes: int = 5             # Maximum gap within a game
    min_gap_minutes: int = 10            # Minimum gap to separate games
    min_photos_per_game: int = 50        # Minimum photos per game
    time_format: str = "%H%M%S"         # Time format in filename


class GameDetector:
    """
    Detects game boundaries in soccer photos based on temporal analysis.
    
    This class analyzes photo timestamps to identify natural breaks that
    indicate separate games, then organizes photos into game-specific folders.
    """
    
    def __init__(self, config: Optional[GameDetectionConfig] = None):
        """
        Initialize the game detector.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or GameDetectionConfig()
        self.games: List[GameSession] = []
        self.photo_metadata: List[Dict] = []
        
    def extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Extract timestamp from filename.
        
        Args:
            filename: Image filename (e.g., "20250920_080527.960.jpg")
            
        Returns:
            Datetime object or None if parsing fails
        """
        try:
            # Pattern: YYYYMMDD_HHMMSS.mmm
            pattern = r'(\d{8})_(\d{6})\.(\d{3})'
            match = re.match(pattern, filename)
            
            if match:
                date_str, time_str, _ = match.groups()
                datetime_str = f"{date_str}_{time_str}"
                return datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing timestamp from {filename}: {e}")
            return None
    
    def analyze_photo_timestamps(self, photo_paths: List[Path]) -> List[Dict]:
        """
        Analyze photo timestamps and extract metadata.
        
        Args:
            photo_paths: List of photo file paths
            
        Returns:
            List of photo metadata dictionaries
        """
        logger.info(f"Analyzing timestamps for {len(photo_paths)} photos")
        
        photo_metadata = []
        
        for photo_path in photo_paths:
            timestamp = self.extract_timestamp_from_filename(photo_path.name)
            if timestamp:
                photo_metadata.append({
                    'path': photo_path,
                    'filename': photo_path.name,
                    'timestamp': timestamp,
                    'time_str': timestamp.strftime("%H%M%S")
                })
            else:
                logger.warning(f"Could not parse timestamp from {photo_path.name}")
        
        # Sort by timestamp
        photo_metadata.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Successfully analyzed {len(photo_metadata)} photos")
        self.photo_metadata = photo_metadata
        
        return photo_metadata
    
    def get_detected_dates(self) -> List[str]:
        """Get list of unique dates detected in the photos."""
        if not self.photo_metadata:
            return []
        
        dates = set()
        for meta in self.photo_metadata:
            dates.add(meta['timestamp'].strftime("%Y-%m-%d"))
        
        return sorted(list(dates))
    
    def detect_game_boundaries(self, photo_metadata: List[Dict]) -> List[Tuple[int, int]]:
        """
        Detect game boundaries based on temporal gaps.
        
        Args:
            photo_metadata: List of photo metadata dictionaries
            
        Returns:
            List of (start_index, end_index) tuples for each game
        """
        logger.info("Detecting game boundaries")
        
        if len(photo_metadata) < 2:
            return [(0, len(photo_metadata) - 1)] if photo_metadata else []
        
        boundaries = []
        current_start = 0
        
        for i in range(1, len(photo_metadata)):
            prev_time = photo_metadata[i-1]['timestamp']
            curr_time = photo_metadata[i]['timestamp']
            
            gap_seconds = (curr_time - prev_time).total_seconds()
            gap_minutes = gap_seconds / 60
            
            # Check if this gap indicates a new game
            if gap_minutes >= self.config.min_gap_minutes:
                # End current game
                game_duration = (prev_time - photo_metadata[current_start]['timestamp']).total_seconds() / 60
                
                if game_duration >= self.config.min_game_duration_minutes:
                    boundaries.append((current_start, i - 1))
                    logger.info(f"Game boundary detected: {photo_metadata[current_start]['timestamp']} to {prev_time} "
                              f"(duration: {game_duration:.1f} min, gap: {gap_minutes:.1f} min)")
                else:
                    logger.info(f"Skipping short session: {game_duration:.1f} min < {self.config.min_game_duration_minutes} min")
                
                current_start = i
        
        # Add final game
        if current_start < len(photo_metadata):
            final_duration = (photo_metadata[-1]['timestamp'] - photo_metadata[current_start]['timestamp']).total_seconds() / 60
            if final_duration >= self.config.min_game_duration_minutes:
                boundaries.append((current_start, len(photo_metadata) - 1))
                logger.info(f"Final game: {photo_metadata[current_start]['timestamp']} to {photo_metadata[-1]['timestamp']} "
                          f"(duration: {final_duration:.1f} min)")
        
        logger.info(f"Detected {len(boundaries)} games")
        return boundaries
    
    def create_game_sessions(self, photo_metadata: List[Dict], boundaries: List[Tuple[int, int]]) -> List[GameSession]:
        """
        Create game session objects from boundaries.
        
        Args:
            photo_metadata: List of photo metadata dictionaries
            boundaries: List of (start_index, end_index) tuples
            
        Returns:
            List of GameSession objects
        """
        logger.info("Creating game sessions")
        
        games = []
        
        for i, (start_idx, end_idx) in enumerate(boundaries):
            game_photos = [meta['path'] for meta in photo_metadata[start_idx:end_idx + 1]]
            start_time = photo_metadata[start_idx]['timestamp']
            end_time = photo_metadata[end_idx]['timestamp']
            
            # Calculate gaps
            gap_before = None
            gap_after = None
            
            if start_idx > 0:
                prev_time = photo_metadata[start_idx - 1]['timestamp']
                gap_before = int((start_time - prev_time).total_seconds())
            
            if end_idx < len(photo_metadata) - 1:
                next_time = photo_metadata[end_idx + 1]['timestamp']
                gap_after = int((next_time - end_time).total_seconds())
            
            game = GameSession(
                game_id=i + 1,
                start_time=start_time,
                end_time=end_time,
                photo_count=len(game_photos),
                photo_files=game_photos,
                gap_before=gap_before,
                gap_after=gap_after
            )
            
            games.append(game)
            
            logger.info(f"Game {game.game_id}: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')} "
                       f"({game.photo_count} photos, {gap_before}s before, {gap_after}s after)")
        
        self.games = games
        return games
    
    def generate_game_summary(self) -> Dict:
        """
        Generate a summary of detected games.
        
        Returns:
            Dictionary containing game summary information
        """
        summary = {
            'total_games': len(self.games),
            'total_photos': sum(game.photo_count for game in self.games),
            'games': []
        }
        
        for game in self.games:
            duration_minutes = (game.end_time - game.start_time).total_seconds() / 60
            
            game_info = {
                'game_id': game.game_id,
                'start_time': game.start_time.strftime('%H:%M:%S'),
                'end_time': game.end_time.strftime('%H:%M:%S'),
                'duration_minutes': round(duration_minutes, 1),
                'photo_count': game.photo_count,
                'gap_before_minutes': round(game.gap_before / 60, 1) if game.gap_before else None,
                'gap_after_minutes': round(game.gap_after / 60, 1) if game.gap_after else None
            }
            
            summary['games'].append(game_info)
        
        return summary
    
    def create_game_folders(self, output_dir: Path, create_symlinks: bool = True) -> Dict[str, Path]:
        """
        Create game folders and organize photos.
        
        Args:
            output_dir: Output directory for game folders
            create_symlinks: Whether to create symlinks (True) or copy files (False)
            
        Returns:
            Dictionary mapping game IDs to folder paths
        """
        logger.info(f"Creating game folders in {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        game_folders = {}
        
        for game in self.games:
            # Create game folder name
            game_folder_name = f"Game_{game.game_id:02d}_{game.start_time.strftime('%H%M')}-{game.end_time.strftime('%H%M')}"
            game_folder = output_dir / game_folder_name
            game_folder.mkdir(exist_ok=True)
            
            logger.info(f"Creating {game_folder_name} with {game.photo_count} photos")
            
            # Copy or symlink photos
            for photo_path in game.photo_files:
                if create_symlinks:
                    symlink_path = game_folder / photo_path.name
                    if not symlink_path.exists():
                        symlink_path.symlink_to(photo_path)
                else:
                    dest_path = game_folder / photo_path.name
                    if not dest_path.exists():
                        shutil.copy2(photo_path, dest_path)
            
            game_folders[f"Game_{game.game_id:02d}"] = game_folder
        
        return game_folders
    
    def save_detection_report(self, output_dir: Path) -> Path:
        """
        Save a detailed detection report.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to the saved report file
        """
        report_path = output_dir / "game_detection_report.json"
        
        report = {
            'detection_config': {
                'min_game_duration_minutes': self.config.min_game_duration_minutes,
                'max_gap_minutes': self.config.max_gap_minutes,
                'min_gap_minutes': self.config.min_gap_minutes,
                'min_photos_per_game': self.config.min_photos_per_game,
            },
            'summary': self.generate_game_summary(),
            'detailed_games': []
        }
        
        for game in self.games:
            game_detail = {
                'game_id': game.game_id,
                'start_time': game.start_time.isoformat(),
                'end_time': game.end_time.isoformat(),
                'photo_count': game.photo_count,
                'gap_before_seconds': game.gap_before,
                'gap_after_seconds': game.gap_after,
                'photo_files': [str(photo) for photo in game.photo_files[:10]]  # First 10 files
            }
            report['detailed_games'].append(game_detail)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Detection report saved to {report_path}")
        return report_path
    
    def detect_games_from_directory(self, 
                                   input_dir: Path, 
                                   output_dir: Path,
                                   pattern: str = "20250920_*",
                                   create_symlinks: bool = True) -> Dict:
        """
        Complete game detection workflow from directory.
        
        Args:
            input_dir: Input directory containing photos
            output_dir: Output directory for organized games
            pattern: File pattern to match
            create_symlinks: Whether to create symlinks or copy files
            
        Returns:
            Dictionary containing detection results
        """
        logger.info(f"Starting game detection from {input_dir}")
        
        # Find photo files
        photo_paths = list(input_dir.glob(pattern))
        photo_paths = [p for p in photo_paths if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if not photo_paths:
            logger.error(f"No photos found matching pattern {pattern} in {input_dir}")
            return {}
        
        logger.info(f"Found {len(photo_paths)} photos")
        
        # Analyze timestamps
        photo_metadata = self.analyze_photo_timestamps(photo_paths)
        
        if not photo_metadata:
            logger.error("No valid timestamps found")
            return {}
        
        # Detect game boundaries
        boundaries = self.detect_game_boundaries(photo_metadata)
        
        if not boundaries:
            logger.warning("No games detected")
            return {}
        
        # Create game sessions
        games = self.create_game_sessions(photo_metadata, boundaries)
        
        # Create output folders
        game_folders = self.create_game_folders(output_dir, create_symlinks)
        
        # Save report
        report_path = self.save_detection_report(output_dir)
        
        # Generate summary
        summary = self.generate_game_summary()
        
        logger.info(f"Game detection complete: {summary['total_games']} games, {summary['total_photos']} photos")
        
        return {
            'games': games,
            'game_folders': game_folders,
            'summary': summary,
            'report_path': report_path
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect games in soccer photos")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--pattern", "-p", default="20250920_*", help="File pattern")
    parser.add_argument("--min-duration", type=int, default=30, help="Minimum game duration (minutes)")
    parser.add_argument("--min-gap", type=int, default=10, help="Minimum gap to separate games (minutes)")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    
    # Create configuration
    config = GameDetectionConfig(
        min_game_duration_minutes=args.min_duration,
        min_gap_minutes=args.min_gap
    )
    
    # Create detector
    detector = GameDetector(config)
    
    # Run detection
    results = detector.detect_games_from_directory(
        input_dir=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        create_symlinks=not args.copy
    )
    
    if results:
        print(f"\nGame Detection Results:")
        print(f"Total Games: {results['summary']['total_games']}")
        print(f"Total Photos: {results['summary']['total_photos']}")
        print(f"\nGames:")
        for game in results['summary']['games']:
            print(f"  Game {game['game_id']}: {game['start_time']} - {game['end_time']} "
                  f"({game['duration_minutes']} min, {game['photo_count']} photos)")
    else:
        print("No games detected")


if __name__ == "__main__":
    main()
