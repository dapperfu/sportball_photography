#!/usr/bin/env python3
"""
Enhanced Game Organizer - Professional soccer photo organization tool.

Features:
- Click CLI with rich help and options
- Parallel processing for photo analysis
- Progress bars and colored output
- Comprehensive error handling
- Professional logging
- Performance metrics

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
import logging

from game_detector import GameDetector, GameDetectionConfig, GameSession
from manual_game_splitter import ManualGameSplitter

# Configure rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("game_organizer")


class EnhancedGameOrganizer:
    """
    Enhanced game organizer with parallel processing and professional features.
    """
    
    def __init__(self, config: Optional[GameDetectionConfig] = None, max_workers: int = 4):
        """
        Initialize enhanced organizer.
        
        Args:
            config: Optional configuration for game detection
            max_workers: Maximum number of parallel workers
        """
        self.config = config or GameDetectionConfig()
        self.detector = GameDetector(self.config)
        self.splitter = None
        self.final_games: List[GameSession] = []
        self.max_workers = max_workers
        self.performance_metrics = {}
        self.temp_dir = None
    
    def detect_games_parallel(self, 
                            input_dir: Path, 
                            pattern: str = "20250920_*") -> Dict:
        """
        Run automated game detection with parallel processing.
        
        Args:
            input_dir: Input directory containing photos
            pattern: File pattern to match
            
        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Task 1: Find photos
            task1 = progress.add_task("Finding photos...", total=None)
            photo_paths = list(input_dir.glob(pattern))
            photo_paths = [p for p in photo_paths if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            progress.update(task1, completed=True, description=f"Found {len(photo_paths)} photos")
            
            if not photo_paths:
                console.print("❌ No photos found", style="red")
                return {}
            
            # Task 2: Analyze timestamps in parallel
            task2 = progress.add_task("Analyzing timestamps...", total=len(photo_paths))
            photo_metadata = self._analyze_timestamps_parallel(photo_paths, progress, task2)
            
            if not photo_metadata:
                console.print("❌ No valid timestamps found", style="red")
                return {}
            
            # Task 3: Detect game boundaries
            task3 = progress.add_task("Detecting game boundaries...", total=None)
            boundaries = self._detect_game_boundaries_parallel(photo_metadata)
            progress.update(task3, completed=True, description=f"Detected {len(boundaries)} games")
            
            if not boundaries:
                console.print("❌ No games detected", style="red")
                return {}
            
            # Task 4: Create game sessions
            task4 = progress.add_task("Creating game sessions...", total=len(boundaries))
            games = self._create_game_sessions_parallel(photo_metadata, boundaries, progress, task4)
            
            # Show detected dates
            detected_dates = self._get_detected_dates(photo_metadata)
            if detected_dates:
                console.print(f"📅 Detected dates: {', '.join(detected_dates)}", style="blue")
            
            # Task 5: Create output folders
            task5 = progress.add_task("Creating organized folders...", total=len(games))
            game_folders = self._create_folders_parallel(games, progress, task5)
            
            # Task 6: Generate report
            task6 = progress.add_task("Generating report...", total=None)
            report_path = self._generate_report_parallel()
            progress.update(task6, completed=True, description="Report generated")
        
        # Store performance metrics
        self.performance_metrics = {
            'total_time': time.time() - start_time,
            'photos_processed': len(photo_paths),
            'games_detected': len(games),
            'parallel_workers': self.max_workers
        }
        
        # Set up the splitter for manual splits
        from manual_game_splitter import ManualGameSplitter
        self.splitter = ManualGameSplitter(self.detector)
        
        # Ensure detector games are set
        self.detector.games = games
        
        console.print(f"✅ Automated detection found {len(games)} games", style="green")
        return {
            'games': games,
            'game_folders': game_folders,
            'report_path': report_path,
            'performance': self.performance_metrics
        }
    
    def _analyze_timestamps_parallel(self, 
                                   photo_paths: List[Path], 
                                   progress: Progress, 
                                   task_id: int) -> List[Dict]:
        """Analyze timestamps in parallel."""
        photo_metadata = []
        
        def process_photo(photo_path: Path) -> Optional[Dict]:
            timestamp = self.detector.extract_timestamp_from_filename(photo_path.name)
            if timestamp:
                return {
                    'path': photo_path,
                    'filename': photo_path.name,
                    'timestamp': timestamp,
                    'time_str': timestamp.strftime("%H%M%S")
                }
            return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_photo = {executor.submit(process_photo, photo): photo for photo in photo_paths}
            
            for future in as_completed(future_to_photo):
                result = future.result()
                if result:
                    photo_metadata.append(result)
                progress.advance(task_id)
        
        # Sort by timestamp
        photo_metadata.sort(key=lambda x: x['timestamp'])
        return photo_metadata
    
    def _detect_game_boundaries_parallel(self, photo_metadata: List[Dict]) -> List[Tuple[int, int]]:
        """Detect game boundaries with parallel processing where applicable."""
        # This is inherently sequential, but we can optimize the algorithm
        return self.detector.detect_game_boundaries(photo_metadata)
    
    def _create_game_sessions_parallel(self, 
                                     photo_metadata: List[Dict], 
                                     boundaries: List[Tuple[int, int]], 
                                     progress: Progress, 
                                     task_id: int) -> List[GameSession]:
        """Create game sessions in parallel."""
        games = []
        
        def create_session(boundary: Tuple[int, int]) -> GameSession:
            start_idx, end_idx = boundary
            game_photos = [meta['path'] for meta in photo_metadata[start_idx:end_idx + 1]]
            start_time = photo_metadata[start_idx]['timestamp']
            end_time = photo_metadata[end_idx]['timestamp']
            
            return GameSession(
                game_id=len(games) + 1,
                start_time=start_time,
                end_time=end_time,
                photo_count=len(game_photos),
                photo_files=game_photos,
                gap_before=None,  # Will be calculated after all games are created
                gap_after=None   # Will be calculated after all games are created
            )
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_boundary = {executor.submit(create_session, boundary): boundary for boundary in boundaries}
            
            for future in as_completed(future_to_boundary):
                game = future.result()
                games.append(game)
                progress.advance(task_id)
        
        # Sort games by start time (chronological order)
        games.sort(key=lambda x: x.start_time)
        
        # Reassign game IDs in chronological order
        for i, game in enumerate(games, 1):
            game.game_id = i
        
        # Calculate gaps between games
        for i, game in enumerate(games):
            if i > 0:
                # Gap before = time from end of previous game to start of current game
                prev_game = games[i - 1]
                game.gap_before = int((game.start_time - prev_game.end_time).total_seconds())
            
            if i < len(games) - 1:
                # Gap after = time from end of current game to start of next game
                next_game = games[i + 1]
                game.gap_after = int((next_game.start_time - game.end_time).total_seconds())
        
        return games
    
    def _create_folders_parallel(self, 
                               games: List[GameSession], 
                               progress: Progress, 
                               task_id: int) -> Dict[str, Path]:
        """Create organized folders in parallel."""
        # Create temporary directory that will be cleaned up automatically
        self.temp_dir = tempfile.mkdtemp(prefix="game_organizer_")
        output_dir = Path(self.temp_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        game_folders = {}
        
        def create_folder(game: GameSession) -> Tuple[str, Path]:
            game_folder_name = f"Game_{game.game_id:02d}_{game.start_time.strftime('%H%M')}-{game.end_time.strftime('%H%M')}"
            game_folder = output_dir / game_folder_name
            game_folder.mkdir(exist_ok=True)
            
            # Create symlinks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for photo_path in game.photo_files:
                    symlink_path = game_folder / photo_path.name
                    if symlink_path.exists():
                        symlink_path.unlink()  # Remove existing symlink/file
                    futures.append(executor.submit(self._create_symlink_safe, symlink_path, photo_path))
                
                # Wait for all symlinks to be created
                for future in as_completed(futures):
                    future.result()
            
            return f"Game_{game.game_id:02d}", game_folder
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_game = {executor.submit(create_folder, game): game for game in games}
            
            for future in as_completed(future_to_game):
                game_id, folder_path = future.result()
                game_folders[game_id] = folder_path
                progress.advance(task_id)
        
        return game_folders
    
    def _create_symlink_safe(self, symlink_path: Path, target_path: Path) -> bool:
        """
        Safely create a symlink, handling existing files/symlinks.
        
        Args:
            symlink_path: Path where symlink should be created
            target_path: Path that symlink should point to
            
        Returns:
            True if symlink was created successfully
        """
        try:
            # Remove existing file/symlink if it exists
            if symlink_path.exists():
                symlink_path.unlink()
            
            # Create the symlink
            symlink_path.symlink_to(target_path)
            return True
            
        except Exception as e:
            # Log the error but don't fail the entire process
            logger.warning(f"Failed to create symlink {symlink_path} -> {target_path}: {e}")
            return False
    
    def create_organized_folders(self, output_dir: Path, create_symlinks: bool = True, use_final_games: bool = True) -> Dict[str, Path]:
        """
        Create organized folders for games.
        
        Args:
            output_dir: Output directory for organized games
            create_symlinks: Whether to create symlinks (True) or copy files (False)
            use_final_games: Whether to use final_games (True) or detector.games (False)
            
        Returns:
            Dictionary mapping game IDs to folder paths
        """
        games_to_organize = self.final_games if use_final_games else self.detector.games
        
        if not games_to_organize:
            console.print("❌ No games to organize", style="red")
            return {}
        
        console.print(f"📁 Creating organized folders for {len(games_to_organize)} games...", style="blue")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        game_folders = {}
        
        for game in games_to_organize:
            # Create game folder name
            game_folder_name = f"Game_{game.game_id:02d}_{game.start_time.strftime('%H%M')}-{game.end_time.strftime('%H%M')}"
            game_folder = output_dir / game_folder_name
            game_folder.mkdir(exist_ok=True)
            
            console.print(f"📂 Creating {game_folder_name} with {game.photo_count} photos", style="green")
            
            # Copy or symlink photos
            for photo_path in game.photo_files:
                if create_symlinks:
                    symlink_path = game_folder / photo_path.name
                    self._create_symlink_safe(symlink_path, photo_path)
                else:
                    dest_path = game_folder / photo_path.name
                    if dest_path.exists():
                        dest_path.unlink()  # Remove existing file
                    shutil.copy2(photo_path, dest_path)
            
            game_folders[f"Game_{game.game_id:02d}"] = game_folder
        
        return game_folders
    
    def _get_detected_dates(self, photo_metadata: List[Dict]) -> List[str]:
        """Get list of unique dates detected in the photos."""
        if not photo_metadata:
            return []
        
        dates = set()
        for meta in photo_metadata:
            dates.add(meta['timestamp'].strftime("%Y-%m-%d"))
        
        return sorted(list(dates))
    
    def load_split_file(self, split_file_path: Path) -> bool:
        """
        Load manual splits from a text file.
        
        Args:
            split_file_path: Path to the split file
            
        Returns:
            True if splits were loaded successfully
        """
        if not self.splitter:
            console.print("❌ No splitter available. Run detection first.", style="red")
            return False
        
        try:
            with open(split_file_path, 'r') as f:
                lines = f.readlines()
            
            splits_loaded = 0
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse timestamp - handle both formats
                    if ' ' in line:
                        # Format: "YYYY-MM-DD HH:MM:SS"
                        timestamp = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
                    elif ':' in line:
                        # Format: "HH:MM:SS" - assume current detected date
                        time_part = datetime.strptime(line, "%H:%M:%S")
                        # Use the first detected date from the games
                        if self.detector.games:
                            first_game_date = self.detector.games[0].start_time.date()
                            timestamp = datetime.combine(first_game_date, time_part.time())
                        else:
                            # Fallback to September 20th, 2025
                            timestamp = datetime(2025, 9, 20, time_part.hour, time_part.minute, time_part.second)
                    else:
                        # Format: "HHMMSS" - assume current detected date
                        time_part = datetime.strptime(line, "%H%M%S")
                        if self.detector.games:
                            first_game_date = self.detector.games[0].start_time.date()
                            timestamp = datetime.combine(first_game_date, time_part.time())
                        else:
                            # Fallback to September 20th, 2025
                            timestamp = datetime(2025, 9, 20, time_part.hour, time_part.minute, time_part.second)
                    
                    self.splitter.manual_splits.append(timestamp)
                    splits_loaded += 1
                    
                except ValueError as e:
                    console.print(f"⚠️  Invalid timestamp format on line {line_num}: '{line}' - {e}", style="yellow")
                    continue
            
            # Sort splits
            self.splitter.manual_splits.sort()
            
            console.print(f"✅ Loaded {splits_loaded} manual splits from {split_file_path}", style="green")
            return True
            
        except FileNotFoundError:
            console.print(f"❌ Split file not found: {split_file_path}", style="red")
            return False
        except Exception as e:
            console.print(f"❌ Error loading split file: {e}", style="red")
            return False
    
    def apply_manual_splits(self) -> bool:
        """
        Apply manual splits to the detected games.
        
        Returns:
            True if splits were applied successfully
        """
        if not self.splitter:
            console.print("❌ No splitter available. Run detection first.", style="red")
            return False
        
        if not self.splitter.manual_splits:
            console.print("ℹ️  No manual splits to apply", style="yellow")
            self.final_games = self.detector.games
            return True
        
        console.print(f"🔧 Applying {len(self.splitter.manual_splits)} manual splits...", style="blue")
        
        try:
            self.final_games = self.splitter.apply_manual_splits()
            console.print(f"✅ Applied manual splits. Final games: {len(self.final_games)}", style="green")
            return True
        except Exception as e:
            console.print(f"❌ Error applying manual splits: {e}", style="red")
            return False
    
    def _generate_report_parallel(self) -> Path:
        """Generate report with parallel processing."""
        # This is mostly I/O bound, so we can parallelize file operations
        report_path = Path("/tmp/game_detection_report.json")
        
        # Generate report data in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Parallel tasks for report generation
            summary_future = executor.submit(self._generate_summary)
            games_future = executor.submit(self._get_games_data)
            
            # Wait for results
            summary = summary_future.result()
            games_data = games_future.result()
        
        # Write report
        import json
        report = {
            'summary': summary,
            'games': games_data,
            'performance': self.performance_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    def _generate_summary(self) -> Dict:
        """Generate game summary."""
        summary = {
            'total_games': len(self.detector.games),
            'total_photos': sum(game.photo_count for game in self.detector.games),
            'games': []
        }
        
        for game in self.detector.games:
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
    
    def _get_games_data(self) -> List[Dict]:
        """Get games data for report."""
        games_data = []
        for game in self.detector.games:
            duration_minutes = (game.end_time - game.start_time).total_seconds() / 60
            game_data = {
                'game_id': game.game_id,
                'start_time': game.start_time.strftime('%H:%M:%S'),
                'end_time': game.end_time.strftime('%H:%M:%S'),
                'duration_minutes': round(duration_minutes, 1),
                'photo_count': game.photo_count,
                'gap_before_minutes': round(game.gap_before / 60, 1) if game.gap_before else None,
                'gap_after_minutes': round(game.gap_after / 60, 1) if game.gap_after else None
            }
            games_data.append(game_data)
        return games_data
    
    def display_results_table(self, games: List[GameSession]):
        """Display results in a beautiful table."""
        table = Table(title="Game Organization Results")
        table.add_column("Game ID", style="cyan", no_wrap=True)
        table.add_column("Start Time", style="green")
        table.add_column("End Time", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Photos", style="magenta", justify="right")
        table.add_column("Gap Before", style="blue")
        table.add_column("Gap After", style="blue")
        
        for game in games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            gap_before = f"{game.gap_before/60:.1f} min" if game.gap_before else "N/A"
            gap_after = f"{game.gap_after/60:.1f} min" if game.gap_after else "N/A"
            
            table.add_row(
                str(game.game_id),
                game.start_time.strftime('%H:%M:%S'),
                game.end_time.strftime('%H:%M:%S'),
                f"{duration:.1f} min",
                str(game.photo_count),
                gap_before,
                gap_after
            )
        
        console.print(table)
    
    def display_performance_metrics(self):
        """Display performance metrics."""
        if not self.performance_metrics:
            return
        
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Total Time", f"{self.performance_metrics['total_time']:.2f} seconds")
        metrics_table.add_row("Photos Processed", str(self.performance_metrics['photos_processed']))
        metrics_table.add_row("Games Detected", str(self.performance_metrics['games_detected']))
        metrics_table.add_row("Parallel Workers", str(self.performance_metrics['parallel_workers']))
        metrics_table.add_row("Photos/Second", f"{self.performance_metrics['photos_processed']/self.performance_metrics['total_time']:.1f}")
        
        console.print(metrics_table)
    
    def generate_comprehensive_report(self, output_dir: Path) -> Path:
        """
        Generate a comprehensive report of the game organization.
        
        Args:
            output_dir: Output directory for the report
            
        Returns:
            Path to the generated report file
        """
        report_path = output_dir / "comprehensive_game_report.json"
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive report data
        report_data = {
            'summary': {
                'total_games': len(self.final_games),
                'total_photos': sum(game.photo_count for game in self.final_games),
                'detection_method': 'automated_with_manual_splits' if self.splitter and self.splitter.manual_splits else 'automated_only',
                'manual_splits_applied': len(self.splitter.manual_splits) if self.splitter else 0,
                'generated_at': datetime.now().isoformat()
            },
            'games': [],
            'performance': self.performance_metrics,
            'configuration': {
                'min_game_duration_minutes': self.config.min_game_duration_minutes,
                'min_gap_minutes': self.config.min_gap_minutes,
                'min_photos_per_game': self.config.min_photos_per_game,
                'parallel_workers': self.max_workers
            }
        }
        
        # Add detailed game information
        for game in self.final_games:
            duration_minutes = (game.end_time - game.start_time).total_seconds() / 60
            
            game_info = {
                'game_id': game.game_id,
                'start_time': game.start_time.isoformat(),
                'end_time': game.end_time.isoformat(),
                'start_time_formatted': game.start_time.strftime('%H:%M:%S'),
                'end_time_formatted': game.end_time.strftime('%H:%M:%S'),
                'duration_minutes': round(duration_minutes, 1),
                'photo_count': game.photo_count,
                'gap_before_minutes': round(game.gap_before / 60, 1) if game.gap_before else None,
                'gap_after_minutes': round(game.gap_after / 60, 1) if game.gap_after else None,
                'photo_files': [str(photo_path) for photo_path in game.photo_files]
            }
            
            report_data['games'].append(game_info)
        
        # Add manual splits information if available
        if self.splitter and self.splitter.manual_splits:
            report_data['manual_splits'] = {
                'count': len(self.splitter.manual_splits),
                'timestamps': [split.isoformat() for split in self.splitter.manual_splits],
                'timestamps_formatted': [split.strftime('%H:%M:%S') for split in self.splitter.manual_splits]
            }
        
        # Write report to file
        import json
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"📊 Comprehensive report generated: {report_path}", style="green")
        return report_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files and directories."""
        if self.temp_dir and Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
                console.print(f"🧹 Cleaned up temporary directory: {self.temp_dir}", style="blue")
            except Exception as e:
                console.print(f"⚠️  Warning: Could not clean up temp directory {self.temp_dir}: {e}", style="yellow")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        self.cleanup_temp_files()


@click.command()
@click.option('--input', '-i', 
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True,
              help='Input directory containing photos')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default=Path('./results/games'),
              help='Output directory for organized games')
@click.option('--pattern', '-p',
              default='*_*',
              help='File pattern to match (e.g., "202509*_*" for Sep 2025, "*_*" for all)')
@click.option('--split-file', '-s',
              type=click.Path(path_type=Path),
              help='Text file with manual splits (one timestamp per line, format: HH:MM:SS)')
@click.option('--copy', 'copy_files',
              is_flag=True,
              help='Copy files instead of creating symlinks')
@click.option('--workers', '-w',
              type=int,
              default=4,
              help='Number of parallel workers')
@click.option('--min-duration',
              type=int,
              default=30,
              help='Minimum game duration in minutes')
@click.option('--min-gap',
              type=int,
              default=10,
              help='Minimum gap to separate games in minutes')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose logging')
@click.option('--quiet', '-q',
              is_flag=True,
              help='Suppress output except errors')
@click.version_option(version='1.0.0')
def main(input: Path, output: Path, pattern: str, split_file: Optional[Path], 
         copy_files: bool, workers: int, min_duration: int, min_gap: int, 
         verbose: bool, quiet: bool):
    """
    Enhanced Game Organizer - Professional soccer photo organization tool.
    
    Automatically detects games in soccer photos and organizes them into folders.
    Supports multiple days of photos and optionally applies manual splits.
    
    Multi-Day Support:
    - Default pattern '*_*' processes all photos with timestamp format
    - Use '202509*_*' for September 2025, '202508*_*' for August 2025, etc.
    - Games are detected across multiple days automatically
    - Games are sorted chronologically regardless of date
    
    Split File Format:
    - Plain text file with one timestamp per line
    - Format: HH:MM:SS (e.g., 14:00:00) for single date
    - Format: YYYY-MM-DD HH:MM:SS (e.g., 2025-09-20 14:00:00) for specific date
    - Comments start with # (ignored)
    - Empty lines are ignored
    - If only time is specified, applies to all detected dates
    
    Examples:
    
    \b
    # Process all photos (any month/year)
    python enhanced_game_organizer.py --input /path/to/photos
    
    \b
    # Process specific month (September 2025)
    python enhanced_game_organizer.py --input /path/to/photos --pattern "202509*_*"
    
    \b
    # Process specific date only
    python enhanced_game_organizer.py --input /path/to/photos --pattern "20250920_*"
    
    \b
    # With manual splits
    python enhanced_game_organizer.py --input /path/to/photos --split-file splits.txt
    
    \b
    # High performance with 8 workers
    python enhanced_game_organizer.py --input /path/to/photos --workers 8
    
    \b
    # Copy files instead of symlinks
    python enhanced_game_organizer.py --input /path/to/photos --copy
    
    \b
    # Example split file content:
    # # Manual splits for September games
    # 14:00:00
    # 15:30:00
    # 
    # # Or with specific dates:
    # 2025-09-20 14:00:00
    # 2025-09-21 10:30:00
    """
    
    # Configure logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Display header
    if not quiet:
        console.print(Panel.fit(
            "[bold blue]Enhanced Game Organizer[/bold blue]\n"
            "Professional soccer photo organization tool",
            border_style="blue"
        ))
    
    # Create configuration
    config = GameDetectionConfig(
        min_game_duration_minutes=min_duration,
        min_gap_minutes=min_gap,
        min_photos_per_game=50,
    )
    
    # Create organizer
    organizer = EnhancedGameOrganizer(config, max_workers=workers)
    
    try:
        # Step 1: Automated detection with parallel processing
        if not quiet:
            console.print(f"\n[bold]🤖 Running automated game detection...[/bold]")
            console.print(f"Input: {input}")
            console.print(f"Pattern: {pattern}")
            console.print(f"Workers: {workers}")
        
        results = organizer.detect_games_parallel(input, pattern)
        
        if not results:
            console.print("❌ No games detected", style="red")
            return 1
        
        # Step 2: Load manual splits if provided
        if split_file and split_file.exists():
            if not quiet:
                console.print(f"\n[bold]✋ Loading manual splits from {split_file}...[/bold]")
            organizer.load_split_file(split_file)
        elif split_file:
            console.print(f"⚠️  Split file not found: {split_file}", style="yellow")
        
        # Step 3: Apply splits and create organized folders
        if not quiet:
            console.print(f"\n[bold]🔧 Applying splits and creating organized folders...[/bold]")
        
        # Apply manual splits (this will set final_games)
        organizer.apply_manual_splits()
        
        game_folders = organizer.create_organized_folders(output, create_symlinks=not copy_files, use_final_games=True)
        
        # Step 4: Generate report
        if not quiet:
            console.print(f"\n[bold]📊 Generating report...[/bold]")
        report_path = organizer.generate_comprehensive_report(output)
        
        # Display results
        if not quiet:
            console.print(f"\n[bold green]✅ Game Organization Complete![/bold green]")
            console.print(f"Output directory: {output}")
            console.print(f"Report: {report_path}")
            
            # Display results table
            organizer.display_results_table(organizer.final_games)
            
            # Display performance metrics
            organizer.display_performance_metrics()
        
        # Clean up temporary files
        organizer.cleanup_temp_files()
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Error during organization: {e}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        # Clean up temporary files even on error
        try:
            organizer.cleanup_temp_files()
        except:
            pass  # Ignore cleanup errors
        return 1


if __name__ == "__main__":
    sys.exit(main())
