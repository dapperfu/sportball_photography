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

import sys
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

from unified_game_organizer import UnifiedGameOrganizer, GameDetectionConfig, GameSession

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


class EnhancedGameOrganizer(UnifiedGameOrganizer):
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
        super().__init__(config)
        self.max_workers = max_workers
        self.performance_metrics = {}
    
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
            
            # Calculate gaps
            gap_before = None
            gap_after = None
            
            if start_idx > 0:
                prev_time = photo_metadata[start_idx - 1]['timestamp']
                gap_before = int((start_time - prev_time).total_seconds())
            
            if end_idx < len(photo_metadata) - 1:
                next_time = photo_metadata[end_idx + 1]['timestamp']
                gap_after = int((next_time - end_time).total_seconds())
            
            return GameSession(
                game_id=len(games) + 1,
                start_time=start_time,
                end_time=end_time,
                photo_count=len(game_photos),
                photo_files=game_photos,
                gap_before=gap_before,
                gap_after=gap_after
            )
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_boundary = {executor.submit(create_session, boundary): boundary for boundary in boundaries}
            
            for future in as_completed(future_to_boundary):
                game = future.result()
                games.append(game)
                progress.advance(task_id)
        
        # Sort games by ID
        games.sort(key=lambda x: x.game_id)
        return games
    
    def _create_folders_parallel(self, 
                               games: List[GameSession], 
                               progress: Progress, 
                               task_id: int) -> Dict[str, Path]:
        """Create organized folders in parallel."""
        output_dir = Path("/tmp")  # Temporary for this demo
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
                    if not symlink_path.exists():
                        futures.append(executor.submit(symlink_path.symlink_to, photo_path))
                
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
              default='20250920_*',
              help='File pattern to match')
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
    Optionally apply manual splits from a text file.
    
    Split File Format:
    - Plain text file with one timestamp per line
    - Format: HH:MM:SS (e.g., 14:00:00)
    - Comments start with # (ignored)
    - Empty lines are ignored
    
    Examples:
    
    \b
    # Basic usage
    python enhanced_game_organizer.py --input /path/to/photos
    
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
    # # Manual splits for September 20th games
    # 14:00:00
    # 15:30:00
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
        max_photos_per_game=2000
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
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Error during organization: {e}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
