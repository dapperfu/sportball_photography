"""
Game Detection Commands

CLI commands for game boundary detection and splitting operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..utils import get_core

console = Console()


@click.group()
def game_group():
    """Game detection and splitting commands."""
    pass


@game_group.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default=Path('./games'),
              help='Output directory for organized games')
@click.option('--pattern', '-p',
              default='*_*',
              help='File pattern to match (e.g., "202509*_*" for Sep 2025)')
@click.option('--min-duration', 'min_duration',
              type=int,
              default=30,
              help='Minimum game duration in minutes')
@click.option('--min-gap', 'min_gap',
              type=int,
              default=10,
              help='Minimum gap to separate games in minutes')
@click.option('--min-photos', 'min_photos',
              type=int,
              default=50,
              help='Minimum photos per game')
@click.option('--copy/--symlink',
              default=False,
              help='Copy files instead of creating symlinks')
@click.option('--save-sidecar/--no-sidecar',
              default=True,
              help='Save results to sidecar files')
@click.pass_context
def detect(ctx: click.Context, 
           input_path: Path, 
           output: Path,
           pattern: str,
           min_duration: int,
           min_gap: int,
           min_photos: int,
           copy: bool,
           save_sidecar: bool):
    """
    Detect game boundaries in photos based on timestamps.
    
    INPUT_PATH should be a directory containing photos with timestamp filenames.
    """
    
    core = get_core(ctx)
    
    console.print(f"🎮 Detecting games in {input_path}...", style="blue")
    console.print(f"Pattern: {pattern}")
    console.print(f"Output: {output}")
    
    # Perform game detection
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Analyzing photos...", total=None)
        
        results = core.detect_games(
            input_path,
            pattern=pattern,
            save_sidecar=save_sidecar,
            min_duration=min_duration,
            min_gap=min_gap,
            min_photos=min_photos
        )
        
        progress.update(task, completed=True, description="Game detection complete")
    
    # Display results
    display_game_results(results, output, copy)


def display_game_results(results: dict, output_dir: Path, copy_files: bool):
    """Display game detection results."""
    
    if not results.get('success', False):
        console.print(f"❌ Game detection failed: {results.get('error', 'Unknown error')}", style="red")
        return
    
    games = results.get('games', [])
    if not games:
        console.print("❌ No games detected", style="red")
        return
    
    # Create results table
    table = Table(title="Game Detection Results")
    table.add_column("Game ID", style="cyan", justify="right")
    table.add_column("Start Time", style="green")
    table.add_column("End Time", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Photos", style="magenta", justify="right")
    table.add_column("Gap Before", style="blue")
    table.add_column("Gap After", style="blue")
    
    total_photos = 0
    total_duration = 0
    
    for game in games:
        duration_minutes = game.get('duration_minutes', 0)
        photo_count = game.get('photo_count', 0)
        gap_before = game.get('gap_before_minutes')
        gap_after = game.get('gap_after_minutes')
        
        total_photos += photo_count
        total_duration += duration_minutes
        
        table.add_row(
            str(game.get('game_id', 'N/A')),
            game.get('start_time_formatted', 'N/A'),
            game.get('end_time_formatted', 'N/A'),
            f"{duration_minutes:.1f} min",
            str(photo_count),
            f"{gap_before:.1f} min" if gap_before else "N/A",
            f"{gap_after:.1f} min" if gap_after else "N/A"
        )
    
    console.print(table)
    console.print(f"\n📊 Summary: {len(games)} games detected, {total_photos} photos, {total_duration:.1f} minutes total")
    
    # Create organized folders
    console.print(f"\n📁 Creating organized folders in {output_dir}...", style="blue")
    # TODO: Implement folder creation
    console.print("Folder creation not yet implemented", style="yellow")


@game_group.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--split-file', '-s',
              type=click.Path(path_type=Path),
              help='Text file with manual split points (one timestamp per line)')
@click.option('--pattern', '-p',
              default='*_*',
              help='File pattern to match')
@click.option('--copy/--symlink',
              default=False,
              help='Copy files instead of creating symlinks')
@click.option('--interactive', '-i',
              is_flag=True,
              help='Interactive mode for adding splits')
@click.pass_context
def split(ctx: click.Context, 
          input_path: Path, 
          output_dir: Path,
          split_file: Optional[Path],
          pattern: str,
          copy: bool,
          interactive: bool):
    """
    Split photos into games with optional manual split points.
    
    INPUT_PATH should be a directory containing photos.
    OUTPUT_DIR is where organized games will be created.
    """
    
    core = get_core(ctx)
    
    console.print(f"✂️  Splitting photos in {input_path} into games...", style="blue")
    console.print(f"Output: {output_dir}")
    
    # Load manual splits if provided
    manual_splits = []
    if split_file and split_file.exists():
        console.print(f"📄 Loading manual splits from {split_file}...", style="blue")
        # TODO: Implement split file loading
        console.print("Split file loading not yet implemented", style="yellow")
    elif split_file:
        console.print(f"⚠️  Split file not found: {split_file}", style="yellow")
    
    # Interactive mode
    if interactive:
        console.print("🖱️  Interactive mode not yet implemented", style="yellow")
    
    # Perform game splitting
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing photos...", total=None)
        
        # First detect games automatically
        results = core.detect_games(
            input_path,
            pattern=pattern,
            save_sidecar=True
        )
        
        progress.update(task, completed=True, description="Game splitting complete")
    
    # Display results
    display_game_results(results, output_dir, copy)


@game_group.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--pattern', '-p',
              default='*_*',
              help='File pattern to match')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output file for report')
@click.pass_context
def analyze(ctx: click.Context, 
            input_path: Path, 
            pattern: str,
            output: Optional[Path]):
    """
    Analyze photos and generate game statistics without splitting.
    
    INPUT_PATH should be a directory containing photos.
    """
    
    core = get_core(ctx)
    
    console.print(f"📊 Analyzing photos in {input_path}...", style="blue")
    
    # Perform analysis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Analyzing photos...", total=None)
        
        results = core.detect_games(
            input_path,
            pattern=pattern,
            save_sidecar=True
        )
        
        progress.update(task, completed=True, description="Analysis complete")
    
    # Display results
    display_game_results(results, Path.cwd(), False)
    
    # Save report if requested
    if output:
        console.print(f"💾 Saving report to {output}...", style="blue")
        # TODO: Implement report saving
        console.print("Report saving not yet implemented", style="yellow")
