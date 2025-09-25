"""
Ball Detection Commands

CLI commands for ball detection and tracking operations.

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
def ball_group():
    """Ball detection and tracking commands."""
    pass


@ball_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for results')
@click.option('--confidence', '-c',
              type=float,
              default=0.5,
              help='Detection confidence threshold (0.0-1.0)')
@click.option('--min-size', 'min_size',
              type=int,
              default=16,
              help='Minimum ball size in pixels')
@click.option('--max-size', 'max_size',
              type=int,
              help='Maximum ball size in pixels')
@click.option('--enable-tracking', 'enable_tracking',
              is_flag=True,
              help='Enable ball tracking across frames')
@click.option('--save-sidecar/--no-sidecar',
              default=True,
              help='Save results to sidecar files')
@click.option('--extract-balls', 'extract_balls',
              is_flag=True,
              help='Extract detected balls to separate images')
@click.pass_context
def detect(ctx: click.Context, 
           input_path: Path, 
           output: Optional[Path],
           confidence: float,
           min_size: int,
           max_size: Optional[int],
           enable_tracking: bool,
           save_sidecar: bool,
           extract_balls: bool):
    """
    Detect balls in images.
    
    INPUT_PATH can be a single image file or a directory containing images.
    """
    
    core = get_core(ctx)
    
    # Determine input files
    if input_path.is_file():
        image_paths = [input_path]
    else:
        # Find all image files in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_paths:
        console.print("❌ No image files found", style="red")
        return
    
    console.print(f"⚽ Detecting balls in {len(image_paths)} images...", style="blue")
    
    # Perform ball detection
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing images...", total=len(image_paths))
        
        results = core.detect_balls(
            image_paths,
            save_sidecar=save_sidecar,
            confidence=confidence,
            min_size=min_size,
            max_size=max_size,
            enable_tracking=enable_tracking
        )
        
        progress.update(task, completed=len(image_paths))
    
    # Display results
    display_ball_results(results, extract_balls, output)


def display_ball_results(results: dict, extract_balls: bool, output_dir: Optional[Path]):
    """Display ball detection results."""
    
    # Create results table
    table = Table(title="Ball Detection Results")
    table.add_column("Image", style="cyan")
    table.add_column("Balls Found", style="green", justify="right")
    table.add_column("Confidence", style="yellow")
    table.add_column("Success", style="green")
    table.add_column("Error", style="red")
    
    total_balls = 0
    successful_images = 0
    avg_confidence = 0
    
    for image_path, result in results.items():
        if result.get('success', False):
            balls = result.get('balls', [])
            ball_count = len(balls)
            total_balls += ball_count
            successful_images += 1
            
            # Calculate average confidence
            if balls:
                confidences = [ball.get('confidence', 0) for ball in balls]
                avg_conf = sum(confidences) / len(confidences)
                avg_confidence += avg_conf
                confidence_str = f"{avg_conf:.2f}"
            else:
                confidence_str = "N/A"
            
            table.add_row(
                Path(image_path).name,
                str(ball_count),
                confidence_str,
                "✅",
                ""
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            table.add_row(
                Path(image_path).name,
                "0",
                "N/A",
                "❌",
                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
            )
    
    console.print(table)
    console.print(f"\n📊 Summary: {successful_images}/{len(results)} images processed, {total_balls} balls detected")
    
    if successful_images > 0:
        avg_confidence /= successful_images
        console.print(f"Average confidence: {avg_confidence:.2f}")
    
    # Extract balls if requested
    if extract_balls and output_dir:
        console.print(f"\n💾 Extracting balls to {output_dir}...", style="blue")
        # TODO: Implement ball extraction
        console.print("Ball extraction not yet implemented", style="yellow")


@ball_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for tracking results')
@click.option('--confidence', '-c',
              type=float,
              default=0.5,
              help='Detection confidence threshold (0.0-1.0)')
@click.option('--max-distance', 'max_distance',
              type=int,
              default=100,
              help='Maximum distance for ball tracking in pixels')
@click.option('--save-sidecar/--no-sidecar',
              default=True,
              help='Save results to sidecar files')
@click.pass_context
def track(ctx: click.Context, 
           input_path: Path, 
           output: Optional[Path],
           confidence: float,
           max_distance: int,
           save_sidecar: bool):
    """
    Track balls across multiple images (video frames).
    
    INPUT_PATH should be a directory containing sequential images or a video file.
    """
    
    core = get_core(ctx)
    
    console.print(f"🎯 Tracking balls in {input_path}...", style="blue")
    
    # TODO: Implement ball tracking
    console.print("Ball tracking not yet implemented", style="yellow")


@ball_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for analysis results')
@click.option('--confidence', '-c',
              type=float,
              default=0.5,
              help='Detection confidence threshold (0.0-1.0)')
@click.option('--save-sidecar/--no-sidecar',
              default=True,
              help='Save results to sidecar files')
@click.pass_context
def analyze(ctx: click.Context, 
            input_path: Path, 
            output: Optional[Path],
            confidence: float,
            save_sidecar: bool):
    """
    Analyze ball detection results and generate statistics.
    
    INPUT_PATH should be a directory containing images with ball detection sidecar files.
    """
    
    core = get_core(ctx)
    
    console.print(f"📊 Analyzing ball detection results in {input_path}...", style="blue")
    
    # TODO: Implement ball analysis
    console.print("Ball analysis not yet implemented", style="yellow")
