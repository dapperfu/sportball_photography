"""
Face Detection Commands

CLI commands for face detection and recognition operations.

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
from ...sidecar import Sidecar, OperationType

console = Console()


@click.group()
def face_group():
    """Face detection and recognition commands."""
    pass


@face_group.command()
@click.argument('input_pattern', type=str)
@click.option('--border-padding', '-b', 
              default=0.25, 
              help='Border padding percentage (0.25 = 25%)')
@click.option('--max-images', '-m', 
              default=None, 
              type=int, 
              help='Maximum number of images to process')
@click.option('--gpu/--no-gpu', 
              default=True, 
              help='Use GPU acceleration if available')
@click.option('--force', '-f', 
              is_flag=True, 
              help='Force detection even if JSON sidecar exists')
@click.option('--verbose', '-v', 
              count=True, 
              help='Enable verbose logging (-v for info, -vv for debug)')
@click.pass_context
def detect(ctx: click.Context, 
           input_pattern: str,
           border_padding: float,
           max_images: Optional[int],
           gpu: bool,
           force: bool,
           verbose: int):
    """
    Detect faces in images and save comprehensive data to JSON sidecar files.
    
    INPUT_PATTERN can be a file pattern, directory path, or single image file.
    Supports recursive directory scanning and pattern matching.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        console.print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        console.print("ℹ️  Info logging enabled", style="blue")
    
    core = get_core(ctx)
    
    # Initialize face detector with same parameters as original
    import sys
    sys.path.append('/projects/soccer_photo_sorter')
    from face_detection import FaceDetector
    detector = FaceDetector(
        border_padding=border_padding,
        use_gpu=gpu
    )
    
    console.print(f"🔍 Starting face detection with {border_padding*100:.0f}% border padding", style="blue")
    
    # Use the detector's method directly (same as original)
    results = detector.detect_faces_in_images(input_pattern, max_images, force)
    
    if not results:
        console.print("❌ No images processed", style="red")
        return
    
    # Calculate summary statistics
    total_images = len(results)
    total_faces_found = sum(result.faces_found for result in results)
    total_time = sum(result.detection_time for result in results)
    
    # Display results
    display_face_detection_results(results, total_images, total_faces_found, total_time)


def display_face_results(results: dict, extract_faces: bool, output_dir: Optional[Path]):
    """Display face detection results."""
    
    # Create results table
    table = Table(title="Face Detection Results")
    table.add_column("Image", style="cyan")
    table.add_column("Faces Found", style="green", justify="right")
    table.add_column("Success", style="green")
    table.add_column("Error", style="red")
    
    total_faces = 0
    successful_images = 0
    
    for image_path, result in results.items():
        if result.get('success', False):
            face_count = len(result.get('faces', []))
            total_faces += face_count
            successful_images += 1
            
            table.add_row(
                Path(image_path).name,
                str(face_count),
                "✅",
                ""
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            table.add_row(
                Path(image_path).name,
                "0",
                "❌",
                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
            )
    
    console.print(table)
    console.print(f"\n📊 Summary: {successful_images}/{len(results)} images processed, {total_faces} faces detected")
    
    # Extract faces if requested
    if extract_faces and output_dir:
        console.print(f"\n💾 Extracting faces to {output_dir}...", style="blue")
        # TODO: Implement face extraction
        console.print("Face extraction not yet implemented", style="yellow")


def display_face_detection_results(results, total_images: int, total_faces_found: int, total_time: float):
    """Display face detection results summary."""
    
    console.print(f"\n✅ Face detection complete!", style="green")
    console.print(f"📊 Processed {total_images} images")
    console.print(f"👥 Found {total_faces_found} faces")
    console.print(f"⏱️  Total detection time: {total_time:.2f}s")
    console.print(f"📈 Average time per image: {total_time/total_images:.2f}s")
    
    # Show any errors
    error_count = sum(1 for result in results if result.error and not result.error.startswith("Skipped"))
    if error_count > 0:
        console.print(f"⚠️  {error_count} images had errors", style="yellow")
    
    # Show skipped count
    skipped_count = sum(1 for result in results if result.error and result.error.startswith("Skipped"))
    if skipped_count > 0:
        console.print(f"⏭️  {skipped_count} images skipped (sidecar exists)", style="blue")


@face_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              required=True,
              help='Output directory for clustered faces')
@click.option('--threshold', '-t',
              type=float,
              default=0.6,
              help='Clustering threshold (0.0-1.0)')
@click.option('--min-cluster-size', 'min_cluster_size',
              type=int,
              default=2,
              help='Minimum cluster size')
@click.pass_context
def cluster(ctx: click.Context, 
            input_path: Path, 
            output: Path, 
            threshold: float,
            min_cluster_size: int):
    """
    Cluster detected faces by similarity.
    
    INPUT_PATH should be a directory containing images with face detection sidecar files.
    """
    
    core = get_core(ctx)
    
    console.print(f"🔗 Clustering faces from {input_path}...", style="blue")
    
    # TODO: Implement face clustering
    console.print("Face clustering not yet implemented", style="yellow")


@face_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              required=True,
              help='Output directory for extracted faces')
@click.option('--face-size', 'face_size',
              type=int,
              default=64,
              help='Size of extracted faces')
@click.option('--padding', '-p',
              type=int,
              default=10,
              help='Padding around face in pixels')
@click.pass_context
def extract(ctx: click.Context, 
            input_path: Path, 
            output: Path, 
            face_size: int,
            padding: int):
    """
    Extract detected faces from images.
    
    INPUT_PATH should be a directory containing images with face detection sidecar files.
    """
    
    core = get_core(ctx)
    
    console.print(f"✂️  Extracting faces from {input_path}...", style="blue")
    
    # TODO: Implement face extraction
    console.print("Face extraction not yet implemented", style="yellow")
