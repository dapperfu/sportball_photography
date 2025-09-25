"""
Face Detection Commands

CLI commands for face detection and recognition operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
import json
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..utils import get_core, find_image_files
from ...sidecar import Sidecar, OperationType

console = Console()


def check_sidecar_file(image_file: Path, force: bool) -> Tuple[Path, bool]:
    """
    Check if a sidecar file exists and contains face detection data.
    
    Args:
        image_file: Path to the image file
        force: Whether to force processing even if sidecar exists
        
    Returns:
        Tuple of (image_file, should_skip)
    """
    try:
        # Resolve symlink if needed
        original_image_path = image_file.resolve() if image_file.is_symlink() else image_file
        json_path = original_image_path.parent / f"{original_image_path.stem}.json"
        
        if json_path.exists() and not force:
            # Check if JSON contains face detection data
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if ("Face_detector" in data and 
                    "metadata" in data["Face_detector"] and
                    "extraction_timestamp" in data["Face_detector"]["metadata"]):
                    return (image_file, True)  # Should skip (already processed)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        
        return (image_file, False)  # Should process
        
    except Exception:
        return (image_file, False)  # Should process on error


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
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
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
           no_recursive: bool,
           verbose: int):
    """
    Detect faces in images and save comprehensive data to JSON sidecar files.
    
    INPUT_PATTERN can be a file pattern, directory path, or single image file.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        console.print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        console.print("ℹ️  Info logging enabled", style="blue")
    
    core = get_core(ctx)
    
    # Initialize face detector with same parameters as original
    import sys
    import os
    # Add parent directory to path to import local face_detection module
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from face_detection import FaceDetector
    detector = FaceDetector(
        border_padding=border_padding,
        use_gpu=gpu
    )
    
    console.print(f"🔍 Starting face detection with {border_padding*100:.0f}% border padding", style="blue")
    
    # Pre-scan phase: find all images and check for existing sidecars
    console.print("📁 Scanning directory for images and existing sidecar files...", style="blue")
    
    # Find all image files (recursive by default)
    input_path = Path(input_pattern)
    recursive = not no_recursive
    
    if input_path.is_dir():
        image_files = find_image_files(input_path, recursive=recursive)
    else:
        # Pattern matching
        if input_pattern.startswith('/'):
            parent_dir = Path(input_pattern).parent
            pattern = Path(input_pattern).name
            image_files = list(parent_dir.glob(pattern))
        else:
            image_files = list(Path('.').glob(input_pattern))
    
    # Debug: show what we found
    if verbose >= 1:
        console.print(f"🔍 Debug: Found {len(image_files)} files", style="blue")
        if image_files:
            console.print(f"🔍 Debug: First few files: {[f.name for f in image_files[:3]]}", style="blue")
    
    if not image_files:
        console.print("❌ No images found", style="red")
        return
    
    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]
    
    console.print(f"📊 Found {len(image_files)} images to analyze", style="blue")
    
    # Parallel check for existing sidecar files
    console.print("🔍 Checking for existing sidecar files...", style="blue")
    
    skipped_files = []
    files_to_process = []
    
    # Use parallel processing for sidecar file checking
    max_workers = min(32, len(image_files))  # Limit workers to avoid overwhelming the system
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all sidecar checks
        future_to_file = {
            executor.submit(check_sidecar_file, image_file, force): image_file 
            for image_file in image_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            image_file, should_skip = future.result()
            if should_skip:
                skipped_files.append(image_file)
            else:
                files_to_process.append(image_file)
    
    # Display pre-scan results
    if skipped_files:
        console.print(f"⏭️  Skipped {len(skipped_files)} images with existing face detection data", style="yellow")
        console.print(f"💡 Use --force to reprocess all images", style="blue")
    
    if not files_to_process:
        console.print("✅ All images already have face detection data", style="green")
        return
    
    console.print(f"🔄 Processing {len(files_to_process)} images...", style="blue")
    
    # Process only the files that need processing
    results = []
    total_faces_found = 0
    total_time = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Detecting faces...", total=len(files_to_process))
        
        for image_file in files_to_process:
            try:
                result = detector.detect_faces_in_image(image_file, force)
                results.append(result)
                total_faces_found += result.faces_found
                total_time += result.detection_time
                
                progress.update(task, advance=1, 
                               description=f"Detecting faces... ({result.faces_found} faces in {Path(image_file).name})")
                
            except Exception as e:
                console.print(f"❌ Error processing {image_file.name}: {e}", style="red")
                progress.update(task, advance=1)
    
    # Display final results
    display_face_detection_results(results, len(files_to_process), total_faces_found, total_time, len(skipped_files))


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


def display_face_detection_results(results, total_images: int, total_faces_found: int, total_time: float, skipped_count: int = 0):
    """Display face detection results summary."""
    
    console.print(f"\n✅ Face detection complete!", style="green")
    console.print(f"📊 Processed {total_images} images")
    console.print(f"👥 Found {total_faces_found} faces")
    console.print(f"⏱️  Total detection time: {total_time:.2f}s")
    if total_images > 0:
        console.print(f"📈 Average time per image: {total_time/total_images:.2f}s")
    
    # Show skipped count
    if skipped_count > 0:
        console.print(f"⏭️  {skipped_count} images skipped (existing sidecar data)", style="blue")
    
    # Show any errors
    error_count = sum(1 for result in results if result.error and not result.error.startswith("Skipped"))
    if error_count > 0:
        console.print(f"⚠️  {error_count} images had errors", style="yellow")


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
