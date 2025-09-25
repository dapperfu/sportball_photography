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
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for results')
@click.option('--confidence', '-c',
              type=float,
              default=0.5,
              help='Detection confidence threshold (0.0-1.0)')
@click.option('--min-faces', 'min_faces',
              type=int,
              default=1,
              help='Minimum number of faces to process')
@click.option('--max-faces', 'max_faces',
              type=int,
              help='Maximum number of faces to detect')
@click.option('--face-size', 'face_size',
              type=int,
              default=64,
              help='Minimum face size in pixels')
@click.option('--save-sidecar/--no-sidecar',
              default=True,
              help='Save results to sidecar files')
@click.option('--extract-faces', 'extract_faces',
              is_flag=True,
              help='Extract detected faces to separate images')
@click.pass_context
def detect(ctx: click.Context, 
           input_path: Path, 
           output: Optional[Path],
           confidence: float,
           min_faces: int,
           max_faces: Optional[int],
           face_size: int,
           save_sidecar: bool,
           extract_faces: bool):
    """
    Detect faces in images.
    
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
    
    console.print(f"🔍 Detecting faces in {len(image_paths)} images...", style="blue")
    
    # Initialize sidecar manager for direct usage
    sidecar_manager = Sidecar()
    
    # Perform face detection
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing images...", total=len(image_paths))
        
        results = {}
        for image_path in image_paths:
            try:
                # Check if sidecar already exists
                existing_sidecar = sidecar_manager.find_sidecar_for_image(image_path)
                if existing_sidecar and existing_sidecar.operation == OperationType.FACE_DETECTION:
                    console.print(f"⏭️  Skipping {image_path.name} - sidecar already exists", style="yellow")
                    results[str(image_path)] = existing_sidecar.load()
                    continue
                
                # Perform detection using core
                detection_result = core.detect_faces(
                    [image_path],
                    save_sidecar=False,  # We'll handle sidecar saving manually
                    confidence=confidence,
                    min_faces=min_faces,
                    max_faces=max_faces,
                    face_size=face_size
                )
                
                result = detection_result[str(image_path)]
                
                # Save to sidecar if requested
                if save_sidecar and result.get('success', False):
                    sidecar_info = sidecar_manager.create_sidecar(
                        image_path,
                        OperationType.FACE_DETECTION,
                        result
                    )
                    if sidecar_info:
                        console.print(f"💾 Saved sidecar for {image_path.name}", style="green")
                
                results[str(image_path)] = result
                
            except Exception as e:
                console.print(f"❌ Error processing {image_path.name}: {e}", style="red")
                results[str(image_path)] = {"error": str(e), "success": False}
            
            progress.advance(task)
    
    # Display results
    display_face_results(results, extract_faces, output)


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
