"""
Object Detection Commands

CLI commands for object detection and extraction operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..utils import get_core
from ...sidecar import Sidecar, OperationType

console = Console()


@click.group()
def object_group():
    """Object detection and extraction commands."""
    pass


@object_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for results')
@click.option('--confidence', '-c',
              type=float,
              default=0.5,
              help='Detection confidence threshold (0.0-1.0)')
@click.option('--classes', 'class_names',
              type=str,
              help='Comma-separated list of class names to detect (e.g., "person,sports ball")')
@click.option('--save-sidecar/--no-sidecar',
              default=True,
              help='Save results to sidecar files')
@click.option('--extract-objects', 'extract_objects',
              is_flag=True,
              help='Extract detected objects to separate images')
@click.pass_context
def detect(ctx: click.Context, 
           input_path: Path, 
           output: Optional[Path],
           confidence: float,
           class_names: Optional[str],
           save_sidecar: bool,
           extract_objects: bool):
    """
    Detect objects in images using YOLOv8.
    
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
    
    # Parse class names
    classes = None
    if class_names:
        classes = [name.strip() for name in class_names.split(',')]
    
    console.print(f"🔍 Detecting objects in {len(image_paths)} images...", style="blue")
    
    # Initialize sidecar manager for direct usage
    sidecar_manager = Sidecar()
    
    # Perform object detection
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
                if existing_sidecar and existing_sidecar.operation == OperationType.OBJECT_DETECTION:
                    console.print(f"⏭️  Skipping {image_path.name} - sidecar already exists", style="yellow")
                    results[str(image_path)] = existing_sidecar.load()
                    continue
                
                # Perform detection using core
                detection_result = core.detect_objects(
                    [image_path],
                    save_sidecar=False,  # We'll handle sidecar saving manually
                    confidence=confidence,
                    classes=classes
                )
                
                result = detection_result[str(image_path)]
                
                # Save to sidecar if requested
                if save_sidecar and result.get('success', False):
                    sidecar_info = sidecar_manager.create_sidecar(
                        image_path,
                        OperationType.OBJECT_DETECTION,
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
    display_object_results(results, extract_objects, output)


def display_object_results(results: dict, extract_objects: bool, output_dir: Optional[Path]):
    """Display object detection results."""
    
    # Create results table
    table = Table(title="Object Detection Results")
    table.add_column("Image", style="cyan")
    table.add_column("Objects Found", style="green", justify="right")
    table.add_column("Classes", style="yellow")
    table.add_column("Success", style="green")
    table.add_column("Error", style="red")
    
    total_objects = 0
    successful_images = 0
    class_counts = {}
    
    for image_path, result in results.items():
        if result.get('success', False):
            objects = result.get('objects', [])
            object_count = len(objects)
            total_objects += object_count
            successful_images += 1
            
            # Count classes
            classes_found = set()
            for obj in objects:
                class_name = obj.get('class_name', 'unknown')
                classes_found.add(class_name)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            table.add_row(
                Path(image_path).name,
                str(object_count),
                ', '.join(sorted(classes_found)),
                "✅",
                ""
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            table.add_row(
                Path(image_path).name,
                "0",
                "",
                "❌",
                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
            )
    
    console.print(table)
    console.print(f"\n📊 Summary: {successful_images}/{len(results)} images processed, {total_objects} objects detected")
    
    # Display class statistics
    if class_counts:
        console.print("\n📈 Object Class Statistics:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {class_name}: {count}")
    
    # Extract objects if requested
    if extract_objects and output_dir:
        console.print(f"\n💾 Extracting objects to {output_dir}...", style="blue")
        # TODO: Implement object extraction
        console.print("Object extraction not yet implemented", style="yellow")


@object_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--object-types', 'object_types',
              type=str,
              help='Comma-separated list of object types to extract (e.g., "person,sports ball")')
@click.option('--min-size', 'min_size',
              type=int,
              default=32,
              help='Minimum object size in pixels')
@click.option('--max-size', 'max_size',
              type=int,
              help='Maximum object size in pixels')
@click.option('--padding', '-p',
              type=int,
              default=10,
              help='Padding around objects in pixels')
@click.pass_context
def extract(ctx: click.Context, 
            input_path: Path, 
            output_dir: Path,
            object_types: Optional[str],
            min_size: int,
            max_size: Optional[int],
            padding: int):
    """
    Extract detected objects from images.
    
    INPUT_PATH should be a directory containing images with object detection sidecar files.
    OUTPUT_DIR is where extracted objects will be saved.
    """
    
    core = get_core(ctx)
    
    # Parse object types
    types = None
    if object_types:
        types = [name.strip() for name in object_types.split(',')]
    
    console.print(f"✂️  Extracting objects from {input_path} to {output_dir}...", style="blue")
    
    # TODO: Implement object extraction
    console.print("Object extraction not yet implemented", style="yellow")


@object_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
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
            confidence: float,
            save_sidecar: bool):
    """
    Analyze objects in images and generate statistics.
    
    INPUT_PATH should be a directory containing images.
    """
    
    core = get_core(ctx)
    
    console.print(f"📊 Analyzing objects in {input_path}...", style="blue")
    
    # TODO: Implement object analysis
    console.print("Object analysis not yet implemented", style="yellow")
