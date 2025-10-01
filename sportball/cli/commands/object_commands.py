"""
Object Detection Commands

CLI commands for object detection and extraction operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional, List

# Lazy imports to avoid heavy dependencies at startup
# # Lazy import: from rich.console import Console
# # Lazy import: from rich.table import Table
# # Lazy import: from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Lazy imports to avoid heavy dependencies at startup
# from ..utils import get_core, find_image_files, check_sidecar_files
# from ...sidecar import Sidecar, OperationType

def _get_console():
    """Lazy import of Console to avoid heavy imports at startup."""
    from rich.console import Console
    return Console()

def _get_progress():
    """Lazy import of Progress components to avoid heavy imports at startup."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    return Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

def _get_table():
    """Lazy import of Table to avoid heavy imports at startup."""
    from rich.table import Table
    return Table

console = None  # Will be initialized lazily


@click.group(context_settings={'help_option_names': ['-h', '--help']})
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
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
@click.option('--batch-size', 'batch_size',
              type=int,
              default=8,
              help='Processing batch size (legacy parameter, not used in sequential mode)')
@click.option('--force', '-f',
              is_flag=True,
              help='Force detection even if JSON sidecar exists')
@click.option('--verbose', '-v', 
              count=True, 
              help='Enable verbose logging (-v for info, -vv for debug)')
@click.option('--show-empty-results', 'show_empty_results',
              is_flag=True,
              help='Show output even when no objects are detected (default: suppress empty results)')
@click.pass_context
def detect(ctx: click.Context, 
           input_path: Path, 
           output: Optional[Path],
           confidence: float,
           class_names: Optional[str],
           save_sidecar: bool,
           extract_objects: bool,
           no_recursive: bool,
           batch_size: int,
           force: bool,
           verbose: int,
           show_empty_results: bool):
    """
    Detect objects in images using YOLOv8.
    
    INPUT_PATH can be a single image file or a directory containing images.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        _get_console().print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        _get_console().print("ℹ️  Info logging enabled", style="blue")
    
    # Find image files (recursive by default)
    recursive = not no_recursive
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import find_image_files
    image_paths = find_image_files(input_path, recursive=recursive)
    
    if not image_paths:
        _get_console().print("❌ No image files found", style="red")
        return
    
    # Parse class names
    classes = None
    if class_names:
        classes = [name.strip() for name in class_names.split(',')]
    
    _get_console().print(f"📊 Found {len(image_paths)} images to analyze", style="blue")
    
    # Check for existing sidecar files
    _get_console().print("🔍 Checking for existing sidecar files...", style="blue")
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import check_sidecar_files_parallel
    files_to_process, skipped_files = check_sidecar_files_parallel(
        image_paths, 
        force, 
        operation_type="object_detection"
    )
    
    # Show skipping message after image discovery but before processing
    if skipped_files:
        _get_console().print(f"⏭️  Skipping {len(skipped_files)} images - JSON sidecar already exists (use --force to override)", style="yellow")
    
    _get_console().print(f"📊 Processing {len(files_to_process)} images ({len(skipped_files)} skipped)", style="blue")
    
    if not files_to_process:
        _get_console().print("✅ All images already processed (use --force to reprocess)", style="green")
        return
    
    # Show progress for initialization and processing
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _get_progress()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=_get_console(),
        transient=False  # Keep progress visible after completion
    ) as progress:
        
        # Initialize core with progress indicator
        init_task = progress.add_task("🔧 Initializing object detection system...", total=None)
        
        # Lazy import to avoid heavy dependencies at startup
        from ..utils import get_core
        core = get_core(ctx)
        
        progress.update(init_task, description="✅ System initialized")
        progress.remove_task(init_task)
        
        # Prepare detection parameters
        detection_kwargs = {
            'confidence': confidence,
            'classes': classes,
            'save_sidecar': save_sidecar,
            'force': force
        }
        
        # Perform detection with progress indicator
        detect_task = progress.add_task(f"🔍 Detecting objects in {len(files_to_process)} images...", total=len(files_to_process))
        
        results = core.detect_objects(files_to_process, **detection_kwargs)
        
        progress.update(detect_task, description="✅ Detection complete")
        progress.remove_task(detect_task)
    
    # Display results
    display_object_results(results, extract_objects, output, core, files_to_process, not show_empty_results)


def display_object_results(results: dict, extract_objects: bool, output_dir: Optional[Path], core, image_paths: List[Path], suppress_empty: bool = True):
    """
    Display object detection results.
    
    Args:
        results: Dictionary of detection results
        extract_objects: Whether to extract objects
        output_dir: Output directory for extraction
        core: Core instance
        image_paths: List of image paths processed
        suppress_empty: If True, suppress output when no objects are found (default: True)
    """
    
    # Count total objects found first
    total_objects = 0
    successful_images = 0
    
    for image_path, result in results.items():
        if result.get('success', False):
            successful_images += 1
            objects = result.get('objects', [])
            total_objects += len(objects)
    
    # If suppress mode is enabled and no objects were found, suppress output
    if suppress_empty and total_objects == 0:
        return
    
    # Create results table
    table = _get_table()(title="Object Detection Results")
    table.add_column("Image", style="cyan")
    table.add_column("Objects Found", style="green", justify="right")
    table.add_column("Classes", style="yellow")
    table.add_column("Success", style="green")
    table.add_column("Error", style="red")
    
    class_counts = {}
    
    for image_path, result in results.items():
        if result.get('success', False):
            objects = result.get('objects', [])
            object_count = len(objects)
            
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
    
    _get_console().print(table)
    _get_console().print(f"\n📊 Summary: {successful_images}/{len(results)} images processed, {total_objects} objects detected")
    
    # Display class statistics
    if class_counts:
        _get_console().print("\n📈 Object Class Statistics:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            _get_console().print(f"  {class_name}: {count}")
    
    # Extract objects if requested
    if extract_objects and output_dir:
        _get_console().print(f"\n💾 Extracting objects to {output_dir}...", style="blue")
        extraction_results = core.extract_objects(image_paths, output_dir)
        display_extraction_results(extraction_results)


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
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
@click.pass_context
def extract(ctx: click.Context, 
            input_path: Path, 
            output_dir: Path,
            object_types: Optional[str],
            min_size: int,
            max_size: Optional[int],
            padding: int,
            no_recursive: bool):
    """
    Extract detected objects from images.
    
    INPUT_PATH should be a directory containing images with object detection sidecar files.
    OUTPUT_DIR is where extracted objects will be saved.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    # Parse object types
    types = None
    if object_types:
        types = [name.strip() for name in object_types.split(',')]
    
    # Find image files (recursive by default)
    recursive = not no_recursive
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import find_image_files
    image_paths = find_image_files(input_path, recursive=recursive)
    
    if not image_paths:
        _get_console().print("❌ No image files found", style="red")
        return
    
    # Show progress for initialization and processing
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _get_progress()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=_get_console(),
        transient=False  # Keep progress visible after completion
    ) as progress:
        
        # Initialize core with progress indicator
        init_task = progress.add_task("🔧 Initializing object detection system...", total=None)
        
        # Lazy import to avoid heavy dependencies at startup
        from ..utils import get_core
        core = get_core(ctx)
        
        progress.update(init_task, description="✅ System initialized")
        progress.remove_task(init_task)
        
        # Extract objects with progress indicator
        extract_task = progress.add_task(f"✂️  Extracting objects from {len(image_paths)} images...", total=len(image_paths))
        
        extraction_results = core.extract_objects(
            image_paths,
            output_dir,
            object_types=types,
            min_size=min_size,
            max_size=max_size,
            padding=padding
        )
        
        progress.update(extract_task, description="✅ Extraction complete")
        progress.remove_task(extract_task)
    
    # Display extraction results
    display_extraction_results(extraction_results)


@object_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--confidence', '-c',
              type=float,
              default=0.5,
              help='Detection confidence threshold (0.0-1.0)')
@click.option('--save-sidecar/--no-sidecar',
              default=True,
              help='Save results to sidecar files')
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
@click.pass_context
def analyze(ctx: click.Context, 
            input_path: Path, 
            confidence: float,
            save_sidecar: bool,
            no_recursive: bool):
    """
    Analyze objects in images and generate statistics.
    
    INPUT_PATH should be a directory containing images.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    _get_console().print(f"📊 Analyzing objects in {input_path}...", style="blue")
    
    # Find image files (recursive by default)
    recursive = not no_recursive
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import find_image_files
    image_paths = find_image_files(input_path, recursive=recursive)
    
    if not image_paths:
        _get_console().print("❌ No image files found", style="red")
        return
    
    # Show progress for initialization and processing
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _get_progress()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=_get_console(),
        transient=False  # Keep progress visible after completion
    ) as progress:
        
        # Initialize core with progress indicator
        init_task = progress.add_task("🔧 Initializing object detection system...", total=None)
        
        # Lazy import to avoid heavy dependencies at startup
        from ..utils import get_core
        core = get_core(ctx)
        
        progress.update(init_task, description="✅ System initialized")
        progress.remove_task(init_task)
        
        # Perform object detection for analysis
        detection_kwargs = {
            'confidence': confidence,
            'save_sidecar': save_sidecar
        }
        
        # Perform detection with progress indicator
        analyze_task = progress.add_task(f"🔍 Analyzing objects in {len(image_paths)} images...", total=len(image_paths))
        
        results = core.detect_objects(image_paths, **detection_kwargs)
        
        progress.update(analyze_task, description="✅ Analysis complete")
        progress.remove_task(analyze_task)
    
    # Display analysis results
    display_object_analysis(results)


def display_extraction_results(results: dict):
    """Display object extraction results."""
    
    # Create results table
    table = _get_table()(title="Object Extraction Results")
    table.add_column("Image", style="cyan")
    table.add_column("Objects Extracted", style="green", justify="right")
    table.add_column("Output Directory", style="yellow")
    table.add_column("Success", style="green")
    table.add_column("Error", style="red")
    
    total_objects = 0
    successful_extractions = 0
    
    for image_path, result in results.items():
        if result.get('success', False):
            objects_extracted = result.get('objects_extracted', 0)
            total_objects += objects_extracted
            successful_extractions += 1
            
            table.add_row(
                Path(image_path).name,
                str(objects_extracted),
                result.get('output_directory', 'N/A'),
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
    
    _get_console().print(table)
    _get_console().print(f"\n📊 Summary: {successful_extractions}/{len(results)} extractions successful, {total_objects} objects extracted")


def display_object_analysis(results: dict):
    """Display object analysis results."""
    
    # Calculate comprehensive statistics
    total_images = len(results)
    successful_detections = len([r for r in results.values() if r.get('success', False)])
    total_objects = sum(len(r.get('objects', [])) for r in results.values() if r.get('success', False))
    
    # Count objects by class
    class_counts = {}
    confidence_stats = {}
    size_stats = {}
    
    for result in results.values():
        if result.get('success', False):
            objects = result.get('objects', [])
            for obj in objects:
                class_name = obj.get('class_name', 'unknown')
                confidence = obj.get('confidence', 0)
                coords = obj.get('coordinates_pixels', {})
                width = coords.get('width', 0)
                height = coords.get('height', 0)
                area = width * height
                
                # Count classes
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Collect confidence stats
                if class_name not in confidence_stats:
                    confidence_stats[class_name] = []
                confidence_stats[class_name].append(confidence)
                
                # Collect size stats
                if class_name not in size_stats:
                    size_stats[class_name] = []
                size_stats[class_name].append(area)
    
    # Display summary
    _get_console().print(f"\n📊 Object Analysis Summary")
    _get_console().print(f"Total images processed: {total_images}")
    _get_console().print(f"Successful detections: {successful_detections}")
    _get_console().print(f"Total objects detected: {total_objects}")
    _get_console().print(f"Average objects per image: {total_objects/total_images:.2f}" if total_images > 0 else "N/A")
    
    # Display class statistics
    if class_counts:
        _get_console().print(f"\n📈 Object Class Distribution:")
        class_table = _get_table()()
        class_table.add_column("Class", style="cyan")
        class_table.add_column("Count", style="green", justify="right")
        class_table.add_column("Percentage", style="yellow", justify="right")
        class_table.add_column("Avg Confidence", style="blue", justify="right")
        class_table.add_column("Avg Size", style="magenta", justify="right")
        
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            avg_confidence = sum(confidence_stats[class_name]) / len(confidence_stats[class_name]) if class_name in confidence_stats else 0
            avg_size = sum(size_stats[class_name]) / len(size_stats[class_name]) if class_name in size_stats else 0
            
            class_table.add_row(
                class_name,
                str(count),
                f"{percentage:.1f}%",
                f"{avg_confidence:.3f}",
                f"{avg_size:.0f} px²"
            )
        
        _get_console().print(class_table)
    
    # Display confidence distribution
    if confidence_stats:
        _get_console().print(f"\n📊 Confidence Distribution:")
        for class_name, confidences in confidence_stats.items():
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            _get_console().print(f"  {class_name}: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")
