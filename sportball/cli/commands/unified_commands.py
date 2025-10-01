"""
Unified Sportball Commands

Combined face and object detection commands for efficient processing.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def _get_console():
    """Lazy import of Console to avoid heavy imports at startup."""
    from rich.console import Console
    return Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def unified_group():
    """Unified detection and extraction commands."""
    pass


@unified_group.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for extracted faces/objects",
)
@click.option(
    "--confidence",
    "-c",
    type=float,
    default=0.5,
    help="Detection confidence threshold (0.0-1.0)",
)
@click.option(
    "--class-names",
    type=str,
    help='Comma-separated list of object classes to detect (e.g., "person,sports ball")',
)
@click.option(
    "--border-padding",
    "-b",
    default=0.25,
    help="Border padding percentage for face detection (0.25 = 25%)",
)
@click.option(
    "--save-sidecar/--no-sidecar",
    default=True,
    help="Save results to sidecar files",
)
@click.option(
    "--force", "-f", is_flag=True, help="Force detection even if sidecar exists"
)
@click.option(
    "--no-recursive",
    "no_recursive",
    is_flag=True,
    help="Disable recursive directory processing",
)
@click.option(
    "--workers", "-w", type=int, help="Number of parallel workers (default: auto)"
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Enable verbose logging (-v for info, -vv for debug)",
)
@click.option(
    "--show-empty-results",
    "show_empty_results",
    is_flag=True,
    help="Show output even when no faces/objects are detected (default: suppress empty results)",
)
@click.pass_context
def detect(
    ctx: click.Context,
    input_path: Path,
    output: Optional[Path],
    confidence: float,
    class_names: Optional[str],
    border_padding: float,
    save_sidecar: bool,
    force: bool,
    no_recursive: bool,
    workers: Optional[int],
    verbose: int,
    show_empty_results: bool,
):
    """
    Detect faces and objects in images using unified processing.
    
    This command runs both face detection and object detection on each image
    in a single pass, loading and resizing each image only once for efficiency.
    
    INPUT_PATH can be a single image file or a directory containing images.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:
        _get_console().print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:
        _get_console().print("ℹ️  Info logging enabled", style="blue")

    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core, find_image_files

    core = get_core(ctx)
    console = _get_console()

    # Find image files
    recursive = not no_recursive
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = find_image_files(input_path, recursive=recursive)

    if not image_files:
        console.print("❌ No image files found", style="red")
        return

    console.print(f"📊 Found {len(image_files)} images to process", style="blue")

    # Parse object classes first
    classes = None
    if class_names:
        classes = [cls.strip() for cls in class_names.split(",")]
        console.print(f"🎯 Detecting objects: {', '.join(classes)}", style="blue")

    # Display system information if verbose
    if verbose >= 1:
        console.print("\n🔧 System Information:", style="bold blue")
        
        # Get face detector info
        try:
            face_detector = core.face_detector
            face_model_info = face_detector.get_model_info()
            face_model_name = face_model_info.get("model", "unknown")
            face_device = face_model_info.get("device", "cpu")
            face_gpu_enabled = face_model_info.get("gpu_enabled", False)
            face_gpu_test_passed = face_model_info.get("gpu_test_passed", False)
            
            console.print(f"   Face Detection:")
            console.print(f"     Model: {face_model_name}")
            console.print(f"     Device: {face_device}")
            if face_gpu_enabled and face_gpu_test_passed:
                console.print(f"     GPU: ✅ Enabled and tested", style="green")
            elif face_gpu_enabled:
                console.print(f"     GPU: ⚠️  Enabled but test failed", style="yellow")
            else:
                console.print(f"     GPU: ❌ Disabled", style="red")
        except Exception as e:
            console.print(f"   Face Detection: ❌ Error getting info: {e}", style="red")
        
        # Get object detector info
        try:
            object_detector = core.get_object_detector()
            
            # Try to get model info, fallback to basic info
            try:
                object_model_info = object_detector.get_model_info()
                object_model_name = object_model_info.get("model", "unknown")
                object_device = object_model_info.get("device", "cpu")
                object_gpu_enabled = object_model_info.get("gpu_enabled", False)
                object_gpu_test_passed = object_model_info.get("gpu_test_passed", False)
            except AttributeError:
                # Fallback for detectors without get_model_info
                object_model_name = "YOLOv8"
                object_device = getattr(object_detector, 'device', 'cpu')
                object_gpu_enabled = getattr(object_detector, 'enable_gpu', False)
                object_gpu_test_passed = object_gpu_enabled and object_device.startswith('cuda')
            
            console.print(f"   Object Detection:")
            console.print(f"     Model: {object_model_name}")
            console.print(f"     Device: {object_device}")
            if object_gpu_enabled and object_gpu_test_passed:
                console.print(f"     GPU: ✅ Enabled and tested", style="green")
            elif object_gpu_enabled:
                console.print(f"     GPU: ⚠️  Enabled but test failed", style="yellow")
            else:
                console.print(f"     GPU: ❌ Disabled", style="red")
        except Exception as e:
            console.print(f"   Object Detection: ❌ Error getting info: {e}", style="red")
        
        # Show processing configuration
        console.print(f"   Processing Configuration:")
        console.print(f"     Workers: {workers if workers else 'auto'}")
        console.print(f"     Confidence: {confidence}")
        console.print(f"     Border Padding: {border_padding * 100:.0f}%")
        console.print(f"     Save Sidecar: {'Yes' if save_sidecar else 'No'}")
        console.print(f"     Force Reprocess: {'Yes' if force else 'No'}")
        
        if classes:
            console.print(f"     Target Classes: {', '.join(classes)}")
        
        console.print()  # Add spacing

    # Prepare detection parameters
    detection_kwargs = {
        "confidence": confidence,
        "classes": classes,
        "border_padding": border_padding,
        "save_sidecar": save_sidecar,
        "force": force,
    }

    # Perform unified detection
    if workers and workers > 1:
        console.print(f"🔄 Processing images with {workers} parallel workers...", style="blue")
        results = core.detect_unified(image_files, max_workers=workers, **detection_kwargs)
    else:
        results = core.detect_unified(image_files, **detection_kwargs)

    # Display results
    display_unified_results(results, len(image_files), not show_empty_results)


@unified_group.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--face-padding",
    "-fp",
    type=int,
    default=20,
    help="Padding around faces in pixels (default: 20)",
)
@click.option(
    "--object-padding",
    "-op",
    type=int,
    default=10,
    help="Padding around objects in pixels (default: 10)",
)
@click.option(
    "--object-types",
    "object_types",
    type=str,
    help='Comma-separated list of object types to extract (e.g., "person,sports ball")',
)
@click.option(
    "--min-size", "min_size", type=int, default=32, help="Minimum object size in pixels"
)
@click.option("--max-size", "max_size", type=int, help="Maximum object size in pixels")
@click.option(
    "--no-recursive",
    "no_recursive",
    is_flag=True,
    help="Disable recursive directory processing",
)
@click.option(
    "--workers", "-w", type=int, help="Number of parallel workers (default: auto)"
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Enable verbose logging (-v for info, -vv for debug)",
)
@click.pass_context
def extract(
    ctx: click.Context,
    input_path: Path,
    output_dir: Path,
    face_padding: int,
    object_padding: int,
    object_types: Optional[str],
    min_size: int,
    max_size: Optional[int],
    no_recursive: bool,
    workers: Optional[int],
    verbose: int,
):
    """
    Extract detected faces and objects from images.
    
    INPUT_PATH should be a directory containing images with detection sidecar files.
    OUTPUT_DIR is where extracted faces and objects will be saved.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:
        _get_console().print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:
        _get_console().print("ℹ️  Info logging enabled", style="blue")

    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core, find_image_files

    core = get_core(ctx)
    console = _get_console()

    # Find image files
    recursive = not no_recursive
    image_files = find_image_files(input_path, recursive=recursive)

    if not image_files:
        console.print("❌ No image files found", style="red")
        return

    console.print(f"📊 Found {len(image_files)} images to process", style="blue")

    # Parse object types
    types = None
    if object_types:
        types = [t.strip() for t in object_types.split(",")]
        console.print(f"🎯 Extracting objects: {', '.join(types)}", style="blue")

    # Extract faces and objects
    extraction_results = core.extract_unified(
        image_files,
        output_dir,
        face_padding=face_padding,
        object_padding=object_padding,
        object_types=types,
        min_size=min_size,
        max_size=max_size,
        max_workers=workers,
    )

    # Display extraction results
    display_extraction_results(extraction_results)


def display_unified_results(results: Dict[str, Any], total_images: int, suppress_empty: bool):
    """Display unified detection results."""
    console = _get_console()
    
    # Count results
    face_results = results.get("faces", {})
    object_results = results.get("objects", {})
    
    total_faces = sum(result.get("face_count", 0) for result in face_results.values())
    total_objects = sum(result.get("objects_found", 0) for result in object_results.values())
    
    # Display summary
    console.print(f"\n📊 Detection Summary:", style="bold blue")
    console.print(f"   Images processed: {total_images}")
    console.print(f"   Faces detected: {total_faces}")
    console.print(f"   Objects detected: {total_objects}")
    
    if not suppress_empty:
        # Show detailed results
        console.print(f"\n📋 Detailed Results:", style="bold blue")
        
        for image_path in sorted(face_results.keys()):
            face_result = face_results[image_path]
            object_result = object_results.get(image_path, {})
            
            face_count = face_result.get("face_count", 0)
            object_count = object_result.get("objects_found", 0)
            
            if face_count > 0 or object_count > 0 or not suppress_empty:
                console.print(f"   {Path(image_path).name}: {face_count} faces, {object_count} objects")


def display_extraction_results(results: Dict[str, Any]):
    """Display extraction results."""
    console = _get_console()
    
    faces_extracted = results.get("faces_extracted", 0)
    objects_extracted = results.get("objects_extracted", 0)
    
    console.print(f"\n📊 Extraction Summary:", style="bold blue")
    console.print(f"   Faces extracted: {faces_extracted}")
    console.print(f"   Objects extracted: {objects_extracted}")
    
    if "output_directory" in results:
        console.print(f"   Output directory: {results['output_directory']}")
