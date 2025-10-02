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
        console.print(f"     Save Sidecar: {'Yes' if save_sidecar else 'No'}")
        console.print(f"     Force Reprocess: {'Yes' if force else 'No'}")
        
        if classes:
            console.print(f"     Target Classes: {', '.join(classes)}")
        
        console.print()  # Add spacing

    # Prepare detection parameters
    detection_kwargs = {
        "confidence": confidence,
        "classes": classes,
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
    "--faces",
    "extract_faces",
    is_flag=True,
    help="Extract only faces from images",
)
@click.option(
    "--objects",
    "extract_objects", 
    is_flag=True,
    help="Extract only objects from images",
)
@click.option(
    "--both",
    "extract_both",
    is_flag=True,
    help="Extract both faces and objects from images",
)
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
@click.option(
    "--show-empty-results",
    "show_empty_results",
    is_flag=True,
    help="Show output even when no faces/objects are extracted (default: suppress empty results)",
)
@click.pass_context
def extract(
    ctx: click.Context,
    input_path: Path,
    output_dir: Path,
    extract_faces: bool,
    extract_objects: bool,
    extract_both: bool,
    face_padding: int,
    object_padding: int,
    object_types: Optional[str],
    min_size: int,
    max_size: Optional[int],
    no_recursive: bool,
    workers: Optional[int],
    verbose: int,
    show_empty_results: bool,
):
    """
    Extract detected faces and objects from images using unified processing.
    
    This command provides a single interface for extracting faces, objects, or both
    from images with detection sidecar files. Use flags to specify what to extract.
    
    INPUT_PATH should be a directory containing images with detection sidecar files.
    OUTPUT_DIR is where extracted faces and objects will be saved.
    
    Extraction Types:
    - --faces: Extract only faces (saved to <output_dir>/faces/)
    - --objects: Extract only objects (saved to <output_dir>/objects/)  
    - --both: Extract both faces and objects (default if no flag specified)
    
    By default, directories are processed recursively. Use --no-recursive to disable.
    
    Examples:
    
    \b
    # Extract both faces and objects (default behavior)
    sb extract /path/to/images /path/to/output
    
    \b
    # Extract only faces
    sb extract /path/to/images /path/to/output --faces
    
    \b
    # Extract only objects with custom padding
    sb extract /path/to/images /path/to/output --objects --object-padding 20
    
    \b
    # Extract specific object types
    sb extract /path/to/images /path/to/output --objects --object-types "person,sports ball"
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:
        _get_console().print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:
        _get_console().print("ℹ️  Info logging enabled", style="blue")

    # FR-002.5: Validate flag mutual exclusivity
    flags_specified = sum([extract_faces, extract_objects, extract_both])
    if flags_specified > 1:
        console = _get_console()
        console.print("❌ Error: Only one extraction type flag can be specified at a time", style="red")
        console.print("💡 Use --faces, --objects, or --both, but not multiple flags together", style="yellow")
        return

    # FR-002.4: Default behavior is --both when no flag is specified
    if flags_specified == 0:
        extract_both = True
        extraction_type = "both"
    elif extract_faces:
        extraction_type = "faces"
    elif extract_objects:
        extraction_type = "objects"
    elif extract_both:
        extraction_type = "both"

    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core, find_image_files

    core = get_core(ctx)
    console = _get_console()

    # FR-006.2: Validate input path exists and contains images
    if not input_path.exists():
        console.print(f"❌ Input path does not exist: {input_path}", style="red")
        return

    # Find image files
    recursive = not no_recursive
    image_files = find_image_files(input_path, recursive=recursive)

    if not image_files:
        console.print("❌ No image files found", style="red")
        console.print("💡 Make sure the input path contains valid image files (jpg, jpeg, png)", style="yellow")
        return

    console.print(f"📊 Found {len(image_files)} images to process", style="blue")

    # FR-006.3: Validate output directory is writable
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        console.print(f"❌ Cannot write to output directory: {output_dir}", style="red")
        console.print(f"💡 Error: {e}", style="yellow")
        return

    # Parse object types
    types = None
    if object_types:
        types = [t.strip() for t in object_types.split(",")]
        console.print(f"🎯 Extracting objects: {', '.join(types)}", style="blue")

    # Display extraction configuration
    console.print(f"🎯 Extraction type: {extraction_type}", style="blue")
    console.print(f"📁 Output directory: {output_dir}", style="blue")
    
    if extraction_type in ["faces", "both"]:
        console.print(f"👥 Face padding: {face_padding}px", style="blue")
    if extraction_type in ["objects", "both"]:
        console.print(f"🎯 Object padding: {object_padding}px", style="blue")
        if types:
            console.print(f"🎯 Object types: {', '.join(types)}", style="blue")
        console.print(f"📏 Object size range: {min_size}px - {max_size or 'unlimited'}px", style="blue")

    # Perform extraction based on type
    try:
        if extraction_type == "faces":
            # FR-002.1: Extract only faces
            extraction_results = core.extract_faces(
                image_files,
                output_dir / "faces",
                padding=face_padding,
                max_workers=workers,
            )
            # Format results for display
            formatted_results = {
                "faces_extracted": sum(r.get("faces_extracted", 0) for r in extraction_results.values() if r.get("success", False)),
                "objects_extracted": 0,
                "output_directory": str(output_dir / "faces"),
            }
            
        elif extraction_type == "objects":
            # FR-002.2: Extract only objects
            extraction_results = core.extract_objects(
                image_files,
                output_dir / "objects",
                object_types=types,
                min_size=min_size,
                max_size=max_size,
                padding=object_padding,
                max_workers=workers,
            )
            # Format results for display
            formatted_results = {
                "faces_extracted": 0,
                "objects_extracted": sum(r.get("objects_extracted", 0) for r in extraction_results.values() if r.get("success", False)),
                "output_directory": str(output_dir / "objects"),
            }
            
        else:  # extraction_type == "both"
            # FR-002.3: Extract both faces and objects
            # Create subdirectories
            faces_dir = output_dir / "faces"
            objects_dir = output_dir / "objects"
            
            # Extract faces
            face_results = core.extract_faces(
                image_files,
                faces_dir,
                padding=face_padding,
                max_workers=workers,
            )
            
            # Extract objects
            object_results = core.extract_objects(
                image_files,
                objects_dir,
                object_types=types,
                min_size=min_size,
                max_size=max_size,
                padding=object_padding,
                max_workers=workers,
            )
            
            # Format results for display
            formatted_results = {
                "faces_extracted": sum(r.get("faces_extracted", 0) for r in face_results.values() if r.get("success", False)),
                "objects_extracted": sum(r.get("objects_extracted", 0) for r in object_results.values() if r.get("success", False)),
                "output_directory": str(output_dir),
            }

        # Display extraction results
        display_unified_extraction_results(formatted_results, not show_empty_results)

    except Exception as e:
        console.print(f"❌ Extraction failed: {e}", style="red")
        if verbose >= 1:
            import traceback
            console.print(traceback.format_exc(), style="red")
        return


def display_unified_results(results: Dict[str, Any], total_images: int, suppress_empty: bool):
    """Display unified detection results."""
    console = _get_console()
    
    # Count results accurately
    face_results = results.get("faces", {})
    object_results = results.get("objects", {})
    
    # FR-004.1-004.2: Count actual results from detection
    total_faces = 0
    total_objects = 0
    successful_images = 0
    
    for image_path in face_results.keys():
        face_result = face_results[image_path]
        object_result = object_results.get(image_path, {})
        
        # Count faces from actual detection results
        if face_result.get("success", False):
            total_faces += face_result.get("face_count", 0)
        
        # Count objects from actual detection results  
        if object_result.get("success", False):
            total_objects += object_result.get("objects_found", 0)
        
        # Count successful images
        if face_result.get("success", False) or object_result.get("success", False):
            successful_images += 1
    
    # FR-004.3: Display correct processing statistics
    console.print(f"\n✅ Unified detection complete!", style="green")
    console.print(f"📊 Processed {total_images} images")
    console.print(f"👥 Faces detected: {total_faces}")
    console.print(f"🎯 Objects detected: {total_objects}")
    
    if successful_images > 0:
        console.print(f"✅ {successful_images} images processed successfully", style="green")
    
    # FR-004.4: Show detailed results per image when verbose mode is enabled
    if not suppress_empty:
        console.print(f"\n📋 Detailed Results:", style="bold blue")
        
        for image_path in sorted(face_results.keys()):
            face_result = face_results[image_path]
            object_result = object_results.get(image_path, {})
            
            face_count = face_result.get("face_count", 0)
            object_count = object_result.get("objects_found", 0)
            
            if face_count > 0 or object_count > 0 or not suppress_empty:
                console.print(f"   {Path(image_path).name}: {face_count} faces, {object_count} objects")


def display_unified_extraction_results(results: Dict[str, Any], suppress_empty: bool):
    """Display unified extraction results."""
    console = _get_console()
    
    faces_extracted = results.get("faces_extracted", 0)
    objects_extracted = results.get("objects_extracted", 0)
    
    # FR-004.5: Suppress empty results if requested
    if suppress_empty and faces_extracted == 0 and objects_extracted == 0:
        return
    
    console.print(f"\n✅ Unified extraction complete!", style="green")
    console.print(f"📊 Extraction Summary:", style="bold blue")
    
    if faces_extracted > 0:
        console.print(f"   👥 Faces extracted: {faces_extracted}")
    if objects_extracted > 0:
        console.print(f"   🎯 Objects extracted: {objects_extracted}")
    
    if "output_directory" in results:
        console.print(f"   📁 Output directory: {results['output_directory']}")


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
