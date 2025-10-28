"""
Annotation Commands

CLI commands for annotating images with detection results.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, Any, Tuple


# Lazy imports to avoid heavy dependencies at startup
def _get_console():
    """Lazy import of Console to avoid heavy imports at startup."""
    from rich.console import Console

    return Console()


def _get_progress():
    """Lazy import of Progress components to avoid heavy imports at startup."""
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    )

    return (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    )


def _get_table():
    """Lazy import of Table to avoid heavy imports at startup."""
    from rich.table import Table

    return Table


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def viz_group():
    """Visualization and annotation commands."""
    pass


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--size",
    "-s",
    type=int,
    default=1080,
    help="Maximum dimension for output image (default: 1080)",
)
@click.option("--no-faces", "no_faces", is_flag=True, help="Skip face annotations")
@click.option(
    "--no-objects", "no_objects", is_flag=True, help="Skip object annotations"
)
@click.option(
    "--face-color",
    type=str,
    default="blue",
    help="Color for face annotations (blue, green, red, yellow, etc.)",
)
@click.option(
    "--object-color",
    type=str,
    default="green",
    help="Color for object annotations (blue, green, red, yellow, etc.)",
)
@click.option(
    "--font-scale",
    type=float,
    default=1.0,
    help="Font scale for annotations (default: 1.0)",
)
@click.option(
    "--thickness",
    type=int,
    default=2,
    help="Line thickness for annotations (default: 2)",
)
@click.option(
    "--show-confidence/--no-confidence",
    default=True,
    help="Show confidence scores in annotations",
)
@click.option(
    "--no-recursive",
    "no_recursive",
    is_flag=True,
    help="Disable recursive directory processing",
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
    help="Show output even when no annotations are added (default: suppress empty results)",
)
@click.pass_context
def annotate(
    ctx: click.Context,
    input_path: Path,
    output_path: Path,
    size: int,
    no_faces: bool,
    no_objects: bool,
    face_color: str,
    object_color: str,
    font_scale: float,
    thickness: int,
    show_confidence: bool,
    no_recursive: bool,
    verbose: int,
    show_empty_results: bool,
):
    """
    Annotate images with face and object detection results.

    INPUT_PATH can be a single image file or a directory containing images.
    OUTPUT_PATH is where annotated images will be saved.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """

    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        _get_console().print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        _get_console().print("ℹ️  Info logging enabled", style="blue")

    # Find image files (recursive by default)
    recursive = not no_recursive
    from ..utils import find_image_files

    image_paths = find_image_files(input_path, recursive=recursive)

    if not image_paths:
        _get_console().print("❌ No image files found", style="red")
        return

    # Parse colors
    face_color_rgb = _parse_color(face_color)
    object_color_rgb = _parse_color(object_color)

    _get_console().print(
        f"📊 Found {len(image_paths)} images to annotate", style="blue"
    )
    _get_console().print(f"📏 Output size: {size}px (max dimension)", style="blue")
    _get_console().print(
        f"🎨 Face color: {face_color} (RGB: {face_color_rgb})", style="blue"
    )
    _get_console().print(
        f"🎨 Object color: {object_color} (RGB: {object_color_rgb})", style="blue"
    )

    # Show progress for initialization and processing
    (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    ) = _get_progress()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_get_console(),
        transient=False,  # Keep progress visible after completion
    ) as progress:
        # Initialize core with progress indicator
        init_task = progress.add_task(
            "🔧 Initializing annotation system...", total=None
        )

        from ..utils import get_core

        core = get_core(ctx)

        progress.update(init_task, description="✅ System initialized")
        progress.remove_task(init_task)

        # Annotate images with progress indicator
        annotate_task = progress.add_task(
            f"🎨 Annotating {len(image_paths)} images...", total=len(image_paths)
        )

        results = {}
        errors = []

        for i, image_path in enumerate(image_paths):
            try:
                result = _annotate_single_image(
                    image_path,
                    output_path,
                    core,
                    size=size,
                    no_faces=no_faces,
                    no_objects=no_objects,
                    face_color=face_color_rgb,
                    object_color=object_color_rgb,
                    font_scale=font_scale,
                    thickness=thickness,
                    show_confidence=show_confidence,
                )
                results[str(image_path)] = result

                progress.update(annotate_task, advance=1)

            except Exception as e:
                errors.append(f"❌ {image_path.name}: {e}")
                results[str(image_path)] = {"success": False, "error": str(e)}
                progress.update(annotate_task, advance=1)

        progress.update(annotate_task, description="✅ Annotation complete")
        progress.remove_task(annotate_task)

    # Display errors if any
    if errors:
        _get_console().print(
            f"\n⚠️  Errors encountered ({len(errors)} images):", style="yellow"
        )
        for error in errors[:10]:  # Show first 10 errors
            _get_console().print(f"  {error}", style="red")
        if len(errors) > 10:
            _get_console().print(
                f"  ... and {len(errors) - 10} more errors", style="red"
            )

    # Display results
    display_annotation_results(results, not show_empty_results)


def _annotate_single_image(
    image_path: Path,
    output_path: Path,
    core,
    size: int,
    no_faces: bool,
    no_objects: bool,
    face_color: Tuple[int, int, int],
    object_color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
    show_confidence: bool,
) -> Dict[str, Any]:
    """Annotate a single image with detection results."""

    annotations_added = 0

    # Check for sidecar files and available data first
    face_data = None
    object_data = None
    has_faces = False
    has_objects = False

    # Load face data if faces are enabled
    if not no_faces:
        face_data = core.sidecar.load_data(image_path, "face_detection")
        if face_data and "face_detection" in face_data:
            face_detection_data = face_data["face_detection"]
            if face_detection_data.get("success", False):
                faces = face_detection_data.get("faces", [])
                has_faces = len(faces) > 0

    # Load object data if objects are enabled
    if not no_objects:
        object_data = core.sidecar.load_data(image_path, "yolov8")
        if (
            object_data
            and "yolov8" in object_data
            and object_data["yolov8"].get("success", False)
        ):
            objects = object_data["yolov8"].get("objects", [])
            has_objects = len(objects) > 0

    # Skip image if no annotations are available for the enabled detection types
    if no_faces and not has_objects:
        return {"success": False, "error": "No objects found (skipped)"}
    if no_objects and not has_faces:
        return {"success": False, "error": "No faces found (skipped)"}

    # If we reach here, we have at least one type of annotation to process
    # Load the original image only if we're going to process it
    # Apply EXIF rotation to ensure correct orientation
    try:
        from ..utils import load_image_with_exif_rotation
        image = load_image_with_exif_rotation(image_path)
    except Exception as e:
        return {"success": False, "error": f"Could not load image: {e}"}

    original_width, original_height = image.size

    # Calculate resize dimensions while maintaining aspect ratio
    if original_width > original_height:
        new_width = size
        new_height = int((size * original_height) / original_width)
    else:
        new_height = size
        new_width = int((size * original_width) / original_height)

    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    annotated_image = resized_image.copy()

    # Calculate scale factors for coordinate conversion
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Load and annotate faces
    if not no_faces and has_faces and face_data and "face_detection" in face_data:
        face_detection_data = face_data["face_detection"]
        faces = face_detection_data.get("faces", [])
        for i, face in enumerate(faces):
            bbox = face.get("bbox", {})
            if bbox:
                # Convert normalized coordinates to pixel coordinates
                x = int(bbox.get("x", 0) * original_width)
                y = int(bbox.get("y", 0) * original_height)
                width = int(bbox.get("width", 0) * original_width)
                height = int(bbox.get("height", 0) * original_height)

                # Scale to output size
                x = int(x * scale_x)
                y = int(y * scale_y)
                width = int(width * scale_x)
                height = int(height * scale_y)

                confidence = face.get("confidence", 0.0)
                label = f"Face {i + 1}"
                if show_confidence:
                    label += f" ({confidence:.2f})"

                annotated_image = _draw_annotation(
                    annotated_image,
                    x,
                    y,
                    width,
                    height,
                    label,
                    face_color,
                    font_scale,
                    thickness,
                )
                annotations_added += 1

    # Load and annotate objects
    if not no_objects and has_objects and object_data:
        objects = object_data["yolov8"].get("objects", [])
        for i, obj in enumerate(objects):
            coords = obj.get("coordinates_pixels", {})
            if coords:
                x = int(coords.get("x", 0) * scale_x)
                y = int(coords.get("y", 0) * scale_y)
                width = int(coords.get("width", 0) * scale_x)
                height = int(coords.get("height", 0) * scale_y)

                class_name = obj.get("class_name", "object")
                confidence = obj.get("confidence", 0.0)
                label = f"{class_name.title()} {i + 1}"
                if show_confidence:
                    label += f" ({confidence:.2f})"

                annotated_image = _draw_annotation(
                    annotated_image,
                    x,
                    y,
                    width,
                    height,
                    label,
                    object_color,
                    font_scale,
                    thickness,
                )
                annotations_added += 1

    # Determine output path
    if output_path.is_dir() or (not output_path.exists() and not output_path.suffix):
        # If it's a directory or a path without extension (treat as directory)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{image_path.stem}_annotated.jpg"
    else:
        # Treat as a single file path
        output_file = output_path
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save annotated image first
    try:
        annotated_image.save(str(output_file), "JPEG", quality=95)
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = f"Failed to save annotated image: {e}"

    # Copy EXIF data from original image to annotated image using fast-exif-rs
    if success:
        exif_copied = _copy_exif_data_with_fast_exif_rs(image_path, output_file)
        if not exif_copied:
            # EXIF copying failed, but the annotation was successful
            # This is not a critical failure, so we continue
            pass

    if success:
        return {
            "success": True,
            "output_path": str(output_file),
            "annotations_added": annotations_added,
            "original_size": (original_width, original_height),
            "output_size": (new_width, new_height),
        }
    else:
        return {"success": False, "error": error_msg}


def _copy_exif_data_with_fast_exif_rs(
    original_image_path: Path, annotated_image_path: Path
) -> bool:
    """
    Copy EXIF data from original image to annotated image using fast-exif-rs.

    Args:
        original_image_path: Path to the original image with EXIF data
        annotated_image_path: Path to the annotated image to receive EXIF data

    Returns:
        True if EXIF copying was successful, False otherwise
    """
    try:
        import fast_exif_rs_py

        # Create a temporary file for the EXIF-copied image
        temp_path = annotated_image_path.with_suffix(
            ".tmp" + annotated_image_path.suffix
        )

        # Use fast-exif-rs to copy all EXIF data
        copier = fast_exif_rs_py.PyFastExifCopier()
        copier.copy_all_exif(
            str(original_image_path), str(annotated_image_path), str(temp_path)
        )

        # Replace the original annotated image with the EXIF-copied version
        temp_path.replace(annotated_image_path)

        return True

    except Exception:
        # If EXIF copying fails, the original annotated image remains unchanged
        # This ensures the annotation process doesn't fail due to EXIF issues
        return False


def _draw_annotation(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> Image.Image:
    """Draw annotation on image using PIL."""
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Calculate font size based on font_scale
    try:
        # Try to load a default font
        font_size = max(12, int(12 * font_scale))
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    except (OSError, IOError):
        # Fallback to default font
        font = ImageFont.load_default()

    # Draw rectangle
    draw.rectangle([x, y, x + width, y + height], outline=color, width=thickness)

    # Get text size
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position label above the bounding box
    label_x = x
    label_y = y - text_height - 5 if y - text_height - 5 > 0 else y + height + 5

    # Draw label background rectangle
    draw.rectangle(
        [label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
        fill=color,
    )

    # Draw label text
    draw.text((label_x + 2, label_y + 2), label, fill=(255, 255, 255), font=font)

    return annotated_image


def _parse_color(color_name: str) -> Tuple[int, int, int]:
    """Parse color name to RGB tuple."""
    color_map = {
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (0, 165, 255),
        "purple": (128, 0, 128),
        "pink": (203, 192, 255),
        "gray": (128, 128, 128),
        "brown": (42, 42, 165),
        "lime": (0, 255, 0),
        "navy": (128, 0, 0),
        "teal": (128, 128, 0),
        "olive": (0, 128, 128),
        "maroon": (0, 0, 128),
        "silver": (192, 192, 192),
        "gold": (0, 215, 255),
    }

    return color_map.get(color_name.lower(), (255, 255, 255))  # Default to white


def display_annotation_results(
    results: Dict[str, Dict[str, Any]], suppress_empty: bool = True
):
    """
    Display annotation results.

    Args:
        results: Dictionary of annotation results
        suppress_empty: If True, suppress output when no annotations are added (default: True)
    """

    # Count total annotations first
    total_annotations = 0
    successful_images = 0

    for image_path, result in results.items():
        if result.get("success", False):
            successful_images += 1
            annotations = result.get("annotations_added", 0)
            total_annotations += annotations

    # If suppress mode is enabled and no annotations were added, suppress output
    if suppress_empty and total_annotations == 0:
        return

    # Create results table
    table = _get_table()(title="Annotation Results")
    table.add_column("Image", style="cyan")
    table.add_column("Annotations", style="green", justify="right")
    table.add_column("Output Size", style="yellow")
    table.add_column("Success", style="green")
    table.add_column("Error", style="red")

    for image_path, result in results.items():
        if result.get("success", False):
            annotations = result.get("annotations_added", 0)

            output_size = result.get("output_size", (0, 0))
            size_str = f"{output_size[0]}x{output_size[1]}"

            table.add_row(Path(image_path).name, str(annotations), size_str, "✅", "")
        else:
            error_msg = result.get("error", "Unknown error")
            table.add_row(
                Path(image_path).name,
                "0",
                "",
                "❌",
                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg,
            )

    _get_console().print(table)
    _get_console().print(
        f"\n📊 Summary: {successful_images}/{len(results)} images annotated, {total_annotations} annotations added"
    )
