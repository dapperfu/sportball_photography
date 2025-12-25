"""
Shared CLI Utilities

Common utility functions used across multiple command modules to avoid duplication.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from typing import Tuple, Type
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table


def get_console() -> Console:
    """Get a Rich Console instance."""
    return Console()


def get_progress_components() -> Tuple[Type[Progress], Type[SpinnerColumn], Type[TextColumn], Type[BarColumn], Type[TimeElapsedColumn]]:
    """Get Rich Progress components."""
    return Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


def get_table() -> Type[Table]:
    """Get a Rich Table class."""
    return Table


def setup_verbose_logging(verbose: int) -> None:
    """Setup verbose logging based on level."""
    console = get_console()
    
    if verbose >= 2:  # -vv: debug level
        console.print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        console.print("ℹ️  Info logging enabled", style="blue")


def check_and_display_sidecar_status(
    files_to_process: list, 
    skipped_files: list, 
    force: bool, 
    operation_type: str
) -> None:
    """Check sidecar files and display status messages."""
    console = get_console()
    
    # Show skipping message after image discovery but before processing
    if skipped_files:
        console.print(
            f"⏭️  Skipping {len(skipped_files)} images - JSON sidecar already exists (use --force to override)",
            style="yellow",
        )

    console.print(
        f"📊 Processing {len(files_to_process)} images ({len(skipped_files)} skipped)",
        style="blue",
    )

    if not files_to_process:
        console.print(
            "✅ All images already processed (use --force to reprocess)", style="green"
        )
        return


def display_processing_start(image_count: int, workers: int = None) -> None:
    """Display processing start message."""
    console = get_console()
    
    if workers and workers > 1:
        console.print(f"🔄 Processing {image_count} images with {workers} parallel workers...", style="blue")
    else:
        console.print(f"🔄 Processing {image_count} images...", style="blue")


def display_system_info(core, operation_type: str = "detection", verbose: int = 1) -> None:
    """Display comprehensive system information for detection operations."""
    if verbose < 1:
        return
        
    console = get_console()
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
    
    console.print()  # Add spacing
