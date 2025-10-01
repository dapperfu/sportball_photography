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
