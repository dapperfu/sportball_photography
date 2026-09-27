"""
Shared CLI Utilities

Common utility functions used across multiple command modules to avoid duplication.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from typing import Optional, Tuple, Type

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


def get_console() -> Console:
    """Get a Rich Console instance."""
    return Console()


def get_progress_components() -> Tuple[
    Type[Progress],
    Type[SpinnerColumn],
    Type[TextColumn],
    Type[BarColumn],
    Type[TimeElapsedColumn],
]:
    """Get Rich Progress components."""
    return Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


def get_table() -> Type[Table]:
    """Get a Rich Table class."""
    return Table


def setup_verbose_logging(verbose: int) -> None:
    """Setup verbose logging based on level."""
    console = get_console()

    if verbose >= 2:
        console.print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:
        console.print("ℹ️  Info logging enabled", style="blue")


def display_processing_start(image_count: int, workers: Optional[int] = None) -> None:
    """Display processing start message."""
    console = get_console()

    if workers and workers > 1:
        console.print(
            f"🔄 Processing {image_count} images with {workers} parallel workers...",
            style="blue",
        )
    else:
        console.print(f"🔄 Processing {image_count} images...", style="blue")


def display_system_info(core, operation_type: str = "detection", verbose: int = 1) -> None:
    """Display core configuration when verbose output is on."""
    if verbose < 1:
        return

    console = get_console()
    console.print("\n🔧 System Information:", style="bold blue")
    console.print(f"   Operation: {operation_type}")
    console.print(f"   Base directory: {core.base_dir}")
    console.print(f"   Cache enabled: {core.cache_enabled}")
    console.print(
        f"   Max workers: {core.max_workers if core.max_workers else 'Auto'}"
    )
    console.print()
