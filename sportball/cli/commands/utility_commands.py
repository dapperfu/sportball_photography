"""
Utility Commands

CLI commands for utility operations like cache management and system info.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional

# Lazy import: from rich.console import Console

from ..utils import get_core

console = None  # Will be initialized lazily


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
    )

    return Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


def _get_table():
    """Lazy import of Table to avoid heavy imports at startup."""
    from rich.table import Table

    return Table


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def utility_group():
    """Utility commands for cache management and system operations."""
    pass


@utility_group.command()
@click.pass_context
def clear_cache(ctx: click.Context):
    """
    Clear all cached data.
    """

    core = get_core(ctx)

    _get_console().print("🗑️  Clearing cache...", style="blue")

    core.cleanup_cache()

    _get_console().print("✅ Cache cleared successfully", style="green")


@utility_group.command()
@click.pass_context
def system_info(ctx: click.Context):
    """
    Show system information and sportball configuration.
    """

    core = get_core(ctx)

    # System information
    import platform
    import sys
    import os

    Table = _get_table()
    info_table = Table(title="System Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Platform", platform.platform())
    info_table.add_row("Python Version", sys.version.split()[0])
    info_table.add_row("CPU Count", str(os.cpu_count()))
    info_table.add_row("Base Directory", str(core.base_dir))
    info_table.add_row("GPU Enabled", "✅" if core.enable_gpu else "❌")
    info_table.add_row("Cache Enabled", "✅" if core.cache_enabled else "❌")
    info_table.add_row(
        "Max Workers", str(core.max_workers) if core.max_workers else "Auto"
    )

    dependencies = [
        ("PIL", "PIL"),
        ("Click", "click"),
        ("Rich", "rich"),
        ("tqdm", "tqdm"),
        ("fast-exif-rs-py", "fast_exif_rs_py"),
    ]

    for name, module in dependencies:
        try:
            __import__(module)
            info_table.add_row(name, "✅")
        except ImportError:
            info_table.add_row(name, "❌")

    _get_console().print(info_table)


@utility_group.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for converted images",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["jpg", "png", "tiff"]),
    default="jpg",
    help="Output image format",
)
@click.option("--quality", "-q", type=int, default=95, help="JPEG quality (1-100)")
@click.option(
    "--resize", "-r", type=str, help='Resize images (e.g., "1920x1080", "50%")'
)
@click.pass_context
def convert_images(
    ctx: click.Context,
    input_path: Path,
    output: Optional[Path],
    output_format: str,
    quality: int,
    resize: Optional[str],
):
    """
    Convert images to different formats and sizes.

    INPUT_PATH can be a single image file or a directory containing images.
    """

    _get_console().print(f"🔄 Converting images in {input_path}...", style="blue")

    # TODO: Implement image conversion
    _get_console().print("Image conversion not yet implemented", style="yellow")


@utility_group.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for organized images",
)
@click.option(
    "--by",
    "organize_by",
    type=click.Choice(["date", "size", "quality"]),
    default="date",
    help="Organization criteria",
)
@click.option("--copy/--move", default=False, help="Copy files instead of moving them")
@click.pass_context
def organize(
    ctx: click.Context,
    input_path: Path,
    output: Optional[Path],
    organize_by: str,
    copy: bool,
):
    """
    Organize images by various criteria.

    INPUT_PATH should be a directory containing images.
    OUTPUT_DIR is where organized images will be saved.
    """

    _get_console().print(
        f"📁 Organizing images by {organize_by} in {input_path}...", style="blue"
    )

    # TODO: Implement image organization
    _get_console().print("Image organization not yet implemented", style="yellow")
