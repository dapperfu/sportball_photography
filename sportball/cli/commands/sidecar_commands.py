"""
Sidecar Management Commands

CLI commands for sidecar file operations and statistics.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional, Dict, Any

# Lazy import: from rich.console import Console
# Lazy import: from rich.table import Table
# Lazy import: from rich.panel import _get_panel()
# Lazy import: from rich.progress import _get_progress()
import json

# Lazy imports to avoid heavy dependencies at startup
# from ..utils import get_core
# from ...sidecar import Sidecar, OperationType

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


def _get_panel():
    """Lazy import of Panel to avoid heavy imports at startup."""
    from rich.panel import Panel

    return Panel


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def sidecar_group():
    """Sidecar file management and statistics commands."""
    pass


@sidecar_group.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--operation",
    "-o",
    type=click.Choice(
        [
            "face_detection",
            "object_detection",
            "ball_detection",
            "quality_assessment",
            "game_detection",
        ]
    ),
    help="Filter by specific operation type",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "summary"]),
    default="table",
    help="Output format",
)
@click.option(
    "--save-report",
    "save_report",
    type=click.Path(path_type=Path),
    help="Save detailed report to file",
)
@click.pass_context
def stats(
    ctx: click.Context,
    directory: Path,
    operation: Optional[str],
    output_format: str,
    save_report: Optional[Path],
):
    """
    Generate comprehensive statistics about sidecar files in a directory.

    DIRECTORY should contain images with sidecar files.
    """

    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core

    core = get_core(ctx)

    _get_console().print(f"📊 Analyzing sidecar files in {directory}...", style="blue")

    # Use tqdm for directory scanning progress
    from tqdm import tqdm

    # First count files to scan
    from ..utils import find_image_files

    image_files = find_image_files(directory, recursive=True)
    total_files = len(image_files)

    with tqdm(total=total_files, desc="Scanning directory", unit="files") as pbar:
        stats_data = collect_sidecar_statistics(directory, operation, pbar)

    # Display results based on format
    if output_format == "json":
        display_json_stats(stats_data)
    elif output_format == "summary":
        display_summary_stats(stats_data)
    else:
        display_table_stats(stats_data)

    # Save report if requested
    if save_report:
        save_statistics_report(stats_data, save_report)


def collect_sidecar_statistics(
    directory: Path, operation_filter: Optional[str], pbar
) -> Dict[str, Any]:
    """Collect comprehensive sidecar statistics with progress tracking."""

    # Lazy import to avoid heavy dependencies at startup
    from ...sidecar import Sidecar, OperationType
    from ..utils import find_image_files

    # Convert operation filter to OperationType enum
    operation_type_filter = None
    if operation_filter:
        try:
            operation_type_filter = OperationType(operation_filter)
        except ValueError:
            _get_console().print(
                f"⚠️  Invalid operation filter: {operation_filter}", style="yellow"
            )

    # Find all image files
    image_files = find_image_files(directory, recursive=True)

    # Initialize statistics
    stats_data = {
        "directory": str(directory.resolve()),
        "total_images": len(image_files),
        "symlink_count": 0,
        "broken_symlinks": 0,
        "total_sidecars": 0,
        "coverage_percentage": 0.0,
        "operation_counts": {},
        "processing_times": {},
        "success_rates": {},
        "data_sizes": {},
        "object_type_counts": {},  # New: YOLOv8 object type statistics
        "filter_applied": operation_filter,
        "broken_symlink_details": [],
    }

    # Process each image file with progress tracking
    sidecar_manager = Sidecar()
    sidecar_files = []

    for image_file in image_files:
        # Check if it's a symlink
        if image_file.is_symlink():
            stats_data["symlink_count"] += 1
            try:
                if not image_file.resolve().exists():
                    stats_data["broken_symlinks"] += 1
                    stats_data["broken_symlink_details"].append(
                        {
                            "symlink_path": str(image_file),
                            "target_path": str(image_file.resolve())
                            if image_file.resolve()
                            else "MISSING",
                        }
                    )
            except Exception:
                stats_data["broken_symlinks"] += 1
                stats_data["broken_symlink_details"].append(
                    {"symlink_path": str(image_file), "target_path": "MISSING"}
                )

        # Look for sidecar files
        sidecar_path = image_file.with_suffix(".json")
        if sidecar_path.exists():
            stats_data["total_sidecars"] += 1
            sidecar_files.append(sidecar_path)

        # Update progress
        pbar.update(1)

    # Calculate coverage percentage
    if stats_data["total_images"] > 0:
        stats_data["coverage_percentage"] = (
            stats_data["total_sidecars"] / stats_data["total_images"]
        ) * 100

    # Analyze sidecar files for operation statistics
    if sidecar_files:
        for sidecar_file in sidecar_files:
            try:
                import json

                with open(sidecar_file, "r") as f:
                    data = json.load(f)

                # Extract operation type and statistics
                for operation_key, operation_data in data.items():
                    if isinstance(operation_data, dict) and "success" in operation_data:
                        operation_name = operation_key.replace("_", " ").title()

                        # Apply filter if specified
                        if (
                            operation_type_filter
                            and operation_key != operation_type_filter.value
                        ):
                            continue

                        # Count operations
                        if operation_name not in stats_data["operation_counts"]:
                            stats_data["operation_counts"][operation_name] = 0
                        stats_data["operation_counts"][operation_name] += 1

                        # Track processing times
                        if "processing_time" in operation_data:
                            if operation_name not in stats_data["processing_times"]:
                                stats_data["processing_times"][operation_name] = []
                            stats_data["processing_times"][operation_name].append(
                                operation_data["processing_time"]
                            )

                        # Track success rates
                        if operation_name not in stats_data["success_rates"]:
                            stats_data["success_rates"][operation_name] = {
                                "success": 0,
                                "total": 0,
                            }
                        stats_data["success_rates"][operation_name]["total"] += 1
                        if operation_data.get("success", False):
                            stats_data["success_rates"][operation_name]["success"] += 1

                        # Track data sizes
                        data_size = len(json.dumps(operation_data))
                        if operation_name not in stats_data["data_sizes"]:
                            stats_data["data_sizes"][operation_name] = []
                        stats_data["data_sizes"][operation_name].append(data_size)

                        # Track YOLOv8 object types if this is object detection data
                        if operation_key == "yolov8" and operation_data.get("success", False):
                            objects = operation_data.get("objects", [])
                            for obj in objects:
                                if isinstance(obj, dict) and "class_name" in obj:
                                    class_name = obj["class_name"]
                                    if class_name not in stats_data["object_type_counts"]:
                                        stats_data["object_type_counts"][class_name] = 0
                                    stats_data["object_type_counts"][class_name] += 1

            except Exception:
                # Skip corrupted sidecar files
                continue

    # Calculate averages
    for operation_name in stats_data["processing_times"]:
        times = stats_data["processing_times"][operation_name]
        stats_data["processing_times"][operation_name] = (
            sum(times) / len(times) if times else 0
        )

    for operation_name in stats_data["data_sizes"]:
        sizes = stats_data["data_sizes"][operation_name]
        stats_data["data_sizes"][operation_name] = (
            sum(sizes) / len(sizes) if sizes else 0
        )

    # Add the formatted data for display compatibility
    stats_data["avg_processing_times"] = stats_data["processing_times"].copy()
    stats_data["avg_data_sizes"] = stats_data["data_sizes"].copy()

    # Calculate success rate percentages
    stats_data["success_rate_percentages"] = {}
    for operation_name in stats_data["success_rates"]:
        success_data = stats_data["success_rates"][operation_name]
        if success_data["total"] > 0:
            stats_data["success_rate_percentages"][operation_name] = (
                success_data["success"] / success_data["total"]
            ) * 100
        else:
            stats_data["success_rate_percentages"][operation_name] = 0

    # Update progress bar description with results
    pbar.set_description(
        f"Found {stats_data['total_images']} images ({stats_data['symlink_count']} symlinks), {stats_data['total_sidecars']} sidecar files"
    )

    return stats_data


def display_table_stats(stats_data: Dict[str, Any]):
    """Display statistics in table format."""

    # Import Table component
    Table = _get_table()

    # Main statistics table
    main_table = Table(title="Sidecar Statistics Overview")
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", style="green", justify="right")

    main_table.add_row("Directory", stats_data["directory"])
    main_table.add_row("Total Images", str(stats_data["total_images"]))
    main_table.add_row("Symlinks", str(stats_data.get("symlink_count", 0)))
    main_table.add_row("Broken Symlinks", str(stats_data.get("broken_symlinks", 0)))
    main_table.add_row("Total Sidecar Files", str(stats_data["total_sidecars"]))
    main_table.add_row("Coverage", f"{stats_data['coverage_percentage']:.1f}%")

    if stats_data["filter_applied"]:
        main_table.add_row("Filter Applied", stats_data["filter_applied"])

    _get_console().print(main_table)

    # Operation breakdown table
    if stats_data["operation_counts"]:
        ops_table = Table(title="Operation Breakdown")
        ops_table.add_column("Operation", style="cyan")
        ops_table.add_column("Count", style="green", justify="right")
        ops_table.add_column("Percentage", style="yellow", justify="right")
        ops_table.add_column("Avg Time (s)", style="blue", justify="right")
        ops_table.add_column("Success Rate", style="magenta", justify="right")
        ops_table.add_column("Avg Data Size", style="red", justify="right")

        total_sidecars = stats_data["total_sidecars"]

        for operation, count in stats_data["operation_counts"].items():
            percentage = (count / total_sidecars * 100) if total_sidecars > 0 else 0
            avg_time = stats_data["avg_processing_times"].get(operation, 0)
            success_rate = stats_data["success_rate_percentages"].get(operation, 0)
            avg_size = stats_data["avg_data_sizes"].get(operation, 0)

            ops_table.add_row(
                operation.replace("_", " ").title(),
                str(count),
                f"{percentage:.1f}%",
                f"{avg_time:.3f}" if avg_time > 0 else "N/A",
                f"{success_rate:.1f}%" if success_rate > 0 else "N/A",
                f"{avg_size:.0f} chars" if avg_size > 0 else "N/A",
            )

        _get_console().print(ops_table)

    # YOLOv8 Object Type Breakdown
    if stats_data["object_type_counts"]:
        obj_table = Table(title="YOLOv8 Object Type Breakdown")
        obj_table.add_column("Object Type", style="cyan")
        obj_table.add_column("Count", style="green", justify="right")
        obj_table.add_column("Percentage", style="yellow", justify="right")

        total_objects = sum(stats_data["object_type_counts"].values())
        
        # Sort by count (descending)
        sorted_objects = sorted(
            stats_data["object_type_counts"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )

        for obj_type, count in sorted_objects:
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            obj_table.add_row(
                obj_type.title(),
                str(count),
                f"{percentage:.1f}%"
            )

        _get_console().print(obj_table)

    # Coverage analysis
    if stats_data.get("file_coverage"):
        coverage_table = Table(title="File Coverage Analysis")
        coverage_table.add_column("Coverage Status", style="cyan")
        coverage_table.add_column("Count", style="green", justify="right")
        coverage_table.add_column("Percentage", style="yellow", justify="right")

        covered_count = len(stats_data.get("file_coverage", []))
        uncovered_count = stats_data["total_images"] - covered_count

        coverage_table.add_row(
            "Images with Sidecars",
            str(covered_count),
            f"{(covered_count / stats_data['total_images'] * 100):.1f}%"
            if stats_data["total_images"] > 0
            else "0%",
        )
        coverage_table.add_row(
            "Images without Sidecars",
            str(uncovered_count),
            f"{(uncovered_count / stats_data['total_images'] * 100):.1f}%"
            if stats_data["total_images"] > 0
            else "0%",
        )

        _get_console().print(coverage_table)

    # Symlink analysis
    if stats_data.get("symlink_count", 0) > 0:
        symlink_table = Table(title="Symlink Analysis")
        symlink_table.add_column("Symlink Status", style="cyan")
        symlink_table.add_column("Count", style="green", justify="right")
        symlink_table.add_column("Percentage", style="yellow", justify="right")

        symlink_count = stats_data.get("symlink_count", 0)
        broken_count = stats_data.get("broken_symlinks", 0)
        working_count = symlink_count - broken_count

        symlink_table.add_row(
            "Working Symlinks",
            str(working_count),
            f"{(working_count / stats_data['total_images'] * 100):.1f}%"
            if stats_data["total_images"] > 0
            else "0%",
        )
        symlink_table.add_row(
            "Broken Symlinks",
            str(broken_count),
            f"{(broken_count / stats_data['total_images'] * 100):.1f}%"
            if stats_data["total_images"] > 0
            else "0%",
        )
        symlink_table.add_row(
            "Regular Files",
            str(stats_data["total_images"] - symlink_count),
            f"{((stats_data['total_images'] - symlink_count) / stats_data['total_images'] * 100):.1f}%"
            if stats_data["total_images"] > 0
            else "0%",
        )

        _get_console().print(symlink_table)

        # Show broken symlinks if any
        if broken_count > 0:
            _get_console().print(
                f"\n⚠️  Found {broken_count} broken symlinks:", style="yellow"
            )
            broken_symlinks = [
                info
                for info in stats_data.get("symlink_info", {}).values()
                if info.get("broken", False)
            ]
            for symlink_info in broken_symlinks[:5]:  # Show first 5
                _get_console().print(
                    f"  {symlink_info['symlink_path']} -> {symlink_info.get('target_path', 'MISSING')}"
                )
            if len(broken_symlinks) > 5:
                _get_console().print(f"  ... and {len(broken_symlinks) - 5} more")


def display_summary_stats(stats_data: Dict[str, Any]):
    """Display statistics in summary format."""

    _get_console().print(
        f"\n📊 Sidecar Statistics Summary for {stats_data['directory']}"
    )
    _get_console().print(f"📁 Total Images: {stats_data['total_images']}")

    symlink_count = stats_data.get("symlink_count", 0)
    broken_count = stats_data.get("broken_symlinks", 0)
    if symlink_count > 0:
        _get_console().print(
            f"🔗 Symlinks: {symlink_count} ({symlink_count - broken_count} working, {broken_count} broken)"
        )

    _get_console().print(f"📄 Total Sidecar Files: {stats_data['total_sidecars']}")
    _get_console().print(f"📈 Coverage: {stats_data['coverage_percentage']:.1f}%")

    if stats_data["operation_counts"]:
        _get_console().print("\n🔍 Operation Breakdown:")
        for operation, count in stats_data["operation_counts"].items():
            percentage = (
                (count / stats_data["total_sidecars"] * 100)
                if stats_data["total_sidecars"] > 0
                else 0
            )
            _get_console().print(
                f"  {operation.replace('_', ' ').title()}: {count} files ({percentage:.1f}%)"
            )

    # YOLOv8 Object Type Summary
    if stats_data["object_type_counts"]:
        _get_console().print("\n🎯 YOLOv8 Object Types Found:")
        total_objects = sum(stats_data["object_type_counts"].values())
        
        # Sort by count (descending) and show top 10
        sorted_objects = sorted(
            stats_data["object_type_counts"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for obj_type, count in sorted_objects[:10]:  # Show top 10
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            _get_console().print(
                f"  {obj_type.title()}: {count} ({percentage:.1f}%)"
            )
        
        if len(sorted_objects) > 10:
            remaining = len(sorted_objects) - 10
            _get_console().print(f"  ... and {remaining} more object types")


def display_json_stats(stats_data: Dict[str, Any]):
    """Display statistics in JSON format."""
    import json

    _get_console().print(json.dumps(stats_data, indent=2))


def save_statistics_report(stats_data: Dict[str, Any], output_path: Path):
    """Save detailed statistics report to file."""

    report = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "statistics": stats_data,
        "summary": {
            "total_images": stats_data["total_images"],
            "total_sidecars": stats_data["total_sidecars"],
            "coverage_percentage": stats_data["coverage_percentage"],
            "operations_found": list(stats_data["operation_counts"].keys()),
        },
    }

    try:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        _get_console().print(f"💾 Report saved to {output_path}", style="green")
    except Exception as e:
        _get_console().print(f"❌ Failed to save report: {e}", style="red")


@sidecar_group.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--operation",
    "-o",
    type=click.Choice(
        [
            "face_detection",
            "object_detection",
            "ball_detection",
            "quality_assessment",
            "game_detection",
        ]
    ),
    help="Filter by specific operation type",
)
@click.option(
    "--min-age-days",
    "min_age_days",
    type=int,
    default=30,
    help="Minimum age in days for sidecar files to be considered stale",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    help="Show what would be cleaned without actually cleaning",
)
@click.pass_context
def cleanup(
    ctx: click.Context,
    directory: Path,
    operation: Optional[str],
    min_age_days: int,
    dry_run: bool,
):
    """
    Clean up stale or orphaned sidecar files.

    DIRECTORY should contain images with sidecar files.
    """

    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core

    core = get_core(ctx)

    _get_console().print(
        f"🧹 Cleaning up sidecar files in {directory}...", style="blue"
    )

    if dry_run:
        _get_console().print(
            "🔍 Dry run mode - no files will be deleted", style="yellow"
        )

    # Find orphaned sidecar files
    orphaned_count = core.cleanup_orphaned_sidecars(directory)

    if orphaned_count > 0:
        if dry_run:
            _get_console().print(
                f"Would remove {orphaned_count} orphaned sidecar files", style="yellow"
            )
        else:
            _get_console().print(
                f"✅ Removed {orphaned_count} orphaned sidecar files", style="green"
            )
    else:
        _get_console().print("✅ No orphaned sidecar files found", style="green")

    # TODO: Add stale file cleanup based on age
    _get_console().print("Stale file cleanup not yet implemented", style="yellow")


@sidecar_group.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--operation",
    "-o",
    type=click.Choice(
        [
            "face_detection",
            "object_detection",
            "ball_detection",
            "quality_assessment",
            "game_detection",
        ]
    ),
    help="Filter by specific operation type",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for extracted data",
)
@click.pass_context
def export(
    ctx: click.Context,
    directory: Path,
    operation: Optional[str],
    output: Optional[Path],
):
    """
    Export sidecar data to various formats.

    DIRECTORY should contain images with sidecar files.
    """

    _get_console().print(f"📤 Exporting sidecar data from {directory}...", style="blue")

    # TODO: Implement sidecar data export
    _get_console().print("Sidecar data export not yet implemented", style="yellow")
