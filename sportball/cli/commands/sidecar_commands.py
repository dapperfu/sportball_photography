"""
Sidecar Management Commands

CLI commands for sidecar file operations and statistics.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional, Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from collections import defaultdict, Counter
import json

# Lazy imports to avoid heavy dependencies at startup
# from ..utils import get_core
# from ...sidecar import Sidecar, OperationType

console = Console()


@click.group()
def sidecar_group():
    """Sidecar file management and statistics commands."""
    pass


@sidecar_group.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--operation', '-o',
              type=click.Choice(['face_detection', 'object_detection', 'ball_detection', 'quality_assessment', 'game_detection']),
              help='Filter by specific operation type')
@click.option('--format', 'output_format',
              type=click.Choice(['table', 'json', 'summary']),
              default='table',
              help='Output format')
@click.option('--save-report', 'save_report',
              type=click.Path(path_type=Path),
              help='Save detailed report to file')
@click.pass_context
def stats(ctx: click.Context, 
          directory: Path, 
          operation: Optional[str],
          output_format: str,
          save_report: Optional[Path]):
    """
    Generate comprehensive statistics about sidecar files in a directory.
    
    DIRECTORY should contain images with sidecar files.
    """
    
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core
    core = get_core(ctx)
    
    console.print(f"📊 Analyzing sidecar files in {directory}...", style="blue")
    
    # Collect comprehensive statistics
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Scanning directory...", total=None)
        
        stats_data = collect_sidecar_statistics(directory, operation, progress, task)
        
        progress.update(task, completed=True, description="Analysis complete")
    
    # Display results based on format
    if output_format == 'json':
        display_json_stats(stats_data)
    elif output_format == 'summary':
        display_summary_stats(stats_data)
    else:
        display_table_stats(stats_data)
    
    # Save report if requested
    if save_report:
        save_statistics_report(stats_data, save_report)


def collect_sidecar_statistics(directory: Path, 
                              operation_filter: Optional[str],
                              progress: Progress,
                              task_id: int) -> Dict[str, Any]:
    """Collect comprehensive sidecar statistics using the new Sidecar class."""
    
    # Lazy import to avoid heavy dependencies at startup
    from ...sidecar import Sidecar, OperationType
    
    # Convert operation filter to OperationType enum
    operation_type_filter = None
    if operation_filter:
        try:
            operation_type_filter = OperationType(operation_filter)
        except ValueError:
            console.print(f"⚠️  Invalid operation filter: {operation_filter}", style="yellow")
    
    # Use the new Sidecar class
    sidecar_manager = Sidecar()
    stats_data = sidecar_manager.get_statistics(directory, operation_type_filter)
    
    progress.update(task_id, description=f"Found {stats_data['total_images']} images ({stats_data['symlink_count']} symlinks), {stats_data['total_sidecars']} sidecar files")
    
    return stats_data


def display_table_stats(stats_data: Dict[str, Any]):
    """Display statistics in table format."""
    
    # Main statistics table
    main_table = Table(title="Sidecar Statistics Overview")
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", style="green", justify="right")
    
    main_table.add_row("Directory", stats_data['directory'])
    main_table.add_row("Total Images", str(stats_data['total_images']))
    main_table.add_row("Symlinks", str(stats_data.get('symlink_count', 0)))
    main_table.add_row("Broken Symlinks", str(stats_data.get('broken_symlinks', 0)))
    main_table.add_row("Total Sidecar Files", str(stats_data['total_sidecars']))
    main_table.add_row("Coverage", f"{stats_data['coverage_percentage']:.1f}%")
    
    if stats_data['filter_applied']:
        main_table.add_row("Filter Applied", stats_data['filter_applied'])
    
    console.print(main_table)
    
    # Operation breakdown table
    if stats_data['operation_counts']:
        ops_table = Table(title="Operation Breakdown")
        ops_table.add_column("Operation", style="cyan")
        ops_table.add_column("Count", style="green", justify="right")
        ops_table.add_column("Percentage", style="yellow", justify="right")
        ops_table.add_column("Avg Time (s)", style="blue", justify="right")
        ops_table.add_column("Success Rate", style="magenta", justify="right")
        ops_table.add_column("Avg Data Size", style="red", justify="right")
        
        total_sidecars = stats_data['total_sidecars']
        
        for operation, count in stats_data['operation_counts'].items():
            percentage = (count / total_sidecars * 100) if total_sidecars > 0 else 0
            avg_time = stats_data['avg_processing_times'].get(operation, 0)
            success_rate = stats_data['success_rate_percentages'].get(operation, 0)
            avg_size = stats_data['avg_data_sizes'].get(operation, 0)
            
            ops_table.add_row(
                operation.replace('_', ' ').title(),
                str(count),
                f"{percentage:.1f}%",
                f"{avg_time:.3f}" if avg_time > 0 else "N/A",
                f"{success_rate:.1f}%" if success_rate > 0 else "N/A",
                f"{avg_size:.0f} chars" if avg_size > 0 else "N/A"
            )
        
        console.print(ops_table)
    
    # Coverage analysis
    if stats_data['file_coverage']:
        coverage_table = Table(title="File Coverage Analysis")
        coverage_table.add_column("Coverage Status", style="cyan")
        coverage_table.add_column("Count", style="green", justify="right")
        coverage_table.add_column("Percentage", style="yellow", justify="right")
        
        covered_count = len(stats_data['file_coverage'])
        uncovered_count = stats_data['total_images'] - covered_count
        
        coverage_table.add_row(
            "Images with Sidecars",
            str(covered_count),
            f"{(covered_count / stats_data['total_images'] * 100):.1f}%" if stats_data['total_images'] > 0 else "0%"
        )
        coverage_table.add_row(
            "Images without Sidecars",
            str(uncovered_count),
            f"{(uncovered_count / stats_data['total_images'] * 100):.1f}%" if stats_data['total_images'] > 0 else "0%"
        )
        
        console.print(coverage_table)
    
    # Symlink analysis
    if stats_data.get('symlink_count', 0) > 0:
        symlink_table = Table(title="Symlink Analysis")
        symlink_table.add_column("Symlink Status", style="cyan")
        symlink_table.add_column("Count", style="green", justify="right")
        symlink_table.add_column("Percentage", style="yellow", justify="right")
        
        symlink_count = stats_data.get('symlink_count', 0)
        broken_count = stats_data.get('broken_symlinks', 0)
        working_count = symlink_count - broken_count
        
        symlink_table.add_row(
            "Working Symlinks",
            str(working_count),
            f"{(working_count / stats_data['total_images'] * 100):.1f}%" if stats_data['total_images'] > 0 else "0%"
        )
        symlink_table.add_row(
            "Broken Symlinks",
            str(broken_count),
            f"{(broken_count / stats_data['total_images'] * 100):.1f}%" if stats_data['total_images'] > 0 else "0%"
        )
        symlink_table.add_row(
            "Regular Files",
            str(stats_data['total_images'] - symlink_count),
            f"{((stats_data['total_images'] - symlink_count) / stats_data['total_images'] * 100):.1f}%" if stats_data['total_images'] > 0 else "0%"
        )
        
        console.print(symlink_table)
        
        # Show broken symlinks if any
        if broken_count > 0:
            console.print(f"\n⚠️  Found {broken_count} broken symlinks:", style="yellow")
            broken_symlinks = [info for info in stats_data.get('symlink_info', {}).values() 
                              if info.get('broken', False)]
            for symlink_info in broken_symlinks[:5]:  # Show first 5
                console.print(f"  {symlink_info['symlink_path']} -> {symlink_info.get('target_path', 'MISSING')}")
            if len(broken_symlinks) > 5:
                console.print(f"  ... and {len(broken_symlinks) - 5} more")


def display_summary_stats(stats_data: Dict[str, Any]):
    """Display statistics in summary format."""
    
    console.print(f"\n📊 Sidecar Statistics Summary for {stats_data['directory']}")
    console.print(f"📁 Total Images: {stats_data['total_images']}")
    
    symlink_count = stats_data.get('symlink_count', 0)
    broken_count = stats_data.get('broken_symlinks', 0)
    if symlink_count > 0:
        console.print(f"🔗 Symlinks: {symlink_count} ({symlink_count - broken_count} working, {broken_count} broken)")
    
    console.print(f"📄 Total Sidecar Files: {stats_data['total_sidecars']}")
    console.print(f"📈 Coverage: {stats_data['coverage_percentage']:.1f}%")
    
    if stats_data['operation_counts']:
        console.print(f"\n🔍 Operation Breakdown:")
        for operation, count in stats_data['operation_counts'].items():
            percentage = (count / stats_data['total_sidecars'] * 100) if stats_data['total_sidecars'] > 0 else 0
            console.print(f"  {operation.replace('_', ' ').title()}: {count} files ({percentage:.1f}%)")


def display_json_stats(stats_data: Dict[str, Any]):
    """Display statistics in JSON format."""
    import json
    console.print(json.dumps(stats_data, indent=2))


def save_statistics_report(stats_data: Dict[str, Any], output_path: Path):
    """Save detailed statistics report to file."""
    
    report = {
        'generated_at': __import__('datetime').datetime.now().isoformat(),
        'statistics': stats_data,
        'summary': {
            'total_images': stats_data['total_images'],
            'total_sidecars': stats_data['total_sidecars'],
            'coverage_percentage': stats_data['coverage_percentage'],
            'operations_found': list(stats_data['operation_counts'].keys())
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        console.print(f"💾 Report saved to {output_path}", style="green")
    except Exception as e:
        console.print(f"❌ Failed to save report: {e}", style="red")


@sidecar_group.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--operation', '-o',
              type=click.Choice(['face_detection', 'object_detection', 'ball_detection', 'quality_assessment', 'game_detection']),
              help='Filter by specific operation type')
@click.option('--min-age-days', 'min_age_days',
              type=int,
              default=30,
              help='Minimum age in days for sidecar files to be considered stale')
@click.option('--dry-run', 'dry_run',
              is_flag=True,
              help='Show what would be cleaned without actually cleaning')
@click.pass_context
def cleanup(ctx: click.Context, 
            directory: Path, 
            operation: Optional[str],
            min_age_days: int,
            dry_run: bool):
    """
    Clean up stale or orphaned sidecar files.
    
    DIRECTORY should contain images with sidecar files.
    """
    
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core
    core = get_core(ctx)
    
    console.print(f"🧹 Cleaning up sidecar files in {directory}...", style="blue")
    
    if dry_run:
        console.print("🔍 Dry run mode - no files will be deleted", style="yellow")
    
    # Find orphaned sidecar files
    orphaned_count = core.cleanup_orphaned_sidecars(directory)
    
    if orphaned_count > 0:
        if dry_run:
            console.print(f"Would remove {orphaned_count} orphaned sidecar files", style="yellow")
        else:
            console.print(f"✅ Removed {orphaned_count} orphaned sidecar files", style="green")
    else:
        console.print("✅ No orphaned sidecar files found", style="green")
    
    # TODO: Add stale file cleanup based on age
    console.print("Stale file cleanup not yet implemented", style="yellow")


@sidecar_group.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--operation', '-o',
              type=click.Choice(['face_detection', 'object_detection', 'ball_detection', 'quality_assessment', 'game_detection']),
              help='Filter by specific operation type')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for extracted data')
@click.pass_context
def export(ctx: click.Context, 
           directory: Path, 
           operation: Optional[str],
           output: Optional[Path]):
    """
    Export sidecar data to various formats.
    
    DIRECTORY should contain images with sidecar files.
    """
    
    console.print(f"📤 Exporting sidecar data from {directory}...", style="blue")
    
    # TODO: Implement sidecar data export
    console.print("Sidecar data export not yet implemented", style="yellow")
