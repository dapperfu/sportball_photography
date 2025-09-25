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

from ..utils import get_core

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
    """Collect comprehensive sidecar statistics."""
    
    # Find all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
    all_image_files = []
    
    for ext in image_extensions:
        all_image_files.extend(directory.glob(f'*{ext}'))
        all_image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    # Separate symlinks from regular files and collect sidecar files
    symlink_info = {}
    sidecar_files = []
    
    for image_file in all_image_files:
        if image_file.is_symlink():
            try:
                # Resolve symlink
                target_path = image_file.resolve()
                
                # Check for sidecar next to target
                sidecar_path = target_path.with_suffix('.json')
                if sidecar_path.exists():
                    sidecar_files.append(sidecar_path)
                
                symlink_info[str(image_file)] = {
                    'symlink_path': str(image_file),
                    'target_path': str(target_path),
                    'is_symlink': True,
                    'broken': not target_path.exists()
                }
            except Exception as e:
                symlink_info[str(image_file)] = {
                    'symlink_path': str(image_file),
                    'target_path': None,
                    'is_symlink': True,
                    'error': str(e)
                }
        else:
            # Regular file - check for sidecar in same directory
            sidecar_path = image_file.with_suffix('.json')
            if sidecar_path.exists():
                sidecar_files.append(sidecar_path)
            
            symlink_info[str(image_file)] = {
                'symlink_path': str(image_file),
                'target_path': str(image_file),
                'is_symlink': False
            }
    
    # Also look for pattern-based sidecars in the directory
    sidecar_files.extend(directory.glob('*_*.json'))
    
    # Remove duplicates
    sidecar_files = list(set(sidecar_files))
    
    progress.update(task_id, description=f"Found {len(all_image_files)} images ({len([f for f in symlink_info.values() if f['is_symlink']])} symlinks), {len(sidecar_files)} sidecar files")
    
    # Analyze sidecar files
    operation_counts = Counter()
    operation_details = defaultdict(list)
    file_coverage = {}
    processing_times = defaultdict(list)
    success_rates = defaultdict(lambda: {'success': 0, 'total': 0})
    data_sizes = defaultdict(list)
    symlink_sidecar_mapping = {}
    
    for sidecar_file in sidecar_files:
        try:
            with open(sidecar_file, 'r') as f:
                data = json.load(f)
            
            # Extract operation type - check multiple possible structures
            operation = 'unknown'
            
            # Check for sidecar_info structure
            if 'sidecar_info' in data:
                operation = data['sidecar_info'].get('operation_type', 'unknown')
            # Check for Face_detector structure
            elif 'Face_detector' in data:
                operation = 'face_detection'
            # Check for other detector structures
            elif 'Object_detector' in data:
                operation = 'object_detection'
            elif 'Ball_detector' in data:
                operation = 'ball_detection'
            elif 'Quality_assessor' in data:
                operation = 'quality_assessment'
            elif 'Game_detector' in data:
                operation = 'game_detection'
            
            # Apply filter if specified
            if operation_filter and operation != operation_filter:
                continue
            
            operation_counts[operation] += 1
            
            # Extract image name from sidecar filename
            image_name = sidecar_file.stem.rsplit('_', 1)[0]
            image_path = None
            symlink_info_for_file = None
            
            # Find corresponding image file (could be symlink)
            # First check in the current directory
            for ext in image_extensions:
                potential_image = directory / f"{image_name}{ext}"
                if potential_image.exists():
                    image_path = potential_image
                    break
            
            # If not found in current directory, check if this sidecar is next to a target file
            if not image_path:
                sidecar_dir = sidecar_file.parent
                for ext in image_extensions:
                    potential_image = sidecar_dir / f"{image_name}{ext}"
                    if potential_image.exists():
                        # This sidecar is next to the actual file, find the corresponding symlink
                        for symlink_path, symlink_data in symlink_info.items():
                            if symlink_data.get('target_path') == str(potential_image):
                                image_path = Path(symlink_data['symlink_path'])
                                break
                        if not image_path:
                            # No symlink found, use the actual file
                            image_path = potential_image
                        break
            
            if image_path:
                # Check if this is a symlink and get target info
                if image_path.is_symlink():
                    try:
                        target_path = image_path.resolve()
                        symlink_info_for_file = {
                            'symlink_path': str(image_path),
                            'target_path': str(target_path),
                            'is_symlink': True,
                            'broken': not target_path.exists()
                        }
                        # Use target path for coverage tracking
                        file_coverage[str(target_path)] = operation
                        symlink_sidecar_mapping[str(target_path)] = {
                            'sidecar_file': str(sidecar_file),
                            'symlink_path': str(image_path),
                            'operation': operation
                        }
                    except Exception as e:
                        symlink_info_for_file = {
                            'symlink_path': str(image_path),
                            'target_path': None,
                            'is_symlink': True,
                            'error': str(e)
                        }
                        file_coverage[str(image_path)] = operation
                else:
                    file_coverage[str(image_path)] = operation
                
                operation_details[operation].append({
                    'sidecar_file': str(sidecar_file),
                    'image_file': str(image_path),
                    'file_size': sidecar_file.stat().st_size,
                    'symlink_info': symlink_info_for_file
                })
            
            # Extract processing time
            processing_time = data.get('data', {}).get('processing_time')
            if processing_time:
                processing_times[operation].append(processing_time)
            
            # Extract success status
            success = data.get('data', {}).get('success', True)
            success_rates[operation]['total'] += 1
            if success:
                success_rates[operation]['success'] += 1
            
            # Extract data size info
            data_size = len(str(data.get('data', {})))
            data_sizes[operation].append(data_size)
            
        except Exception as e:
            console.print(f"⚠️  Error reading {sidecar_file}: {e}", style="yellow")
    
    # Calculate statistics
    total_images = len(all_image_files)
    total_sidecars = sum(operation_counts.values())
    coverage_percentage = (total_sidecars / total_images * 100) if total_images > 0 else 0
    
    # Calculate average processing times
    avg_processing_times = {}
    for operation, times in processing_times.items():
        if times:
            avg_processing_times[operation] = sum(times) / len(times)
    
    # Calculate success rates
    success_rate_percentages = {}
    for operation, rates in success_rates.items():
        if rates['total'] > 0:
            success_rate_percentages[operation] = (rates['success'] / rates['total']) * 100
    
    # Calculate average data sizes
    avg_data_sizes = {}
    for operation, sizes in data_sizes.items():
        if sizes:
            avg_data_sizes[operation] = sum(sizes) / len(sizes)
    
    # Calculate symlink statistics
    symlink_count = len([f for f in symlink_info.values() if f['is_symlink']])
    broken_symlinks = len([f for f in symlink_info.values() if f.get('broken', False)])
    
    return {
        'directory': str(directory),
        'total_images': total_images,
        'total_sidecars': total_sidecars,
        'coverage_percentage': coverage_percentage,
        'operation_counts': dict(operation_counts),
        'operation_details': dict(operation_details),
        'file_coverage': file_coverage,
        'avg_processing_times': avg_processing_times,
        'success_rate_percentages': success_rate_percentages,
        'avg_data_sizes': avg_data_sizes,
        'filter_applied': operation_filter,
        'symlink_info': symlink_info,
        'symlink_count': symlink_count,
        'broken_symlinks': broken_symlinks,
        'symlink_sidecar_mapping': symlink_sidecar_mapping
    }


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
