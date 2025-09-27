"""
Quality Assessment Commands

CLI commands for photo quality assessment operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..utils import get_core, find_image_files

console = Console()


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def quality_group():
    """Photo quality assessment commands."""
    pass


@quality_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for results')
@click.option('--min-score', 'min_score',
              type=float,
              default=0.0,
              help='Minimum quality score threshold (0.0-1.0)')
@click.option('--max-score', 'max_score',
              type=float,
              default=1.0,
              help='Maximum quality score threshold (0.0-1.0)')
@click.option('--save-sidecar/--no-sidecar',
              default=True,
              help='Save results to sidecar files')
@click.option('--filter-low-quality', 'filter_low_quality',
              is_flag=True,
              help='Filter out low-quality images')
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
@click.pass_context
def assess(ctx: click.Context, 
           input_path: Path, 
           output: Optional[Path],
           min_score: float,
           max_score: float,
           save_sidecar: bool,
           filter_low_quality: bool,
           no_recursive: bool):
    """
    Assess photo quality using multiple metrics.
    
    INPUT_PATH can be a single image file or a directory containing images.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    core = get_core(ctx)
    
    # Find image files (recursive by default)
    recursive = not no_recursive
    image_paths = find_image_files(input_path, recursive=recursive)
    
    if not image_paths:
        console.print("❌ No image files found", style="red")
        return
    
    console.print(f"📸 Assessing quality of {len(image_paths)} images...", style="blue")
    
    # Perform quality assessment
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing images...", total=len(image_paths))
        
        results = core.assess_quality(
            image_paths,
            save_sidecar=save_sidecar,
            min_score=min_score,
            max_score=max_score
        )
        
        progress.update(task, completed=len(image_paths))
    
    # Display results
    display_quality_results(results, filter_low_quality, output)


def display_quality_results(results: dict, filter_low_quality: bool, output_dir: Optional[Path]):
    """Display quality assessment results."""
    
    # Create results table
    table = Table(title="Quality Assessment Results")
    table.add_column("Image", style="cyan")
    table.add_column("Overall Score", style="green", justify="right")
    table.add_column("Grade", style="yellow")
    table.add_column("Sharpness", style="blue", justify="right")
    table.add_column("Focus", style="magenta", justify="right")
    table.add_column("Exposure", style="blue", justify="right")
    table.add_column("Contrast", style="blue", justify="right")
    table.add_column("Success", style="green")
    table.add_column("Error", style="red")
    
    total_score = 0
    successful_images = 0
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    low_quality_images = []
    
    for image_path, result in results.items():
        if result.get('success', False):
            quality_data = result.get('quality', {})
            overall_score = quality_data.get('overall_score', 0)
            grade = quality_data.get('grade', 'F')
            sharpness = quality_data.get('sharpness', 0)
            focus = quality_data.get('focus', 0)
            exposure = quality_data.get('exposure', 0)
            contrast = quality_data.get('contrast', 0)
            
            total_score += overall_score
            successful_images += 1
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            # Check if low quality
            if overall_score < 0.5:
                low_quality_images.append(image_path)
            
            table.add_row(
                Path(image_path).name,
                f"{overall_score:.2f}",
                grade,
                f"{sharpness:.2f}",
                f"{focus:.2f}",
                f"{exposure:.2f}",
                f"{contrast:.2f}",
                "✅",
                ""
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            table.add_row(
                Path(image_path).name,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "❌",
                error_msg[:30] + "..." if len(error_msg) > 30 else error_msg
            )
    
    console.print(table)
    
    if successful_images > 0:
        avg_score = total_score / successful_images
        console.print(f"\n📊 Summary: {successful_images}/{len(results)} images processed")
        console.print(f"Average quality score: {avg_score:.2f}")
        
        # Display grade distribution
        console.print("\n📈 Grade Distribution:")
        for grade, count in sorted(grade_counts.items()):
            if count > 0:
                console.print(f"  Grade {grade}: {count} images")
        
        # Filter low quality if requested
        if filter_low_quality and low_quality_images:
            console.print(f"\n🔍 Found {len(low_quality_images)} low-quality images:")
            for img_path in low_quality_images[:10]:  # Show first 10
                console.print(f"  {Path(img_path).name}")
            if len(low_quality_images) > 10:
                console.print(f"  ... and {len(low_quality_images) - 10} more")


@quality_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--min-score', 'min_score',
              type=float,
              default=0.5,
              help='Minimum quality score to keep (0.0-1.0)')
@click.option('--copy/--move',
              default=False,
              help='Copy files instead of moving them')
@click.option('--create-subdirs', 'create_subdirs',
              is_flag=True,
              help='Create subdirectories by quality grade')
@click.pass_context
def filter(ctx: click.Context, 
           input_path: Path, 
           output_dir: Path,
           min_score: float,
           copy: bool,
           create_subdirs: bool):
    """
    Filter images by quality score.
    
    INPUT_PATH should be a directory containing images with quality assessment sidecar files.
    OUTPUT_DIR is where filtered images will be saved.
    """
    
    core = get_core(ctx)
    
    console.print(f"🔍 Filtering images by quality score >= {min_score}...", style="blue")
    console.print(f"Output: {output_dir}")
    
    # TODO: Implement quality filtering
    console.print("Quality filtering not yet implemented", style="yellow")


@quality_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output file for report')
@click.option('--format', 'report_format',
              type=click.Choice(['json', 'csv', 'html']),
              default='json',
              help='Report format')
@click.pass_context
def report(ctx: click.Context, 
           input_path: Path, 
           output: Optional[Path],
           report_format: str):
    """
    Generate quality assessment report.
    
    INPUT_PATH should be a directory containing images with quality assessment sidecar files.
    """
    
    core = get_core(ctx)
    
    console.print(f"📊 Generating quality report for {input_path}...", style="blue")
    
    # TODO: Implement quality reporting
    console.print("Quality reporting not yet implemented", style="yellow")
