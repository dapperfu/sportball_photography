"""
Color-based photo sorting CLI command.

This module provides the CLI interface for sorting photos by jersey colors.
"""

import click
from pathlib import Path
from typing import Optional
from loguru import logger

from ..detectors.color_detector import ColorDetector
from ..utils.file_utils import FileUtils
from ..utils.logging_utils import ProcessingLogger


@click.command()
@click.option('--input', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input directory containing photos')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              required=True,
              help='Output directory for organized photos')
@click.option('--confidence',
              type=float,
              default=0.8,
              help='Color detection confidence threshold (0.0-1.0)')
@click.option('--colors',
              type=str,
              help='Comma-separated list of colors to detect (e.g., "red,blue,green")')
@click.option('--create-symlinks/--no-symlinks',
              default=True,
              help='Create symbolic links instead of copying files')
@click.option('--dry-run',
              is_flag=True,
              help='Preview changes without creating directories')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
def color_sorter(input: Path,
                 output: Path,
                 confidence: float,
                 colors: Optional[str],
                 create_symlinks: bool,
                 dry_run: bool,
                 verbose: bool):
    """
    Sort photos by jersey colors.
    
    This command analyzes photos to detect dominant jersey colors and organizes
    them into color-coded directories (e.g., Red/, Blue/, Green/).
    
    Examples:
    
        # Basic color sorting
        soccer-photo-sorter color-sort -i ./photos -o ./sorted
        
        # Sort with custom confidence threshold
        soccer-photo-sorter color-sort -i ./photos -o ./sorted --confidence 0.9
        
        # Sort only specific colors
        soccer-photo-sorter color-sort -i ./photos -o ./sorted --colors "red,blue"
        
        # Preview changes without organizing
        soccer-photo-sorter color-sort -i ./photos -o ./sorted --dry-run
    """
    # Setup logging
    if verbose:
        logger.remove()
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")
    
    # Validate confidence threshold
    if not 0.0 <= confidence <= 1.0:
        click.echo("Error: Confidence must be between 0.0 and 1.0", err=True)
        return
    
    # Parse colors if specified
    target_colors = None
    if colors:
        target_colors = [color.strip().lower() for color in colors.split(',')]
        click.echo(f"Target colors: {', '.join(target_colors)}")
    
    # Initialize components
    file_utils = FileUtils()
    color_detector = ColorDetector(confidence_threshold=confidence)
    
    # Find image files
    image_files = file_utils.find_image_files(input, recursive=True)
    if not image_files:
        click.echo(f"No image files found in {input}", err=True)
        return
    
    click.echo(f"Found {len(image_files)} image files")
    
    # Process images
    with ProcessingLogger("Color Detection", len(image_files)) as proc_logger:
        color_categories = {}
        
        for i, image_path in enumerate(image_files):
            try:
                # Detect colors
                detected_colors = color_detector.detect_colors(image_path)
                
                if detected_colors:
                    for color_name, confidence_score in detected_colors:
                        # Skip if target colors specified and this color not in list
                        if target_colors and color_name.lower() not in target_colors:
                            continue
                        
                        # Skip if confidence too low
                        if confidence_score < confidence:
                            continue
                        
                        # Add to category
                        if color_name not in color_categories:
                            color_categories[color_name] = []
                        color_categories[color_name].append(image_path)
                        
                        if verbose:
                            click.echo(f"  {image_path.name}: {color_name} (confidence: {confidence_score:.2f})")
                
                proc_logger.log_progress(i + 1)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                proc_logger.log_error(e, image_path)
    
    # Display results
    click.echo(f"\nColor Detection Results:")
    total_categorized = 0
    for color_name, files in color_categories.items():
        click.echo(f"  {color_name}: {len(files)} files")
        total_categorized += len(files)
    
    click.echo(f"Total categorized: {total_categorized}/{len(image_files)}")
    
    if not color_categories:
        click.echo("No colors detected above confidence threshold")
        return
    
    # Organize files
    if not dry_run:
        click.echo(f"\nOrganizing files into {output}...")
        
        # Create output directory
        output.mkdir(parents=True, exist_ok=True)
        
        # Organize files
        stats = file_utils.organize_files(
            files=image_files,
            output_dir=output,
            categories=color_categories,
            use_symlinks=create_symlinks,
            dry_run=False
        )
        
        click.echo(f"Organization complete:")
        click.echo(f"  Files organized: {stats['organized']}")
        click.echo(f"  Errors: {stats['errors']}")
    else:
        click.echo(f"\nDry run - would organize {total_categorized} files into:")
        for color_name, files in color_categories.items():
            click.echo(f"  {output}/{color_name}/ ({len(files)} files)")


@click.command()
@click.option('--input', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input directory containing photos')
@click.option('--confidence',
              type=float,
              default=0.8,
              help='Color detection confidence threshold (0.0-1.0)')
@click.option('--colors',
              type=str,
              help='Comma-separated list of colors to analyze')
@click.option('--output-format',
              type=click.Choice(['table', 'json', 'csv']),
              default='table',
              help='Output format for analysis results')
def analyze_colors(input: Path,
                   confidence: float,
                   colors: Optional[str],
                   output_format: str):
    """
    Analyze color distribution in photos without organizing files.
    
    This command analyzes photos to show color distribution statistics
    without actually organizing the files.
    """
    # Parse colors if specified
    target_colors = None
    if colors:
        target_colors = [color.strip().lower() for color in colors.split(',')]
    
    # Initialize components
    file_utils = FileUtils()
    color_detector = ColorDetector(confidence_threshold=confidence)
    
    # Find image files
    image_files = file_utils.find_image_files(input, recursive=True)
    if not image_files:
        click.echo(f"No image files found in {input}", err=True)
        return
    
    click.echo(f"Analyzing {len(image_files)} image files...")
    
    # Analyze colors
    color_stats = {}
    processed_count = 0
    
    for image_path in image_files:
        try:
            detected_colors = color_detector.detect_colors(image_path)
            
            if detected_colors:
                for color_name, confidence_score in detected_colors:
                    if target_colors and color_name.lower() not in target_colors:
                        continue
                    
                    if confidence_score >= confidence:
                        if color_name not in color_stats:
                            color_stats[color_name] = {
                                'count': 0,
                                'total_confidence': 0.0,
                                'files': []
                            }
                        
                        color_stats[color_name]['count'] += 1
                        color_stats[color_name]['total_confidence'] += confidence_score
                        color_stats[color_name]['files'].append({
                            'file': image_path.name,
                            'confidence': confidence_score
                        })
            
            processed_count += 1
            if processed_count % 10 == 0:
                click.echo(f"Processed {processed_count}/{len(image_files)} files...")
                
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
    
    # Display results
    if output_format == 'table':
        click.echo(f"\nColor Analysis Results:")
        click.echo("=" * 50)
        
        for color_name, stats in sorted(color_stats.items()):
            avg_confidence = stats['total_confidence'] / stats['count']
            click.echo(f"{color_name:12} | {stats['count']:4} files | Avg confidence: {avg_confidence:.2f}")
    
    elif output_format == 'json':
        import json
        results = {
            'total_files': len(image_files),
            'processed_files': processed_count,
            'colors': color_stats
        }
        click.echo(json.dumps(results, indent=2))
    
    elif output_format == 'csv':
        click.echo("Color,Count,Avg_Confidence")
        for color_name, stats in sorted(color_stats.items()):
            avg_confidence = stats['total_confidence'] / stats['count']
            click.echo(f"{color_name},{stats['count']},{avg_confidence:.2f}")
    
    click.echo(f"\nAnalysis complete: {processed_count}/{len(image_files)} files processed")
