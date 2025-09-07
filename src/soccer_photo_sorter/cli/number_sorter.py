"""
Number-based photo sorting CLI command.

This module provides the CLI interface for sorting photos by jersey numbers.
"""

import click
from pathlib import Path
from typing import Optional, List
from loguru import logger

from ..detectors.number_detector import NumberDetector
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
              default=0.7,
              help='Number detection confidence threshold (0.0-1.0)')
@click.option('--numbers',
              type=str,
              help='Comma-separated list of numbers to detect (e.g., "1,7,15,23")')
@click.option('--min-number',
              type=int,
              default=1,
              help='Minimum jersey number to detect')
@click.option('--max-number',
              type=int,
              default=99,
              help='Maximum jersey number to detect')
@click.option('--create-symlinks/--no-symlinks',
              default=True,
              help='Create symbolic links instead of copying files')
@click.option('--dry-run',
              is_flag=True,
              help='Preview changes without creating directories')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
def number_sorter(input: Path,
                  output: Path,
                  confidence: float,
                  numbers: Optional[str],
                  min_number: int,
                  max_number: int,
                  create_symlinks: bool,
                  dry_run: bool,
                  verbose: bool):
    """
    Sort photos by jersey numbers.
    
    This command uses OCR to detect jersey numbers in photos and organizes
    them into number-coded directories (e.g., Number_01/, Number_07/, Number_15/).
    
    Examples:
    
        # Basic number sorting
        soccer-photo-sorter number-sort -i ./photos -o ./sorted
        
        # Sort with custom confidence threshold
        soccer-photo-sorter number-sort -i ./photos -o ./sorted --confidence 0.8
        
        # Sort only specific numbers
        soccer-photo-sorter number-sort -i ./photos -o ./sorted --numbers "1,7,15,23"
        
        # Sort numbers in range
        soccer-photo-sorter number-sort -i ./photos -o ./sorted --min-number 1 --max-number 30
        
        # Preview changes without organizing
        soccer-photo-sorter number-sort -i ./photos -o ./sorted --dry-run
    """
    # Setup logging
    if verbose:
        logger.remove()
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")
    
    # Validate confidence threshold
    if not 0.0 <= confidence <= 1.0:
        click.echo("Error: Confidence must be between 0.0 and 1.0", err=True)
        return
    
    # Validate number range
    if min_number > max_number:
        click.echo("Error: min-number must be less than or equal to max-number", err=True)
        return
    
    # Parse numbers if specified
    target_numbers = None
    if numbers:
        try:
            target_numbers = [int(num.strip()) for num in numbers.split(',')]
            # Validate numbers are in range
            for num in target_numbers:
                if not min_number <= num <= max_number:
                    click.echo(f"Warning: Number {num} is outside range [{min_number}, {max_number}]", err=True)
            click.echo(f"Target numbers: {', '.join(map(str, target_numbers))}")
        except ValueError:
            click.echo("Error: Invalid number format. Use comma-separated integers (e.g., '1,7,15')", err=True)
            return
    
    # Initialize components
    file_utils = FileUtils()
    number_detector = NumberDetector(
        confidence_threshold=confidence,
        min_number=min_number,
        max_number=max_number
    )
    
    # Find image files
    image_files = file_utils.find_image_files(input, recursive=True)
    if not image_files:
        click.echo(f"No image files found in {input}", err=True)
        return
    
    click.echo(f"Found {len(image_files)} image files")
    
    # Process images
    with ProcessingLogger("Number Detection", len(image_files)) as proc_logger:
        number_categories = {}
        
        for i, image_path in enumerate(image_files):
            try:
                # Detect numbers
                detected_numbers = number_detector.detect_numbers(image_path)
                
                if detected_numbers:
                    for number, confidence_score in detected_numbers:
                        # Skip if target numbers specified and this number not in list
                        if target_numbers and number not in target_numbers:
                            continue
                        
                        # Skip if confidence too low
                        if confidence_score < confidence:
                            continue
                        
                        # Add to category
                        category_name = f"Number_{number:02d}"
                        if category_name not in number_categories:
                            number_categories[category_name] = []
                        number_categories[category_name].append(image_path)
                        
                        if verbose:
                            click.echo(f"  {image_path.name}: Number {number} (confidence: {confidence_score:.2f})")
                
                proc_logger.log_progress(i + 1)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                proc_logger.log_error(e, image_path)
    
    # Display results
    click.echo(f"\nNumber Detection Results:")
    total_categorized = 0
    for category_name, files in sorted(number_categories.items()):
        number = int(category_name.split('_')[1])
        click.echo(f"  Number {number:2d}: {len(files)} files")
        total_categorized += len(files)
    
    click.echo(f"Total categorized: {total_categorized}/{len(image_files)}")
    
    if not number_categories:
        click.echo("No numbers detected above confidence threshold")
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
            categories=number_categories,
            use_symlinks=create_symlinks,
            dry_run=False
        )
        
        click.echo(f"Organization complete:")
        click.echo(f"  Files organized: {stats['organized']}")
        click.echo(f"  Errors: {stats['errors']}")
    else:
        click.echo(f"\nDry run - would organize {total_categorized} files into:")
        for category_name, files in sorted(number_categories.items()):
            number = int(category_name.split('_')[1])
            click.echo(f"  {output}/Number_{number:02d}/ ({len(files)} files)")


@click.command()
@click.option('--input', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input directory containing photos')
@click.option('--confidence',
              type=float,
              default=0.7,
              help='Number detection confidence threshold (0.0-1.0)')
@click.option('--numbers',
              type=str,
              help='Comma-separated list of numbers to analyze')
@click.option('--min-number',
              type=int,
              default=1,
              help='Minimum jersey number to detect')
@click.option('--max-number',
              type=int,
              default=99,
              help='Maximum jersey number to detect')
@click.option('--output-format',
              type=click.Choice(['table', 'json', 'csv']),
              default='table',
              help='Output format for analysis results')
def analyze_numbers(input: Path,
                    confidence: float,
                    numbers: Optional[str],
                    min_number: int,
                    max_number: int,
                    output_format: str):
    """
    Analyze number distribution in photos without organizing files.
    
    This command analyzes photos to show number distribution statistics
    without actually organizing the files.
    """
    # Parse numbers if specified
    target_numbers = None
    if numbers:
        try:
            target_numbers = [int(num.strip()) for num in numbers.split(',')]
        except ValueError:
            click.echo("Error: Invalid number format. Use comma-separated integers", err=True)
            return
    
    # Initialize components
    file_utils = FileUtils()
    number_detector = NumberDetector(
        confidence_threshold=confidence,
        min_number=min_number,
        max_number=max_number
    )
    
    # Find image files
    image_files = file_utils.find_image_files(input, recursive=True)
    if not image_files:
        click.echo(f"No image files found in {input}", err=True)
        return
    
    click.echo(f"Analyzing {len(image_files)} image files...")
    
    # Analyze numbers
    number_stats = {}
    processed_count = 0
    
    for image_path in image_files:
        try:
            detected_numbers = number_detector.detect_numbers(image_path)
            
            if detected_numbers:
                for number, confidence_score in detected_numbers:
                    if target_numbers and number not in target_numbers:
                        continue
                    
                    if confidence_score >= confidence:
                        if number not in number_stats:
                            number_stats[number] = {
                                'count': 0,
                                'total_confidence': 0.0,
                                'files': []
                            }
                        
                        number_stats[number]['count'] += 1
                        number_stats[number]['total_confidence'] += confidence_score
                        number_stats[number]['files'].append({
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
        click.echo(f"\nNumber Analysis Results:")
        click.echo("=" * 50)
        
        for number in sorted(number_stats.keys()):
            stats = number_stats[number]
            avg_confidence = stats['total_confidence'] / stats['count']
            click.echo(f"Number {number:2d} | {stats['count']:4} files | Avg confidence: {avg_confidence:.2f}")
    
    elif output_format == 'json':
        import json
        results = {
            'total_files': len(image_files),
            'processed_files': processed_count,
            'numbers': number_stats
        }
        click.echo(json.dumps(results, indent=2))
    
    elif output_format == 'csv':
        click.echo("Number,Count,Avg_Confidence")
        for number in sorted(number_stats.keys()):
            stats = number_stats[number]
            avg_confidence = stats['total_confidence'] / stats['count']
            click.echo(f"{number},{stats['count']},{avg_confidence:.2f}")
    
    click.echo(f"\nAnalysis complete: {processed_count}/{len(image_files)} files processed")


@click.command()
@click.option('--input', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input directory containing photos')
@click.option('--number',
              type=int,
              required=True,
              help='Specific jersey number to find')
@click.option('--confidence',
              type=float,
              default=0.7,
              help='Number detection confidence threshold (0.0-1.0)')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for photos of this number')
@click.option('--copy/--symlink',
              default=False,
              help='Copy files instead of creating symlinks')
def find_number(input: Path,
                number: int,
                confidence: float,
                output: Optional[Path],
                copy: bool):
    """
    Find all photos containing a specific jersey number.
    
    This command searches for photos containing a specific jersey number
    and optionally copies them to a separate directory.
    
    Examples:
    
        # Find all photos with number 15
        soccer-photo-sorter find-number -i ./photos --number 15
        
        # Find and copy photos with number 7
        soccer-photo-sorter find-number -i ./photos --number 7 -o ./player_7 --copy
    """
    # Initialize components
    file_utils = FileUtils()
    number_detector = NumberDetector(confidence_threshold=confidence)
    
    # Find image files
    image_files = file_utils.find_image_files(input, recursive=True)
    if not image_files:
        click.echo(f"No image files found in {input}", err=True)
        return
    
    click.echo(f"Searching for number {number} in {len(image_files)} files...")
    
    # Find files with the number
    matching_files = []
    
    for image_path in image_files:
        try:
            detected_numbers = number_detector.detect_numbers(image_path)
            
            if detected_numbers:
                for detected_number, confidence_score in detected_numbers:
                    if detected_number == number and confidence_score >= confidence:
                        matching_files.append((image_path, confidence_score))
                        break
                        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    # Display results
    click.echo(f"\nFound {len(matching_files)} files with number {number}:")
    
    if matching_files:
        for file_path, conf in matching_files:
            click.echo(f"  {file_path.name} (confidence: {conf:.2f})")
        
        # Copy files if output directory specified
        if output:
            click.echo(f"\nCopying files to {output}...")
            output.mkdir(parents=True, exist_ok=True)
            
            copied_count = 0
            for file_path, _ in matching_files:
                destination = output / file_path.name
                if copy:
                    success = file_utils.copy_file(file_path, destination)
                else:
                    success = file_utils.create_symlink(file_path, destination)
                
                if success:
                    copied_count += 1
            
            click.echo(f"Copied {copied_count}/{len(matching_files)} files")
    else:
        click.echo(f"No files found with number {number} above confidence threshold {confidence}")
