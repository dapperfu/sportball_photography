"""
Face-based photo sorting CLI command.

This module provides the CLI interface for sorting photos by player faces.
"""

import click
from pathlib import Path
from typing import Optional, List
from loguru import logger

from ..detectors.face_detector import FaceDetector
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
              default=0.75,
              help='Face detection confidence threshold (0.0-1.0)')
@click.option('--min-faces',
              type=int,
              default=1,
              help='Minimum number of faces required in photo')
@click.option('--max-faces',
              type=int,
              default=10,
              help='Maximum number of faces to process per photo')
@click.option('--face-size',
              type=int,
              default=64,
              help='Minimum face size in pixels')
@click.option('--create-symlinks/--no-symlinks',
              default=True,
              help='Create symbolic links instead of copying files')
@click.option('--dry-run',
              is_flag=True,
              help='Preview changes without creating directories')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
def face_sorter(input: Path,
                output: Path,
                confidence: float,
                min_faces: int,
                max_faces: int,
                face_size: int,
                create_symlinks: bool,
                dry_run: bool,
                verbose: bool):
    """
    Sort photos by player faces.
    
    This command uses face recognition to detect and group individual players
    in photos, organizing them into player-specific directories (e.g., Player_A/, Player_B/).
    
    Examples:
    
        # Basic face sorting
        soccer-photo-sorter face-sort -i ./photos -o ./sorted
        
        # Sort with custom confidence threshold
        soccer-photo-sorter face-sort -i ./photos -o ./sorted --confidence 0.8
        
        # Sort photos with at least 2 faces
        soccer-photo-sorter face-sort -i ./photos -o ./sorted --min-faces 2
        
        # Preview changes without organizing
        soccer-photo-sorter face-sort -i ./photos -o ./sorted --dry-run
    """
    # Setup logging
    if verbose:
        logger.remove()
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")
    
    # Validate confidence threshold
    if not 0.0 <= confidence <= 1.0:
        click.echo("Error: Confidence must be between 0.0 and 1.0", err=True)
        return
    
    # Validate face parameters
    if min_faces > max_faces:
        click.echo("Error: min-faces must be less than or equal to max-faces", err=True)
        return
    
    if face_size < 32:
        click.echo("Error: face-size must be at least 32 pixels", err=True)
        return
    
    # Initialize components
    file_utils = FileUtils()
    face_detector = FaceDetector(
        confidence_threshold=confidence,
        min_faces=min_faces,
        max_faces=max_faces,
        face_size=face_size
    )
    
    # Find image files
    image_files = file_utils.find_image_files(input, recursive=True)
    if not image_files:
        click.echo(f"No image files found in {input}", err=True)
        return
    
    click.echo(f"Found {len(image_files)} image files")
    
    # Process images
    with ProcessingLogger("Face Detection", len(image_files)) as proc_logger:
        face_categories = {}
        face_encodings = {}  # Store face encodings for clustering
        
        for i, image_path in enumerate(image_files):
            try:
                # Detect faces
                detected_faces = face_detector.detect_faces(image_path)
                
                if detected_faces:
                    # Check if meets minimum face requirement
                    if len(detected_faces) < min_faces:
                        if verbose:
                            click.echo(f"  {image_path.name}: {len(detected_faces)} faces (below minimum {min_faces})")
                        continue
                    
                    # Process each face
                    for face_info in detected_faces:
                        face_id = face_info['face_id']
                        confidence_score = face_info['confidence']
                        
                        # Skip if confidence too low
                        if confidence_score < confidence:
                            continue
                        
                        # Add to category
                        category_name = f"Player_{face_id}"
                        if category_name not in face_categories:
                            face_categories[category_name] = []
                        face_categories[category_name].append(image_path)
                        
                        # Store face encoding for clustering
                        if face_id not in face_encodings:
                            face_encodings[face_id] = face_info['encoding']
                        
                        if verbose:
                            click.echo(f"  {image_path.name}: Player {face_id} (confidence: {confidence_score:.2f})")
                
                proc_logger.log_progress(i + 1)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                proc_logger.log_error(e, image_path)
    
    # Perform face clustering to merge similar faces
    click.echo("\nClustering similar faces...")
    clustered_categories = face_detector.cluster_faces(face_categories, face_encodings)
    
    # Display results
    click.echo(f"\nFace Detection Results:")
    total_categorized = 0
    for category_name, files in sorted(clustered_categories.items()):
        player_id = category_name.split('_')[1]
        click.echo(f"  Player {player_id}: {len(files)} files")
        total_categorized += len(files)
    
    click.echo(f"Total categorized: {total_categorized}/{len(image_files)}")
    
    if not clustered_categories:
        click.echo("No faces detected above confidence threshold")
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
            categories=clustered_categories,
            use_symlinks=create_symlinks,
            dry_run=False
        )
        
        click.echo(f"Organization complete:")
        click.echo(f"  Files organized: {stats['organized']}")
        click.echo(f"  Errors: {stats['errors']}")
    else:
        click.echo(f"\nDry run - would organize {total_categorized} files into:")
        for category_name, files in sorted(clustered_categories.items()):
            player_id = category_name.split('_')[1]
            click.echo(f"  {output}/Player_{player_id}/ ({len(files)} files)")


@click.command()
@click.option('--input', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input directory containing photos')
@click.option('--confidence',
              type=float,
              default=0.75,
              help='Face detection confidence threshold (0.0-1.0)')
@click.option('--min-faces',
              type=int,
              default=1,
              help='Minimum number of faces required in photo')
@click.option('--max-faces',
              type=int,
              default=10,
              help='Maximum number of faces to process per photo')
@click.option('--face-size',
              type=int,
              default=64,
              help='Minimum face size in pixels')
@click.option('--output-format',
              type=click.Choice(['table', 'json', 'csv']),
              default='table',
              help='Output format for analysis results')
def analyze_faces(input: Path,
                  confidence: float,
                  min_faces: int,
                  max_faces: int,
                  face_size: int,
                  output_format: str):
    """
    Analyze face distribution in photos without organizing files.
    
    This command analyzes photos to show face distribution statistics
    without actually organizing the files.
    """
    # Initialize components
    file_utils = FileUtils()
    face_detector = FaceDetector(
        confidence_threshold=confidence,
        min_faces=min_faces,
        max_faces=max_faces,
        face_size=face_size
    )
    
    # Find image files
    image_files = file_utils.find_image_files(input, recursive=True)
    if not image_files:
        click.echo(f"No image files found in {input}", err=True)
        return
    
    click.echo(f"Analyzing {len(image_files)} image files...")
    
    # Analyze faces
    face_stats = {}
    processed_count = 0
    total_faces = 0
    
    for image_path in image_files:
        try:
            detected_faces = face_detector.detect_faces(image_path)
            
            if detected_faces and len(detected_faces) >= min_faces:
                for face_info in detected_faces:
                    face_id = face_info['face_id']
                    confidence_score = face_info['confidence']
                    
                    if confidence_score >= confidence:
                        if face_id not in face_stats:
                            face_stats[face_id] = {
                                'count': 0,
                                'total_confidence': 0.0,
                                'files': []
                            }
                        
                        face_stats[face_id]['count'] += 1
                        face_stats[face_id]['total_confidence'] += confidence_score
                        face_stats[face_id]['files'].append({
                            'file': image_path.name,
                            'confidence': confidence_score
                        })
                        
                        total_faces += 1
            
            processed_count += 1
            if processed_count % 10 == 0:
                click.echo(f"Processed {processed_count}/{len(image_files)} files...")
                
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
    
    # Display results
    if output_format == 'table':
        click.echo(f"\nFace Analysis Results:")
        click.echo("=" * 50)
        click.echo(f"Total faces detected: {total_faces}")
        click.echo(f"Unique players: {len(face_stats)}")
        click.echo("-" * 50)
        
        for face_id in sorted(face_stats.keys()):
            stats = face_stats[face_id]
            avg_confidence = stats['total_confidence'] / stats['count']
            click.echo(f"Player {face_id:3s} | {stats['count']:4} files | Avg confidence: {avg_confidence:.2f}")
    
    elif output_format == 'json':
        import json
        results = {
            'total_files': len(image_files),
            'processed_files': processed_count,
            'total_faces': total_faces,
            'unique_players': len(face_stats),
            'faces': face_stats
        }
        click.echo(json.dumps(results, indent=2))
    
    elif output_format == 'csv':
        click.echo("Player_ID,Count,Avg_Confidence")
        for face_id in sorted(face_stats.keys()):
            stats = face_stats[face_id]
            avg_confidence = stats['total_confidence'] / stats['count']
            click.echo(f"{face_id},{stats['count']},{avg_confidence:.2f}")
    
    click.echo(f"\nAnalysis complete: {processed_count}/{len(image_files)} files processed")


@click.command()
@click.option('--input', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input directory containing photos')
@click.option('--reference', '-r',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Reference photo containing the player to find')
@click.option('--confidence',
              type=float,
              default=0.75,
              help='Face matching confidence threshold (0.0-1.0)')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for matching photos')
@click.option('--copy/--symlink',
              default=False,
              help='Copy files instead of creating symlinks')
def find_player(input: Path,
                reference: Path,
                confidence: float,
                output: Optional[Path],
                copy: bool):
    """
    Find all photos containing a specific player.
    
    This command uses a reference photo to find all photos containing
    the same player and optionally copies them to a separate directory.
    
    Examples:
    
        # Find all photos with the player from reference.jpg
        soccer-photo-sorter find-player -i ./photos -r ./reference.jpg
        
        # Find and copy photos with the player
        soccer-photo-sorter find-player -i ./photos -r ./reference.jpg -o ./player_photos --copy
    """
    # Initialize components
    file_utils = FileUtils()
    face_detector = FaceDetector(confidence_threshold=confidence)
    
    # Find image files
    image_files = file_utils.find_image_files(input, recursive=True)
    if not image_files:
        click.echo(f"No image files found in {input}", err=True)
        return
    
    click.echo(f"Analyzing reference photo: {reference}")
    
    # Get reference face encoding
    try:
        reference_faces = face_detector.detect_faces(reference)
        if not reference_faces:
            click.echo("No faces detected in reference photo", err=True)
            return
        
        # Use the first face as reference
        reference_encoding = reference_faces[0]['encoding']
        click.echo(f"Reference face encoding extracted")
        
    except Exception as e:
        click.echo(f"Error processing reference photo: {e}", err=True)
        return
    
    click.echo(f"Searching for matching player in {len(image_files)} files...")
    
    # Find matching files
    matching_files = []
    
    for image_path in image_files:
        try:
            detected_faces = face_detector.detect_faces(image_path)
            
            if detected_faces:
                for face_info in detected_faces:
                    # Compare with reference encoding
                    match_confidence = face_detector.compare_faces(
                        reference_encoding, 
                        face_info['encoding']
                    )
                    
                    if match_confidence >= confidence:
                        matching_files.append((image_path, match_confidence))
                        break  # Found a match, no need to check other faces
                        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    # Display results
    click.echo(f"\nFound {len(matching_files)} files with matching player:")
    
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
        click.echo(f"No files found with matching player above confidence threshold {confidence}")
