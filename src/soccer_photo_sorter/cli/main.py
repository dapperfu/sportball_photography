"""
Main CLI module for soccer photo sorter.

This module provides the main Click CLI interface for the soccer photo sorter.
"""

import click
from pathlib import Path
from typing import Optional
import sys
import os

# Add the project root to the Python path so we can import the enhanced_game_organizer
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from enhanced_game_organizer import main as enhanced_organizer_main
except ImportError:
    enhanced_organizer_main = None


@click.group()
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
@click.pass_context
def cli(ctx: click.Context, verbose: bool):
    """
    Soccer Photo Sorter - AI-powered photo organization system.
    
    Automatically sort soccer game photographs based on jersey colors,
    jersey numbers, and player faces using computer vision and machine learning.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--input', '-i', 
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True,
              help='Input directory containing photos')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default=Path('./results/games'),
              help='Output directory for organized games')
@click.option('--pattern', '-p',
              default='*_*',
              help='File pattern to match (e.g., "202509*_*" for Sep 2025, "*_*" for all)')
@click.option('--split-file', '-s',
              type=click.Path(path_type=Path),
              help='Text file with manual splits (one timestamp per line, format: HH:MM:SS)')
@click.option('--copy', 'copy_files',
              is_flag=True,
              help='Copy files instead of creating symlinks')
@click.option('--workers', '-w',
              type=int,
              default=4,
              help='Number of parallel workers')
@click.option('--min-duration',
              type=int,
              default=30,
              help='Minimum game duration in minutes')
@click.option('--min-gap',
              type=int,
              default=10,
              help='Minimum gap to separate games in minutes')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose logging')
@click.option('--quiet', '-q',
              is_flag=True,
              help='Suppress output except errors')
@click.pass_context
def organize(ctx: click.Context, input: Path, output: Path, pattern: str, split_file: Optional[Path], 
             copy_files: bool, workers: int, min_duration: int, min_gap: int, 
             verbose: bool, quiet: bool):
    """
    Organize soccer photos into games using the enhanced game organizer.
    
    This command processes images in the input directory and organizes them
    into game-specific folders based on temporal analysis and optional manual splits.
    """
    if enhanced_organizer_main is None:
        click.echo("Error: Enhanced game organizer not available", err=True)
        ctx.exit(1)
    
    # Prepare arguments for the enhanced organizer
    sys.argv = [
        'enhanced_game_organizer.py',
        '--input', str(input),
        '--output', str(output),
        '--pattern', pattern,
        '--workers', str(workers),
        '--min-duration', str(min_duration),
        '--min-gap', str(min_gap),
    ]
    
    if split_file:
        sys.argv.extend(['--split-file', str(split_file)])
    
    if copy_files:
        sys.argv.append('--copy')
    
    if verbose:
        sys.argv.append('--verbose')
    
    if quiet:
        sys.argv.append('--quiet')
    
    # Run the enhanced organizer
    try:
        exit_code = enhanced_organizer_main()
        ctx.exit(exit_code)
    except Exception as e:
        click.echo(f"Error running organizer: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.pass_context
def info(ctx: click.Context):
    """
    Display system information.
    
    Show available features and system capabilities.
    """
    verbose = ctx.obj.get('verbose', False)
    
    click.echo("Soccer Photo Sorter - System Information")
    click.echo("=" * 50)
    
    # Package Information
    click.echo("\nPackage Information:")
    click.echo(f"  Version: 0.1.0")
    click.echo(f"  Python Version: {sys.version}")
    click.echo(f"  Platform: {sys.platform}")
    
    # Available Features
    click.echo("\nAvailable Features:")
    click.echo(f"  Enhanced Game Organizer: {'Yes' if enhanced_organizer_main else 'No'}")
    
    # Dependencies
    click.echo("\nDependencies:")
    try:
        import cv2
        click.echo(f"  OpenCV: {cv2.__version__}")
    except ImportError:
        click.echo(f"  OpenCV: Not installed")
    
    try:
        import numpy
        click.echo(f"  NumPy: {numpy.__version__}")
    except ImportError:
        click.echo(f"  NumPy: Not installed")
    
    try:
        import torch
        click.echo(f"  PyTorch: {torch.__version__}")
        click.echo(f"  CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        click.echo(f"  PyTorch: Not installed")
    
    try:
        import face_recognition
        click.echo(f"  Face Recognition: Available")
    except ImportError:
        click.echo(f"  Face Recognition: Not installed")
    
    # CLI Information
    click.echo("\nCLI Information:")
    click.echo(f"  Verbose Mode: {verbose}")
    click.echo(f"  Working Directory: {os.getcwd()}")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()