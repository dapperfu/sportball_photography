"""
Sportball CLI Main Module

Main command-line interface for the sportball package.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from loguru import logger

from ..core import SportballCore
# Import commands after CLI is defined to avoid circular imports

# Configure rich console
console = Console()


@click.group()
@click.option('--base-dir', '-d', 
              type=click.Path(path_type=Path),
              help='Base directory for operations')
@click.option('--gpu/--no-gpu', 
              default=True,
              help='Enable/disable GPU acceleration')
@click.option('--workers', '-w',
              type=int,
              help='Number of parallel workers')
@click.option('--cache/--no-cache',
              default=True,
              help='Enable/disable result caching')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose logging')
@click.option('--quiet', '-q',
              is_flag=True,
              help='Suppress output except errors')
@click.version_option(version='1.0.0')
@click.pass_context
def cli(ctx: click.Context, 
        base_dir: Optional[Path], 
        gpu: bool, 
        workers: Optional[int], 
        cache: bool,
        verbose: bool, 
        quiet: bool):
    """
    Sportball - Unified Sports Photo Analysis Package
    
    A comprehensive tool for analyzing and organizing sports photographs
    using computer vision, machine learning, and AI techniques.
    
    Features:
    - Face detection and recognition
    - Object detection and extraction (including balls)
    - Game boundary detection
    - Photo quality assessment
    - Sidecar file management and statistics
    - Parallel processing with GPU support
    
    Examples:
    
    \b
    # Detect faces in images
    sportball face detect /path/to/images
    
    \b
    # Extract objects from images
    sportball object extract /path/to/images --output /path/to/output
    
    \b
    # Split photos into games
    sportball games split /path/to/photos --output /path/to/games
    
    \b
    # Detect balls specifically
    sportball object detect /path/to/images --classes "sports ball"
    
    \b
    # Assess photo quality
    sportball quality assess /path/to/images
    
    # Analyze sidecar files
    sportball sidecar stats /path/to/images
    """
    
    # Configure logging
    if verbose:
        logger.add("sportball.log", level="DEBUG", rotation="10 MB")
        logger.info("Verbose logging enabled")
    elif quiet:
        logger.remove()
        logger.add(lambda msg: None, level="ERROR")
    
    # Store configuration in context
    ctx.ensure_object(dict)
    ctx.obj['base_dir'] = base_dir
    ctx.obj['gpu'] = gpu
    ctx.obj['workers'] = workers
    ctx.obj['cache'] = cache
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    
    # Display header if not quiet
    if not quiet:
        console.print(Panel.fit(
            "[bold blue]Sportball[/bold blue] - Unified Sports Photo Analysis\n"
            "AI-powered sports photo processing and organization",
            border_style="blue"
        ))


# Add command groups (import here to avoid circular imports)
from .commands import (
    face_commands,
    object_commands,
    game_commands,
    quality_commands,
    utility_commands,
    sidecar_commands
)

cli.add_command(face_commands.face_group, name='face')
cli.add_command(object_commands.object_group, name='object')
cli.add_command(game_commands.game_group, name='games')
cli.add_command(quality_commands.quality_group, name='quality')
cli.add_command(utility_commands.utility_group, name='util')
cli.add_command(sidecar_commands.sidecar_group, name='sidecar')




if __name__ == '__main__':
    cli()
