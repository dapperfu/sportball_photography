"""
Ultra-minimal CLI entry point with zero heavy imports at startup.

This module provides the absolute minimum needed to start the CLI,
with ALL heavy dependencies loaded only when specific commands are used.
"""

import click
import warnings
from pathlib import Path
from typing import Optional

# Suppress annoying deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

# Ultra-minimal CLI group with zero heavy imports
@click.group(context_settings={'help_option_names': ['-h', '--help']})
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
    
    Bash Completion:
    To enable bash completion, add this to your ~/.bashrc or ~/.bash_profile:
    
        # For virtual environment (recommended):
        eval "$(python -m sportball.cli.main --completion-script-bash)"
        
        # Or if sportball is in your PATH:
        eval "$(sportball --completion-script-bash)"
    
    Then restart your shell or run: source ~/.bashrc
    """
    
    # Configure logging only when needed (lazy import)
    if verbose or quiet:
        from loguru import logger
        
        if verbose:
            logger.add("sportball.log", level="DEBUG", rotation="10 MB")
            logger.info("Verbose logging enabled")
        elif quiet:
            logger.remove()
            logger.add(lambda msg: None, level="ERROR")
        else:
            # Default: INFO level, suppress DEBUG messages
            logger.remove()
            logger.add(lambda msg: None, level="INFO")
    
    # Store configuration in context
    ctx.ensure_object(dict)
    ctx.obj['base_dir'] = base_dir
    ctx.obj['gpu'] = gpu
    ctx.obj['workers'] = workers
    ctx.obj['cache'] = cache
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet

# Ultra-minimal command loading - only load when accessed
def _get_command(self, ctx, name):
    """Load commands only when accessed to avoid ANY heavy imports."""
    # Lazy load command groups - each import happens only when needed
    if name == 'face':
        from .commands import face_commands
        return face_commands.face_group
    elif name == 'object':
        from .commands import object_commands
        return object_commands.object_group
    elif name == 'games':
        from .commands import game_commands
        return game_commands.game_group
    elif name == 'quality':
        from .commands import quality_commands
        return quality_commands.quality_group
    elif name == 'util':
        from .commands import utility_commands
        return utility_commands.utility_group
    elif name == 'sidecar':
        from .commands import sidecar_commands
        return sidecar_commands.sidecar_group
    
    return None

# Override command resolution for lazy loading
cli.get_command = _get_command.__get__(cli, type(cli))

def main():
    """Main entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main()
