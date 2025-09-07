"""
Main CLI module for soccer photo sorter.

This module provides the main Click CLI interface for the soccer photo sorter.
"""

import click
from pathlib import Path
from typing import Optional
from loguru import logger

from ..config.config_loader import ConfigLoader
from ..config.settings import Settings
from ..utils.logging_utils import setup_logging, log_cuda_info
from ..utils.cuda_utils import CudaManager
from ..core.photo_sorter import PhotoSorter


@click.group()
@click.option('--config', '-c', 
              type=click.Path(exists=True, path_type=Path),
              help='Configuration file path')
@click.option('--input', '-i',
              type=click.Path(exists=True, path_type=Path),
              help='Input directory path')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory path')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
@click.option('--log-level',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO',
              help='Logging level')
@click.option('--use-cuda/--no-cuda',
              default=True,
              help='Enable/disable CUDA acceleration')
@click.option('--cpu-only',
              is_flag=True,
              help='Force CPU-only processing')
@click.option('--gpu-memory-limit',
              type=int,
              help='GPU memory limit in GB')
@click.option('--threads', '-t',
              type=int,
              help='Number of processing threads')
@click.option('--dry-run',
              is_flag=True,
              help='Preview changes without creating directories')
@click.pass_context
def cli(ctx: click.Context,
        config: Optional[Path],
        input: Optional[Path],
        output: Optional[Path],
        verbose: bool,
        log_level: str,
        use_cuda: bool,
        cpu_only: bool,
        gpu_memory_limit: Optional[int],
        threads: Optional[int],
        dry_run: bool):
    """
    Soccer Photo Sorter - AI-powered photo organization system.
    
    Automatically sort soccer game photographs based on jersey colors,
    jersey numbers, and player faces using computer vision and machine learning.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Override CUDA setting if CPU-only is specified
    if cpu_only:
        use_cuda = False
    
    # Setup logging
    log_file = Path('logs') / 'soccer_photo_sorter.log' if verbose else None
    setup_logging(log_level, log_file, verbose)
    
    # Load configuration
    config_loader = ConfigLoader(config)
    cli_overrides = {
        'input_path': input,
        'output_path': output,
        'processing.verbose': verbose,
        'processing.log_level': log_level,
        'processing.use_cuda': use_cuda,
        'processing.gpu_memory_limit': gpu_memory_limit,
        'processing.max_threads': threads,
        'dry_run': dry_run,
    }
    
    # Remove None values
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    
    settings = config_loader.load_config(
        file_path=config,
        env_overrides=True,
        cli_overrides=cli_overrides
    )
    
    # Initialize CUDA manager
    cuda_manager = CudaManager(
        memory_limit_gb=settings.processing.gpu_memory_limit
    )
    
    # Log CUDA information
    log_cuda_info(cuda_manager)
    
    # Store in context
    ctx.obj['settings'] = settings
    ctx.obj['cuda_manager'] = cuda_manager
    ctx.obj['config_loader'] = config_loader


@cli.command()
@click.option('--all-methods',
              is_flag=True,
              help='Run all sorting methods')
@click.option('--color-confidence',
              type=float,
              help='Color detection confidence threshold')
@click.option('--number-confidence',
              type=float,
              help='Number detection confidence threshold')
@click.option('--face-confidence',
              type=float,
              help='Face detection confidence threshold')
@click.option('--enable-color/--disable-color',
              default=True,
              help='Enable/disable color detection')
@click.option('--enable-number/--disable-number',
              default=True,
              help='Enable/disable number detection')
@click.option('--enable-face/--disable-face',
              default=True,
              help='Enable/disable face detection')
@click.pass_context
def sort(ctx: click.Context,
         all_methods: bool,
         color_confidence: Optional[float],
         number_confidence: Optional[float],
         face_confidence: Optional[float],
         enable_color: bool,
         enable_number: bool,
         enable_face: bool):
    """
    Sort photos using AI detection methods.
    
    This command processes images in the input directory and organizes them
    into output directories based on detected jersey colors, jersey numbers,
    and player faces.
    """
    settings = ctx.obj['settings']
    cuda_manager = ctx.obj['cuda_manager']
    
    # Update settings with CLI options
    if color_confidence is not None:
        settings.detection.color_confidence = color_confidence
    if number_confidence is not None:
        settings.detection.number_confidence = number_confidence
    if face_confidence is not None:
        settings.detection.face_confidence = face_confidence
    
    # Set processing modes
    if all_methods:
        settings.enable_color_detection = True
        settings.enable_number_detection = True
        settings.enable_face_detection = True
    else:
        settings.enable_color_detection = enable_color
        settings.enable_number_detection = enable_number
        settings.enable_face_detection = enable_face
    
    # Validate paths
    if not settings.input_path:
        click.echo("Error: Input path is required", err=True)
        ctx.exit(1)
    
    if not settings.output_path:
        click.echo("Error: Output path is required", err=True)
        ctx.exit(1)
    
    # Initialize photo sorter
    photo_sorter = PhotoSorter(settings, cuda_manager)
    
    try:
        # Process photos
        results = photo_sorter.process_photos()
        
        # Display results
        click.echo(f"\nProcessing complete!")
        click.echo(f"Files processed: {results.get('total_files', 0)}")
        click.echo(f"Files organized: {results.get('organized_files', 0)}")
        click.echo(f"Errors: {results.get('errors', 0)}")
        
        if settings.dry_run:
            click.echo("\nDry run completed - no files were actually organized")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option('--create-default',
              is_flag=True,
              help='Create default configuration file')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default='config.json',
              help='Output configuration file path')
@click.pass_context
def config(ctx: click.Context, create_default: bool, output: Path):
    """
    Configuration management commands.
    
    Create, validate, or display configuration settings.
    """
    config_loader = ctx.obj['config_loader']
    
    if create_default:
        config_loader.create_default_config_file(output)
        click.echo(f"Default configuration created: {output}")
    else:
        click.echo("Use --create-default to create a default configuration file")


@cli.command()
@click.pass_context
def info(ctx: click.Context):
    """
    Display system information.
    
    Show CUDA availability, device information, and system capabilities.
    """
    cuda_manager = ctx.obj['cuda_manager']
    settings = ctx.obj['settings']
    
    click.echo("Soccer Photo Sorter - System Information")
    click.echo("=" * 50)
    
    # CUDA Information
    click.echo("\nCUDA Information:")
    if cuda_manager.is_available:
        click.echo(f"  CUDA Available: Yes")
        click.echo(f"  Devices: {cuda_manager.device_count}")
        
        for device_id in range(cuda_manager.device_count):
            info = cuda_manager.get_device_info(device_id)
            click.echo(f"    Device {device_id}: {info['name']}")
            click.echo(f"      Memory: {info['memory_total'] / 1024**3:.1f} GB")
            click.echo(f"      Compute Capability: {info['compute_capability']}")
    else:
        click.echo(f"  CUDA Available: No")
    
    # OpenCV CUDA
    click.echo(f"  OpenCV CUDA: {'Yes' if cuda_manager.check_opencv_cuda() else 'No'}")
    
    # Configuration
    click.echo("\nConfiguration:")
    click.echo(f"  Input Path: {settings.input_path}")
    click.echo(f"  Output Path: {settings.output_path}")
    click.echo(f"  Use CUDA: {settings.processing.use_cuda}")
    click.echo(f"  Max Threads: {settings.processing.max_threads}")
    click.echo(f"  Batch Size: {settings.processing.batch_size}")
    
    # Processing Modes
    click.echo("\nProcessing Modes:")
    click.echo(f"  Color Detection: {settings.enable_color_detection}")
    click.echo(f"  Number Detection: {settings.enable_number_detection}")
    click.echo(f"  Face Detection: {settings.enable_face_detection}")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"Unexpected error: {e}", err=True)


if __name__ == '__main__':
    main()
