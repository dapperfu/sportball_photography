"""
Utility Commands

CLI commands for utility operations like cache management, sidecar operations, and system info.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional
# Lazy import: from rich.console import Console
# Lazy import: from rich.table import Table
from rich.panel import Panel
import numpy as np

from ..utils import get_core

console = None  # Will be initialized lazily

def _get_console():
    """Lazy import of Console to avoid heavy imports at startup."""
    from rich.console import Console
    return Console()

def _get_progress():
    """Lazy import of Progress components to avoid heavy imports at startup."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    return Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

def _get_table():
    """Lazy import of Table to avoid heavy imports at startup."""
    from rich.table import Table
    return Table


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def utility_group():
    """Utility commands for cache management and system operations."""
    pass


@utility_group.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--operation', '-o',
              type=click.Choice(['face_detection', 'object_detection', 'ball_detection', 'quality_assessment', 'game_detection']),
              help='Filter by operation type')
@click.pass_context
def sidecar_summary(ctx: click.Context, directory: Path, operation: Optional[str]):
    """
    Show summary of sidecar files in a directory.
    
    DIRECTORY should contain images with sidecar files.
    """
    
    core = get_core(ctx)
    
    _get_console().print(f"📋 Analyzing sidecar files in {directory}...", style="blue")
    
    summary = core.get_sidecar_summary(directory)
    
    if not summary:
        _get_console().print("❌ No sidecar files found", style="red")
        return
    
    # Create summary table
    table = Table(title="Sidecar File Summary")
    table.add_column("Operation Type", style="cyan")
    table.add_column("File Count", style="green", justify="right")
    table.add_column("Percentage", style="yellow", justify="right")
    
    total_files = sum(summary.values())
    
    for op_type, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        if operation and op_type != operation:
            continue
        
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        table.add_row(
            op_type.replace('_', ' ').title(),
            str(count),
            f"{percentage:.1f}%"
        )
    
    _get_console().print(table)
    _get_console().print(f"\n📊 Total sidecar files: {total_files}")


@utility_group.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--dry-run', 'dry_run',
              is_flag=True,
              help='Show what would be deleted without actually deleting')
@click.pass_context
def cleanup_sidecars(ctx: click.Context, directory: Path, dry_run: bool):
    """
    Remove orphaned sidecar files that don't have corresponding image files.
    
    DIRECTORY should contain images and sidecar files.
    """
    
    core = get_core(ctx)
    
    if dry_run:
        _get_console().print(f"🔍 Scanning for orphaned sidecar files in {directory}...", style="blue")
        # TODO: Implement dry run mode
        _get_console().print("Dry run mode not yet implemented", style="yellow")
    else:
        _get_console().print(f"🧹 Cleaning up orphaned sidecar files in {directory}...", style="blue")
        
        removed_count = core.cleanup_orphaned_sidecars(directory)
        
        if removed_count > 0:
            _get_console().print(f"✅ Removed {removed_count} orphaned sidecar files", style="green")
        else:
            _get_console().print("✅ No orphaned sidecar files found", style="green")


@utility_group.command()
@click.pass_context
def clear_cache(ctx: click.Context):
    """
    Clear all cached data.
    """
    
    core = get_core(ctx)
    
    _get_console().print("🗑️  Clearing cache...", style="blue")
    
    core.cleanup_cache()
    
    _get_console().print("✅ Cache cleared successfully", style="green")


@utility_group.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--operation', '-o',
              type=click.Choice(['face_detection', 'object_detection', 'ball_detection', 'quality_assessment', 'game_detection']),
              help='Operation type to delete sidecar files for')
@click.option('--dry-run', 'dry_run',
              is_flag=True,
              help='Show what would be deleted without actually deleting')
@click.pass_context
def delete_sidecars(ctx: click.Context, directory: Path, operation: Optional[str], dry_run: bool):
    """
    Delete sidecar files for specific operations.
    
    DIRECTORY should contain images with sidecar files.
    """
    
    core = get_core(ctx)
    
    if dry_run:
        _get_console().print(f"🔍 Scanning for sidecar files to delete in {directory}...", style="blue")
        # TODO: Implement dry run mode
        _get_console().print("Dry run mode not yet implemented", style="yellow")
    else:
        _get_console().print(f"🗑️  Deleting sidecar files in {directory}...", style="blue")
        
        # TODO: Implement sidecar deletion
        _get_console().print("Sidecar deletion not yet implemented", style="yellow")


@utility_group.command()
@click.pass_context
def gpu_check(ctx: click.Context):
    """
    Check GPU acceleration status and capabilities.
    
    This command provides detailed information about GPU availability,
    CUDA support, and acceleration capabilities for sportball operations.
    """
    
    _get_console().print("🔍 Checking GPU acceleration status...", style="blue")
    
    # Create GPU status table
    gpu_table = Table(title="GPU Acceleration Status")
    gpu_table.add_column("Component", style="cyan")
    gpu_table.add_column("Status", style="green")
    gpu_table.add_column("Details", style="yellow")
    
    # Check PyTorch availability
    try:
        import torch
        gpu_table.add_row("PyTorch", "✅ Installed", f"Version {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_table.add_row("CUDA Support", "✅ Available", f"Version {torch.version.cuda}")
            
            # Get GPU count and details
            device_count = torch.cuda.device_count()
            gpu_table.add_row("GPU Count", "✅ Available", f"{device_count} device(s)")
            
            # Get detailed GPU information
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                gpu_table.add_row(f"GPU {i}", "✅ Active", f"{gpu_name} ({gpu_memory_gb:.1f} GB)")
            
            # Test GPU functionality
            try:
                # Create a simple tensor on GPU
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.mm(test_tensor, test_tensor.t())
                gpu_table.add_row("GPU Operations", "✅ Working", "Tensor operations successful")
                
                # Test memory allocation
                large_tensor = torch.randn(1000, 1000).cuda()
                gpu_table.add_row("Memory Allocation", "✅ Working", "Large tensor allocation successful")
                
            except Exception as e:
                gpu_table.add_row("GPU Operations", "❌ Failed", f"Error: {str(e)[:50]}...")
                
        else:
            gpu_table.add_row("CUDA Support", "❌ Not Available", "CUDA not detected")
            gpu_table.add_row("GPU Count", "❌ None", "No CUDA-capable GPUs found")
            
    except ImportError:
        gpu_table.add_row("PyTorch", "❌ Not Installed", "PyTorch not available")
        gpu_table.add_row("CUDA Support", "❌ Not Available", "PyTorch required for CUDA")
    
    # Check OpenCV GPU support
    try:
        import cv2
        # Check if OpenCV was compiled with CUDA support
        try:
            # Try to access CUDA functions
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                gpu_table.add_row("OpenCV GPU", "✅ Available", f"{cuda_devices} CUDA device(s)")
                
                # Test basic CUDA functionality
                try:
                    # Create a simple GPU matrix
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(np.array([[1, 2], [3, 4]], dtype=np.uint8))
                    gpu_table.add_row("OpenCV CUDA Ops", "✅ Working", "GPU matrix operations successful")
                except Exception as e:
                    gpu_table.add_row("OpenCV CUDA Ops", "❌ Failed", f"GPU operations failed: {str(e)[:30]}...")
            else:
                gpu_table.add_row("OpenCV GPU", "❌ Not Available", "OpenCV compiled without CUDA")
        except AttributeError:
            gpu_table.add_row("OpenCV GPU", "❌ Not Available", "OpenCV compiled without CUDA support")
    except ImportError:
        gpu_table.add_row("OpenCV GPU", "❌ Not Installed", "OpenCV not available")
    except Exception as e:
        gpu_table.add_row("OpenCV GPU", "❌ Error", f"Error checking: {str(e)[:30]}...")
    
    # Check ultralytics GPU support
    try:
        from ultralytics import YOLO
        import torch
        
        # Test if YOLO can use GPU properly
        try:
            # Create a small test model
            model = YOLO('yolov8n.pt')
            
            # Check if model can be moved to GPU
            if torch.cuda.is_available():
                try:
                    # Move model to GPU
                    model.to('cuda')
                    device_info = str(model.device)
                    if 'cuda' in device_info:
                        gpu_table.add_row("YOLO GPU", "✅ Available", f"YOLO using GPU: {device_info}")
                    else:
                        gpu_table.add_row("YOLO GPU", "⚠️  Limited", f"YOLO device: {device_info}")
                except Exception as e:
                    gpu_table.add_row("YOLO GPU", "❌ Failed", f"Failed to move to GPU: {str(e)[:30]}...")
            else:
                gpu_table.add_row("YOLO GPU", "⚠️  Limited", "CUDA not available for YOLO")
        except Exception as e:
            gpu_table.add_row("YOLO GPU", "❌ Error", f"Model loading failed: {str(e)[:30]}...")
    except ImportError:
        gpu_table.add_row("YOLO GPU", "❌ Not Installed", "Ultralytics not available")
    except Exception as e:
        gpu_table.add_row("YOLO GPU", "❌ Error", f"Error checking: {str(e)[:30]}...")
    
    # Check face recognition GPU support
    try:
        import face_recognition
        gpu_table.add_row("Face Recognition", "✅ Available", "Face recognition library loaded")
        # Note: face_recognition doesn't have direct GPU support, but dlib might
        try:
            import dlib
            gpu_table.add_row("Dlib GPU", "✅ Available", "Dlib library loaded")
        except ImportError:
            gpu_table.add_row("Dlib GPU", "❌ Not Installed", "Dlib not available")
    except ImportError:
        gpu_table.add_row("Face Recognition", "❌ Not Installed", "Face recognition not available")
    
    _get_console().print(gpu_table)
    
    # Additional recommendations
    _get_console().print("\n📋 Recommendations:", style="blue")
    
    # Check if GPU is enabled in sportball config
    core = get_core(ctx)
    if core.enable_gpu:
        _get_console().print("✅ GPU acceleration is enabled in sportball configuration", style="green")
    else:
        _get_console().print("⚠️  GPU acceleration is disabled in sportball configuration", style="yellow")
        _get_console().print("   Use --gpu flag to enable GPU acceleration", style="yellow")
    
    # Check for common issues
    try:
        import torch
        if torch.cuda.is_available():
            # Check for memory issues
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if props.total_memory < 2 * 1024**3:  # Less than 2GB
                    _get_console().print(f"⚠️  GPU {i} has limited memory ({props.total_memory / (1024**3):.1f} GB)", style="yellow")
                    _get_console().print("   Consider reducing batch sizes for large operations", style="yellow")
    except ImportError:
        pass
    
    # Performance tips
    _get_console().print("\n💡 Performance Tips:", style="blue")
    _get_console().print("• Use --gpu flag to enable GPU acceleration", style="white")
    _get_console().print("• Install CUDA-enabled PyTorch: pip install torch[cuda]", style="white")
    _get_console().print("• OpenCV GPU support requires building from source with CUDA", style="white")
    _get_console().print("• For full GPU acceleration, build OpenCV with CUDA support", style="white")
    _get_console().print("• Monitor GPU memory usage during large operations", style="white")


@utility_group.command()
@click.pass_context
def system_info(ctx: click.Context):
    """
    Show system information and sportball configuration.
    """
    
    core = get_core(ctx)
    
    # System information
    import platform
    import sys
    import os
    
    info_table = Table(title="System Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Platform", platform.platform())
    info_table.add_row("Python Version", sys.version.split()[0])
    info_table.add_row("CPU Count", str(os.cpu_count()))
    info_table.add_row("Base Directory", str(core.base_dir))
    info_table.add_row("GPU Enabled", "✅" if core.enable_gpu else "❌")
    info_table.add_row("Cache Enabled", "✅" if core.cache_enabled else "❌")
    info_table.add_row("Max Workers", str(core.max_workers) if core.max_workers else "Auto")
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            info_table.add_row("CUDA Available", "✅")
            info_table.add_row("CUDA Version", torch.version.cuda)
            info_table.add_row("GPU Count", str(torch.cuda.device_count()))
        else:
            info_table.add_row("CUDA Available", "❌")
    except ImportError:
        info_table.add_row("PyTorch", "Not Installed")
    
    # Check for other dependencies
    dependencies = [
        ("OpenCV", "cv2"),
        ("PIL", "PIL"),
        ("NumPy", "numpy"),
        ("Click", "click"),
        ("Rich", "rich"),
        ("tqdm", "tqdm"),
        ("face_recognition", "face_recognition"),
        ("ultralytics", "ultralytics")
    ]
    
    for name, module in dependencies:
        try:
            __import__(module)
            info_table.add_row(name, "✅")
        except ImportError:
            info_table.add_row(name, "❌")
    
    _get_console().print(info_table)


@utility_group.command()
@click.option('--max-batch-size', 'max_batch_size',
              type=int,
              default=64,
              help='Maximum batch size to test (default: 64)')
@click.option('--start-batch-size', 'start_batch_size',
              type=int,
              default=1,
              help='Starting batch size for testing (default: 1)')
@click.option('--max-test-images', 'max_test_images',
              type=int,
              default=50,
              help='Maximum number of test images to use (default: 50)')
@click.option('--image-size', 'image_size',
              default='1920x1080',
              help='Size of test images in format WIDTHxHEIGHT (default: 1920x1080)')
@click.option('--test-images', 'test_images_path',
              type=click.Path(exists=True, path_type=Path),
              help='Use existing images for testing instead of synthetic ones')
@click.pass_context
def gpu_tune(ctx: click.Context,
             max_batch_size: int,
             start_batch_size: int,
             max_test_images: int,
             image_size: str,
             test_images_path: Optional[Path]):
    """
    Automatically tune GPU batch size by testing until memory limit is reached.
    
    This command will test different batch sizes to find the optimal setting
    for your GPU hardware, maximizing performance without running out of memory.
    """
    
    _get_console().print("🔧 Starting GPU batch size tuning...", style="blue")
    
    # Parse image size
    try:
        width, height = map(int, image_size.split('x'))
        parsed_image_size = (width, height)
    except ValueError:
        _get_console().print(f"❌ Invalid image size format: {image_size}. Use WIDTHxHEIGHT (e.g., 1920x1080)", style="red")
        return
    
    # Get core and face detector
    core = get_core(ctx)
    face_detector = core.get_face_detector()
    
    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            _get_console().print("❌ CUDA not available. GPU tuning requires CUDA-enabled PyTorch.", style="red")
            return
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        _get_console().print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)", style="green")
        
    except ImportError:
        _get_console().print("❌ PyTorch not available. GPU tuning requires PyTorch.", style="red")
        return
    
    # Prepare test images
    test_image_paths = None
    if test_images_path:
        if test_images_path.is_dir():
            from ..utils import find_image_files
            test_image_paths = find_image_files(test_images_path, recursive=False)
            _get_console().print(f"📁 Using {len(test_image_paths)} images from {test_images_path}", style="blue")
        else:
            test_image_paths = [test_images_path]
            _get_console().print(f"📁 Using single test image: {test_images_path}", style="blue")
    
    # Run GPU tuning
    _get_console().print(f"🧪 Testing batch sizes from {start_batch_size} to {max_batch_size}", style="blue")
    _get_console().print(f"🖼️  Image size: {parsed_image_size[0]}x{parsed_image_size[1]}", style="blue")
    
    try:
        optimal_batch_size = face_detector.tune_gpu_batch_size(
            test_image_paths=test_image_paths,
            max_test_images=max_test_images,
            start_batch_size=start_batch_size,
            max_batch_size=max_batch_size,
            image_size=parsed_image_size
        )
        
        _get_console().print(f"🎯 Optimal GPU batch size: {optimal_batch_size}", style="green")
        _get_console().print(f"💡 You can use this batch size with: --batch-size {optimal_batch_size}", style="blue")
        
        # Show performance improvement estimate
        if optimal_batch_size > 1:
            improvement = (optimal_batch_size - 1) * 100
            _get_console().print(f"📈 Estimated performance improvement: ~{improvement}%", style="green")
        
    except Exception as e:
        _get_console().print(f"❌ GPU tuning failed: {e}", style="red")
        return


@utility_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for converted images')
@click.option('--format', 'output_format',
              type=click.Choice(['jpg', 'png', 'tiff']),
              default='jpg',
              help='Output image format')
@click.option('--quality', '-q',
              type=int,
              default=95,
              help='JPEG quality (1-100)')
@click.option('--resize', '-r',
              type=str,
              help='Resize images (e.g., "1920x1080", "50%")')
@click.pass_context
def convert_images(ctx: click.Context, 
                   input_path: Path, 
                   output: Optional[Path],
                   output_format: str,
                   quality: int,
                   resize: Optional[str]):
    """
    Convert images to different formats and sizes.
    
    INPUT_PATH can be a single image file or a directory containing images.
    """
    
    _get_console().print(f"🔄 Converting images in {input_path}...", style="blue")
    
    # TODO: Implement image conversion
    _get_console().print("Image conversion not yet implemented", style="yellow")


@utility_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for organized images')
@click.option('--by', 'organize_by',
              type=click.Choice(['date', 'size', 'quality', 'faces', 'objects']),
              default='date',
              help='Organization criteria')
@click.option('--copy/--move',
              default=False,
              help='Copy files instead of moving them')
@click.pass_context
def organize(ctx: click.Context, 
             input_path: Path, 
             output: Optional[Path],
             organize_by: str,
             copy: bool):
    """
    Organize images by various criteria.
    
    INPUT_PATH should be a directory containing images.
    OUTPUT_DIR is where organized images will be saved.
    """
    
    _get_console().print(f"📁 Organizing images by {organize_by} in {input_path}...", style="blue")
    
    # TODO: Implement image organization
    _get_console().print("Image organization not yet implemented", style="yellow")
