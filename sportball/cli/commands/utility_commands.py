"""
Utility Commands

CLI commands for utility operations like cache management, sidecar operations, and system info.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np

from ..utils import get_core

console = Console()


@click.group()
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
    
    console.print(f"📋 Analyzing sidecar files in {directory}...", style="blue")
    
    summary = core.get_sidecar_summary(directory)
    
    if not summary:
        console.print("❌ No sidecar files found", style="red")
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
    
    console.print(table)
    console.print(f"\n📊 Total sidecar files: {total_files}")


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
        console.print(f"🔍 Scanning for orphaned sidecar files in {directory}...", style="blue")
        # TODO: Implement dry run mode
        console.print("Dry run mode not yet implemented", style="yellow")
    else:
        console.print(f"🧹 Cleaning up orphaned sidecar files in {directory}...", style="blue")
        
        removed_count = core.cleanup_orphaned_sidecars(directory)
        
        if removed_count > 0:
            console.print(f"✅ Removed {removed_count} orphaned sidecar files", style="green")
        else:
            console.print("✅ No orphaned sidecar files found", style="green")


@utility_group.command()
@click.pass_context
def clear_cache(ctx: click.Context):
    """
    Clear all cached data.
    """
    
    core = get_core(ctx)
    
    console.print("🗑️  Clearing cache...", style="blue")
    
    core.cleanup_cache()
    
    console.print("✅ Cache cleared successfully", style="green")


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
        console.print(f"🔍 Scanning for sidecar files to delete in {directory}...", style="blue")
        # TODO: Implement dry run mode
        console.print("Dry run mode not yet implemented", style="yellow")
    else:
        console.print(f"🗑️  Deleting sidecar files in {directory}...", style="blue")
        
        # TODO: Implement sidecar deletion
        console.print("Sidecar deletion not yet implemented", style="yellow")


@utility_group.command()
@click.pass_context
def gpu_check(ctx: click.Context):
    """
    Check GPU acceleration status and capabilities.
    
    This command provides detailed information about GPU availability,
    CUDA support, and acceleration capabilities for sportball operations.
    """
    
    console.print("🔍 Checking GPU acceleration status...", style="blue")
    
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
    
    console.print(gpu_table)
    
    # Additional recommendations
    console.print("\n📋 Recommendations:", style="blue")
    
    # Check if GPU is enabled in sportball config
    core = get_core(ctx)
    if core.enable_gpu:
        console.print("✅ GPU acceleration is enabled in sportball configuration", style="green")
    else:
        console.print("⚠️  GPU acceleration is disabled in sportball configuration", style="yellow")
        console.print("   Use --gpu flag to enable GPU acceleration", style="yellow")
    
    # Check for common issues
    try:
        import torch
        if torch.cuda.is_available():
            # Check for memory issues
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if props.total_memory < 2 * 1024**3:  # Less than 2GB
                    console.print(f"⚠️  GPU {i} has limited memory ({props.total_memory / (1024**3):.1f} GB)", style="yellow")
                    console.print("   Consider reducing batch sizes for large operations", style="yellow")
    except ImportError:
        pass
    
    # Performance tips
    console.print("\n💡 Performance Tips:", style="blue")
    console.print("• Use --gpu flag to enable GPU acceleration", style="white")
    console.print("• Install CUDA-enabled PyTorch: pip install torch[cuda]", style="white")
    console.print("• OpenCV GPU support requires building from source with CUDA", style="white")
    console.print("• For full GPU acceleration, build OpenCV with CUDA support", style="white")
    console.print("• Monitor GPU memory usage during large operations", style="white")


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
    
    console.print(info_table)


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
    
    console.print(f"🔄 Converting images in {input_path}...", style="blue")
    
    # TODO: Implement image conversion
    console.print("Image conversion not yet implemented", style="yellow")


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
    
    console.print(f"📁 Organizing images by {organize_by} in {input_path}...", style="blue")
    
    # TODO: Implement image organization
    console.print("Image organization not yet implemented", style="yellow")
