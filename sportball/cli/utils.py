"""
CLI Utilities

Utility functions for the sportball CLI.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
# Lazy import SportballCore to avoid heavy imports at startup
# from ..core import SportballCore


def get_core(ctx: click.Context):
    """
    Get or create SportballCore instance from context.
    
    Args:
        ctx: Click context
        
    Returns:
        SportballCore instance
    """
    if 'core' not in ctx.obj:
        # Lazy import to avoid heavy imports at startup
        from ..core import SportballCore
        ctx.obj['core'] = SportballCore(
            base_dir=ctx.obj.get('base_dir'),
            enable_gpu=ctx.obj.get('gpu', True),
            max_workers=ctx.obj.get('workers'),
            cache_enabled=ctx.obj.get('cache', True),
            verbose=ctx.obj.get('verbose', False)
        )
    
    return ctx.obj['core']


def find_image_files(input_path: Path, recursive: bool = True) -> List[Path]:
    """
    Find image files in a directory or return single file.
    
    Args:
        input_path: Path to file or directory
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    if input_path.is_file():
        return [input_path]
    
    if not input_path.is_dir():
        raise click.BadParameter(f"Path does not exist: {input_path}")
    
    image_files = []
    
    # Directories to exclude from search
    exclude_dirs = {'venv', '__pycache__', '.git', '.sportball_cache', 'node_modules'}
    
    if recursive:
        # Recursive search with exclusions
        for ext in image_extensions:
            for pattern in [f'*{ext}', f'*{ext.upper()}']:
                for file_path in input_path.rglob(pattern):
                    # Skip files in excluded directories
                    if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                        continue
                    image_files.append(file_path)
    else:
        # Non-recursive search
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    return sorted(list(set(image_files)))


def check_sidecar_file_parallel(image_file: Path, force: bool, operation_type: str = "face_detection") -> Tuple[Path, bool]:
    """
    Check if a sidecar file exists and contains detection data for a specific operation.
    
    Args:
        image_file: Path to the image file
        force: Whether to force processing even if sidecar exists
        operation_type: Type of operation to check for ("face_detection", "object_detection", etc.)
        
    Returns:
        Tuple of (image_file, should_skip)
    """
    try:
        # Resolve symlink if needed
        original_image_path = image_file.resolve() if image_file.is_symlink() else image_file
        json_path = original_image_path.parent / f"{original_image_path.stem}.json"
        
        if json_path.exists() and not force:
            # Check if JSON contains detection data for the specific operation
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Check for face detection data
                if operation_type == "face_detection":
                    if ("data" in data and 
                        "metadata" in data["data"] and
                        "extraction_timestamp" in data["data"]["metadata"] and
                        "detector" in data["data"]["metadata"]):
                        return (image_file, True)  # Should skip (already processed)
                
                # Check for object detection data (YOLOv8)
                elif operation_type == "object_detection":
                    if 'yolov8' in data:
                        return (image_file, True)  # Should skip (already processed)
                
                # Check for other operation types
                elif operation_type in data:
                    return (image_file, True)  # Should skip (already processed)
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        
        return (image_file, False)  # Should process
        
    except Exception:
        return (image_file, False)  # Should process on error


def check_sidecar_files_parallel(image_files: List[Path], 
                                force: bool, 
                                operation_type: str = "face_detection",
                                max_workers: Optional[int] = None,
                                use_processes: bool = False,
                                show_progress: bool = True) -> Tuple[List[Path], List[Path]]:
    """
    Check sidecar files in parallel to determine which files should be processed.
    
    Args:
        image_files: List of image file paths to check
        force: Whether to force processing even if sidecar exists
        operation_type: Type of operation to check for
        max_workers: Maximum number of parallel workers (defaults to CPU count)
        use_processes: Whether to use ProcessPoolExecutor instead of ThreadPoolExecutor
        show_progress: Whether to show a progress bar during processing
        
    Returns:
        Tuple of (files_to_process, skipped_files)
    """
    if not image_files:
        return [], []
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(image_files), 32)  # Cap at 32 to avoid overhead
    
    files_to_process = []
    skipped_files = []
    
    # Choose executor based on workload
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    # Import tqdm for progress bar
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            show_progress = False
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(check_sidecar_file_parallel, image_file, force, operation_type): image_file
            for image_file in image_files
        }
        
        # Collect results as they complete with progress bar
        if show_progress:
            progress_bar = tqdm(
                total=len(image_files),
                desc="Checking sidecar files",
                unit="files",
                leave=False
            )
        
        for future in as_completed(future_to_file):
            try:
                image_file, should_skip = future.result()
                if should_skip:
                    skipped_files.append(image_file)
                else:
                    files_to_process.append(image_file)
            except Exception as e:
                # If checking fails, process the file to be safe
                original_file = future_to_file[future]
                files_to_process.append(original_file)
            
            if show_progress:
                progress_bar.update(1)
        
        if show_progress:
            progress_bar.close()
    
    return files_to_process, skipped_files
