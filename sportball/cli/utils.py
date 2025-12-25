"""
CLI Utilities

Utility functions for the sportball CLI.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
import json
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
    if "core" not in ctx.obj:
        # Lazy import to avoid heavy imports at startup
        from ..core import SportballCore

        ctx.obj["core"] = SportballCore(
            base_dir=ctx.obj.get("base_dir"),
            enable_gpu=ctx.obj.get("gpu", True),
            max_workers=ctx.obj.get("workers"),
            cache_enabled=ctx.obj.get("cache", True),
            verbose=ctx.obj.get("verbose", False),
        )

    return ctx.obj["core"]


def find_image_files(input_path: Path, recursive: bool = True) -> List[Path]:
    """
    Find image files in a directory or return single file.

    Args:
        input_path: Path to file or directory
        recursive: Whether to search recursively in subdirectories

    Returns:
        List of image file paths
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

    if input_path.is_file():
        # Validate that the file has an image extension
        if input_path.suffix.lower() in image_extensions:
            return [input_path]
        else:
            # Not an image file, return empty list
            return []

    if not input_path.is_dir():
        raise click.BadParameter(f"Path does not exist: {input_path}")

    image_files = []

    # Directories to exclude from search
    exclude_dirs = {"venv", "__pycache__", ".git", ".sportball_cache", "node_modules"}

    if recursive:
        # Recursive search with exclusions
        for ext in image_extensions:
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                for file_path in input_path.rglob(pattern):
                    # Skip files in excluded directories
                    if any(
                        exclude_dir in file_path.parts for exclude_dir in exclude_dirs
                    ):
                        continue
                    image_files.append(file_path)
    else:
        # Non-recursive search
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    return sorted(list(set(image_files)))


def check_sidecar_file_parallel(
    image_file: Path, force: bool, operation_type: str = "face_detection"
) -> Tuple[Path, bool]:
    """
    Check if a sidecar file exists and contains detection data for a specific operation.

    Args:
        image_file: Path to the image file
        force: Whether to force processing even if sidecar exists
        operation_type: Type of operation to check for ("face_detection", "object_detection", etc.)

    Returns:
        Tuple of (image_file, should_skip)
    """
    if force:
        return (image_file, False)  # Force processing, don't skip
    
    try:
        # Use Rust sidecar manager directly - it handles all format detection (.bin, .rkyv, .json)
        from ..detection.rust_sidecar import RustSidecarManager
        
        rust_manager = RustSidecarManager()
        if not rust_manager.rust_available:
            # If Rust is not available, process the file (can't check sidecar)
            return (image_file, False)
        
        # Rust handles finding and reading sidecar files in any format
        data = rust_manager.read_data(str(image_file))
        
        if not data:
            return (image_file, False)  # No sidecar data found, should process
        
        # Check for face detection data
        if operation_type == "face_detection":
            face_data = data.get("face_detection")
            if face_data:
                # Check for unified structure
                if "unified" in face_data:
                    unified = face_data["unified"]
                    faces = unified.get("faces", [])
                    if len(faces) > 0:
                        return (image_file, True)  # Should skip (already processed)
                # Check for direct faces
                elif "faces" in face_data:
                    faces = face_data["faces"]
                    if len(faces) > 0:
                        return (image_file, True)  # Should skip (already processed)
                # Check success flag
                elif face_data.get("success", False):
                    return (image_file, True)  # Should skip (already processed)

        # Check for object detection data (YOLOv8)
        elif operation_type == "object_detection":
            yolov8_data = data.get("yolov8")
            if yolov8_data:
                success = yolov8_data.get("success", False)
                objects = yolov8_data.get("objects", [])
                if success or len(objects) > 0:
                    return (image_file, True)  # Should skip (already processed)

        # Check for other operation types
        elif operation_type in data:
            op_data = data[operation_type]
            if isinstance(op_data, dict) and op_data.get("success", False):
                return (image_file, True)  # Should skip (already processed)

        return (image_file, False)  # Should process

    except Exception:
        return (image_file, False)  # Should process on error


def check_sidecar_files_parallel(
    image_files: List[Path],
    force: bool,
    operation_type: str = "face_detection",
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    show_progress: bool = True,
    use_rust: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """
    Check sidecar files in parallel to determine which files should be processed.

    Args:
        image_files: List of image file paths to check
        force: Whether to force processing even if sidecar exists
        operation_type: Type of operation to check for
        max_workers: Maximum number of parallel workers (defaults to CPU count)
        use_processes: Whether to use ProcessPoolExecutor instead of ThreadPoolExecutor
        show_progress: Whether to show a progress bar during processing
        use_rust: Whether to use Rust implementation if available

    Returns:
        Tuple of (files_to_process, skipped_files)
    """
    if not image_files:
        return [], []

    # Try Rust implementation first if available and requested
    if use_rust:
        try:
            # Lazy import to avoid heavy dependencies at startup
            from ..detection.integration import DetectionIntegration

            detection = DetectionIntegration(max_workers=max_workers)

            if detection.rust_module and detection.rust_module.rust_available:
                # Use Rust implementation for massively parallel validation
                validation_results = detection.validate_sidecar_files(
                    image_files[0].parent if image_files else Path.cwd(),
                    operation_type=operation_type,
                    use_rust=True,
                )

                # Convert validation results to file lists
                files_to_process = []
                skipped_files = []

                for result in validation_results:
                    file_path = Path(result["file_path"])
                    if result["is_valid"] and not force:
                        skipped_files.append(file_path)
                    else:
                        files_to_process.append(file_path)

                return files_to_process, skipped_files

        except Exception:
            # Fall back to Python implementation
            pass

    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(
            cpu_count(), len(image_files), 32
        )  # Cap at 32 to avoid overhead

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
            executor.submit(
                check_sidecar_file_parallel, image_file, force, operation_type
            ): image_file
            for image_file in image_files
        }

        # Collect results as they complete with progress bar
        if show_progress:
            progress_bar = tqdm(
                total=len(image_files),
                desc="Checking sidecar files",
                unit="files",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

        for future in as_completed(future_to_file):
            try:
                image_file, should_skip = future.result()
                if should_skip:
                    skipped_files.append(image_file)
                else:
                    files_to_process.append(image_file)
            except Exception:
                # If checking fails, process the file to be safe
                original_file = future_to_file[future]
                files_to_process.append(original_file)

            if show_progress:
                # Update progress bar with current file name
                progress_bar.set_postfix(file=image_file.name)
                progress_bar.update(1)

        if show_progress:
            progress_bar.close()

    return files_to_process, skipped_files


def check_sidecar_files(
    image_files: List[Path],
    force: bool,
    operation_type: str = "face_detection",
    use_parallel: bool = True,
    show_progress: bool = True,
    use_rust: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """
    High-level function to check sidecar files and determine which files should be processed.

    This is the main entry point for sidecar checking that all commands should use.
    It automatically chooses the best approach (Rust vs Python, parallel vs sequential) based on availability and file count.

    Args:
        image_files: List of image file paths to check
        force: Whether to force processing even if sidecar exists
        operation_type: Type of operation to check for ("face_detection", "object_detection", etc.)
        use_parallel: Whether to use parallel processing (default: True)
        show_progress: Whether to show a progress bar during processing
        use_rust: Whether to use Rust implementation if available (default: True)

    Returns:
        Tuple of (files_to_process, skipped_files)
    """
    if not image_files:
        return [], []

    # For small numbers of files, sequential might be faster due to overhead
    # For large numbers, parallel is definitely faster
    if use_parallel and len(image_files) >= 50:
        return check_sidecar_files_parallel(
            image_files,
            force,
            operation_type,
            use_processes=True,  # Use ProcessPoolExecutor for better I/O performance
            show_progress=show_progress,
            use_rust=use_rust,
        )
    else:
        # Sequential processing for small file sets
        files_to_process = []
        skipped_files = []

        if show_progress:
            try:
                from tqdm import tqdm

                progress_bar = tqdm(
                    total=len(image_files),
                    desc="Checking sidecar files",
                    unit="files",
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                )
            except ImportError:
                progress_bar = None
        else:
            progress_bar = None

        for image_file in image_files:
            _, should_skip = check_sidecar_file_parallel(
                image_file, force, operation_type
            )
            if should_skip:
                skipped_files.append(image_file)
            else:
                files_to_process.append(image_file)

            if progress_bar:
                # Update progress bar with current file name
                progress_bar.set_postfix(file=image_file.name)
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        return files_to_process, skipped_files
