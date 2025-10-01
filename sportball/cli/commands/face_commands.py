"""
Face Detection Commands

CLI commands for face detection and recognition operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
import signal
import sys
from pathlib import Path
from typing import Optional

# Import shared utilities to avoid duplication
from ..shared_utils import get_console, get_progress_components, get_table, setup_verbose_logging, check_and_display_sidecar_status, display_processing_start


console = None  # Will be initialized lazily

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    _get_console().print(
        "\n🛑 Shutdown requested. Finishing current operations...", style="yellow"
    )
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def face_group():
    """Face detection and recognition commands."""
    pass


@face_group.command()
@click.argument("input_pattern", type=str)
@click.option(
    "--border-padding",
    "-b",
    default=0.25,
    help="Border padding percentage (0.25 = 25%)",
)
@click.option(
    "--max-images",
    "-m",
    default=None,
    type=int,
    help="Maximum number of images to process",
)
@click.option("--gpu/--no-gpu", default=True, help="Use GPU acceleration if available")
@click.option(
    "--force", "-f", is_flag=True, help="Force detection even if JSON sidecar exists"
)
@click.option(
    "--no-recursive",
    "no_recursive",
    is_flag=True,
    help="Disable recursive directory processing",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Enable verbose logging (-v for info, -vv for debug)",
)
@click.option(
    "--workers", "-w", type=int, help="Number of parallel workers (default: auto)"
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=8,
    help="Processing batch size (legacy parameter, not used in sequential mode)",
)
@click.option(
    "--auto-tune",
    "auto_tune",
    is_flag=True,
    help="Automatically tune processing parameters for optimal performance",
)
@click.option(
    "--show-empty-results",
    "show_empty_results",
    is_flag=True,
    help="Show output even when no faces are detected (default: suppress empty results)",
)
@click.pass_context
def detect(
    ctx: click.Context,
    input_pattern: str,
    border_padding: float,
    max_images: Optional[int],
    gpu: bool,
    force: bool,
    no_recursive: bool,
    verbose: int,
    workers: Optional[int],
    batch_size: int,
    auto_tune: bool,
    show_empty_results: bool,
):
    """
    Detect faces in images and save comprehensive data to JSON sidecar files.

    INPUT_PATTERN can be a file pattern, directory path, or single image file.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """

    # Setup logging based on verbose level
    setup_verbose_logging(verbose)

    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core

    core = get_core(ctx)

    # Find image files
    input_path = Path(input_pattern)
    recursive = not no_recursive

    if input_path.is_dir():
        # Lazy import to avoid heavy dependencies at startup
        from ..utils import find_image_files

        image_files = find_image_files(input_path, recursive=recursive)
    else:
        # Pattern matching
        if input_pattern.startswith("/"):
            parent_dir = Path(input_pattern).parent
            pattern = Path(input_pattern).name
            image_files = list(parent_dir.glob(pattern))
        else:
            image_files = list(Path(".").glob(input_pattern))

    if not image_files:
        _get_console().print("❌ No images found", style="red")
        return

    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]

    _get_console().print(f"📊 Found {len(image_files)} images to analyze", style="blue")

    # Auto-tune processing parameters if requested
    if auto_tune and gpu:
        _get_console().print("🔧 Auto-tuning processing parameters...", style="blue")
        try:
            # For sequential processing, we don't need batch size tuning
            # This is kept for compatibility but doesn't affect performance
            _get_console().print(
                "ℹ️  Sequential processing mode - batch size tuning not needed",
                style="blue",
            )
        except Exception as e:
            _get_console().print(f"⚠️  Auto-tuning failed: {e}", style="yellow")

    # Check for existing sidecar files
    get_console().print("🔍 Checking for existing sidecar files...", style="blue")
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import check_sidecar_files_parallel

    files_to_process, skipped_files = check_sidecar_files_parallel(
        image_files, force, operation_type="face_detection"
    )

    # Display sidecar status
    check_and_display_sidecar_status(files_to_process, skipped_files, force, "face_detection")
    
    if not files_to_process:
        return

    # Use core's sequential processing for face detection
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core

    core = get_core(ctx)

    # Get model and GPU information before starting detection
    # Create a fresh detector with the correct GPU setting from CLI
    from ...detectors.face import InsightFaceDetector

    test_detector = InsightFaceDetector(
        enable_gpu=gpu, cache_enabled=core.cache_enabled, verbose=core.verbose
    )
    model_info = test_detector.get_model_info()

    # Display enhanced startup message
    console = _get_console()
    console.print("🔍 Starting face detection...", style="blue")

    # Display model information
    model_name = model_info.get("model", "unknown")
    device = model_info.get("device", "cpu")
    gpu_enabled = model_info.get("gpu_enabled", False)
    gpu_test_passed = model_info.get("gpu_test_passed", False)

    if gpu_enabled and gpu_test_passed:
        gpu_memory = model_info.get("gpu_memory_gb", 0)
        console.print(
            f"📱 Model: {model_name} | Device: {device} | GPU: ✅ Available ({gpu_memory}GB)",
            style="green",
        )
    elif gpu_enabled and not gpu_test_passed:
        gpu_error = model_info.get("gpu_test_error", "Unknown error")
        console.print(
            f"📱 Model: {model_name} | Device: {device} | GPU: ❌ Failed ({gpu_error})",
            style="yellow",
        )
    else:
        console.print(
            f"📱 Model: {model_name} | Device: {device} | GPU: Disabled", style="blue"
        )

    # Prepare detection parameters
    detection_kwargs = {
        "confidence": 0.5,
        "min_faces": 0,  # Allow 0 faces - "no face" is a valid result
        "face_size": 64,
    }

    # Perform sequential detection with graceful shutdown handling
    try:
        # Use Rich progress bar for better user experience
        Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = (
            _get_progress()
        )

        # Process all images at once for parallel processing
        if workers and workers > 1:
            # Use parallel processing - let the core handle progress bars
            console.print("🔄 Processing images in parallel...", style="blue")
            results_dict = core.detect_faces(
                files_to_process, max_workers=workers, **detection_kwargs
            )
        else:
            # Use sequential processing with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[progress.completed]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task("Detecting faces", total=len(files_to_process))

                # Process images in chunks to update progress
                chunk_size = max(
                    1, len(files_to_process) // 100
                )  # Update progress ~100 times
                results_dict = {}

                for i in range(0, len(files_to_process), chunk_size):
                    chunk = files_to_process[i : i + chunk_size]
                    chunk_results = core.detect_faces(
                        chunk, max_workers=workers, **detection_kwargs
                    )
                    results_dict.update(chunk_results)
                    progress.update(task, advance=len(chunk))

    except KeyboardInterrupt:
        _get_console().print("\n🛑 Face detection interrupted by user", style="yellow")
        return
    except Exception as e:
        _get_console().print(f"❌ Face detection failed: {e}", style="red")
        return

    # Convert results to list format for compatibility
    console.print("📊 Processing results...", style="blue")
    results = []
    total_faces_found = 0
    total_time = 0.0

    # Add progress indicator for result processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Processing results", total=len(files_to_process))

        for image_file in files_to_process:
            if str(image_file) in results_dict:
                result = results_dict[str(image_file)]
                results.append(result)
                total_faces_found += result.face_count
                total_time += result.processing_time
            else:
                # Handle missing results
                from ...detectors.face import FaceDetectionResult

                error_result = FaceDetectionResult(
                    faces=[],
                    face_count=0,
                    success=False,
                    processing_time=0.0,
                    error="No result returned",
                )
                results.append(error_result)

            progress.update(task, advance=1)

    # Display final results
    display_face_detection_results(
        results,
        len(files_to_process),
        total_faces_found,
        total_time,
        len(skipped_files),
        not show_empty_results,
    )


@face_group.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for extracted faces",
)
@click.option(
    "--padding",
    "-p",
    type=int,
    default=20,
    help="Padding around faces in pixels (default: 20)",
)
@click.option(
    "--workers", "-w", type=int, help="Number of parallel workers (default: auto)"
)
@click.option(
    "--no-recursive",
    "no_recursive",
    is_flag=True,
    help="Disable recursive directory processing",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Enable verbose logging (-v for info, -vv for debug)",
)
@click.option(
    "--show-empty-results",
    "show_empty_results",
    is_flag=True,
    help="Show output even when no faces are extracted (default: suppress empty results)",
)
@click.pass_context
def extract(
    ctx: click.Context,
    input_path: Path,
    output: Optional[Path],
    padding: int,
    workers: Optional[int],
    no_recursive: bool,
    verbose: int,
    show_empty_results: bool,
):
    """
    Extract detected faces to separate images at their natural detected size.

    INPUT_PATH should be a directory containing images with face detection sidecar files.
    """

    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        _get_console().print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        _get_console().print("ℹ️  Info logging enabled", style="blue")

    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core

    core = get_core(ctx)

    # Determine output directory
    if output is None:
        output = input_path / f"{input_path.name}_faces"

    _get_console().print(f"✂️  Extracting faces from {input_path}...", style="blue")
    _get_console().print(f"📁 Output directory: {output}", style="blue")
    _get_console().print(f"📏 Padding: {padding}px", style="blue")

    # Find image files (recursive by default)
    recursive = not no_recursive
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import find_image_files

    image_files = find_image_files(input_path, recursive=recursive)

    if not image_files:
        _get_console().print("❌ No image files found", style="red")
        return

    _get_console().print(f"📊 Found {len(image_files)} images to process", style="blue")

    # Extract faces using core
    extraction_results = core.extract_faces(
        image_files, output, padding=padding, max_workers=workers
    )

    # Display extraction results
    display_face_extraction_results(extraction_results, output, not show_empty_results)


@face_group.command()
@click.argument("input_pattern", type=str)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="face_benchmark_results.json",
    help="Output file for benchmark results (default: face_benchmark_results.json)",
)
@click.option(
    "--max-images",
    "-m",
    type=int,
    default=50,
    help="Maximum number of images to benchmark (default: 50)",
)
@click.option(
    "--detectors",
    "-d",
    type=str,
    help="Comma-separated list of detectors to test (flexible_detector,insightface)",
)
@click.option(
    "--confidence",
    "-c",
    type=float,
    default=0.5,
    help="Confidence threshold for face detection (default: 0.5)",
)
@click.option(
    "--min-face-size",
    type=int,
    default=64,
    help="Minimum face size in pixels (default: 64)",
)
@click.option("--gpu/--no-gpu", default=True, help="Use GPU acceleration if available")
@click.option(
    "--no-recursive",
    "no_recursive",
    is_flag=True,
    help="Disable recursive directory processing",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Enable verbose logging (-v for info, -vv for debug)",
)
@click.pass_context
def benchmark(
    ctx: click.Context,
    input_pattern: str,
    output: Path,
    max_images: int,
    detectors: Optional[str],
    confidence: float,
    min_face_size: int,
    gpu: bool,
    no_recursive: bool,
    verbose: int,
):
    """
    Benchmark different face detection methods for speed and accuracy.

    INPUT_PATTERN can be a file pattern, directory path, or single image file.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """

    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        _get_console().print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        _get_console().print("ℹ️  Info logging enabled", style="blue")

    # Parse detector list
    detector_list = None
    if detectors:
        detector_list = [d.strip() for d in detectors.split(",")]
        _get_console().print(
            f"🎯 Testing detectors: {', '.join(detector_list)}", style="blue"
        )

    # Find image files
    input_path = Path(input_pattern)
    recursive = not no_recursive

    if input_path.is_dir():
        # Lazy import to avoid heavy dependencies at startup
        from ..utils import find_image_files

        image_files = find_image_files(input_path, recursive=recursive)
    else:
        # Pattern matching
        if input_pattern.startswith("/"):
            parent_dir = Path(input_pattern).parent
            pattern = Path(input_pattern).name
            image_files = list(parent_dir.glob(pattern))
        else:
            image_files = list(Path(".").glob(input_pattern))

    if not image_files:
        _get_console().print("❌ No images found", style="red")
        return

    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]

    _get_console().print(
        f"📊 Found {len(image_files)} images for benchmarking", style="blue"
    )

    # Initialize benchmark (lazy import)
    from ...detectors.face_benchmark import FaceDetectionBenchmark

    benchmark = FaceDetectionBenchmark(
        enable_gpu=gpu, confidence_threshold=confidence, min_face_size=min_face_size
    )

    # Get model and GPU information before starting benchmark
    from ...detectors.face import FaceDetector

    test_detector = FaceDetector(enable_gpu=gpu)
    model_info = test_detector.get_model_info()

    # Display enhanced startup message
    console = _get_console()
    console.print("🚀 Starting face detection benchmark...", style="blue")

    # Display model information
    model_name = model_info.get("model", "unknown")
    device = model_info.get("device", "cpu")
    gpu_enabled = model_info.get("gpu_enabled", False)
    gpu_test_passed = model_info.get("gpu_test_passed", False)

    if gpu_enabled and gpu_test_passed:
        gpu_memory = model_info.get("gpu_memory_gb", 0)
        console.print(
            f"📱 Model: {model_name} | Device: {device} | GPU: ✅ Available ({gpu_memory}GB)",
            style="green",
        )
    elif gpu_enabled and not gpu_test_passed:
        gpu_error = model_info.get("gpu_test_error", "Unknown error")
        console.print(
            f"📱 Model: {model_name} | Device: {device} | GPU: ❌ Failed ({gpu_error})",
            style="yellow",
        )
    else:
        console.print(
            f"📱 Model: {model_name} | Device: {device} | GPU: Disabled", style="blue"
        )

    # Run benchmark
    try:
        summary = benchmark.benchmark_batch(
            image_files, detectors=detector_list, max_images=max_images
        )

        # Display results
        display_benchmark_results(summary)

    except Exception as e:
        _get_console().print(f"❌ Benchmark failed: {e}", style="red")
        return


@face_group.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for clustering results",
)
@click.option(
    "--similarity",
    "-s",
    type=float,
    default=0.6,
    help="Similarity threshold for clustering (0.0-1.0, default: 0.6)",
)
@click.option(
    "--min-cluster-size",
    "min_cluster_size",
    type=int,
    default=2,
    help="Minimum number of faces per cluster (default: 2)",
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(["dbscan", "agglomerative", "kmeans"]),
    default="dbscan",
    help="Clustering algorithm (default: dbscan)",
)
@click.option(
    "--max-faces",
    "max_faces",
    type=int,
    help="Maximum number of faces to cluster (default: all)",
)
@click.option(
    "--export-format",
    "export_format",
    type=click.Choice(["json", "csv", "both"]),
    default="both",
    help="Export format (default: both)",
)
@click.option(
    "--no-visualization",
    "no_visualization",
    is_flag=True,
    help="Skip creating cluster visualization images",
)
@click.option(
    "--no-recursive",
    "no_recursive",
    is_flag=True,
    help="Disable recursive directory processing",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Enable verbose logging (-v for info, -vv for debug)",
)
@click.pass_context
def cluster(
    ctx: click.Context,
    input_path: Path,
    output: Optional[Path],
    similarity: float,
    min_cluster_size: int,
    algorithm: str,
    max_faces: Optional[int],
    export_format: str,
    no_visualization: bool,
    no_recursive: bool,
    verbose: int,
):
    """
    Cluster similar faces together.

    INPUT_PATH should be a directory containing face detection sidecar files.
    """

    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        _get_console().print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        _get_console().print("ℹ️  Info logging enabled", style="blue")

    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core

    core = get_core(ctx)

    # Determine output directory
    if output is None:
        output = input_path / f"{input_path.name}_clusters"

    _get_console().print(f"🔗 Clustering faces in {input_path}...", style="blue")
    # Validate parameters
    if not 0.0 <= similarity <= 1.0:
        _get_console().print(
            f"❌ Invalid similarity threshold: {similarity}. Must be between 0.0 and 1.0",
            style="red",
        )
        return

    if min_cluster_size < 1:
        _get_console().print(
            f"❌ Invalid min cluster size: {min_cluster_size}. Must be >= 1",
            style="red",
        )
        return

    _get_console().print(f"📁 Output directory: {output}", style="blue")
    _get_console().print(f"🎯 Similarity threshold: {similarity}", style="blue")
    _get_console().print(f"👥 Min cluster size: {min_cluster_size}", style="blue")
    _get_console().print(f"🧮 Algorithm: {algorithm}", style="blue")

    # Check if input path contains sidecar files with face detection data
    # Handle both direct sidecar files and symlinked images
    sidecar_files = []

    # First, look for direct sidecar files in the directory
    direct_sidecar_files = list(input_path.glob("*.json"))
    sidecar_files.extend(direct_sidecar_files)

    # Also check for sidecar files next to symlinked images
    image_files = (
        list(input_path.glob("*.jpg"))
        + list(input_path.glob("*.jpeg"))
        + list(input_path.glob("*.png"))
    )
    for image_file in image_files:
        if image_file.is_symlink():
            try:
                # Resolve symlink and look for sidecar next to target
                target_path = image_file.resolve()
                target_sidecar = target_path.with_suffix(".json")
                if target_sidecar.exists() and target_sidecar not in sidecar_files:
                    sidecar_files.append(target_sidecar)
            except Exception:
                continue

    if not sidecar_files:
        _get_console().print("❌ No sidecar files found", style="red")
        _get_console().print(
            "💡 Run 'sportball face detect' first to detect faces", style="yellow"
        )
        return

    # Check if any sidecar files contain face detection data
    face_detection_files = []
    for sidecar_file in sidecar_files:
        try:
            import json

            with open(sidecar_file, "r") as f:
                data = json.load(f)
                if "face_detection" in data and data["face_detection"].get(
                    "success", False
                ):
                    face_detection_files.append(sidecar_file)
        except Exception:
            continue

    if not face_detection_files:
        _get_console().print(
            "❌ No face detection data found in sidecar files", style="red"
        )
        _get_console().print(
            "💡 Run 'sportball face detect' first to detect faces", style="yellow"
        )
        return

    _get_console().print(
        f"📊 Found {len(face_detection_files)} sidecar files with face detection data",
        style="blue",
    )

    # Load detection results from sidecar files
    detection_results = {}
    for sidecar_file in face_detection_files:
        try:
            import json

            with open(sidecar_file, "r") as f:
                data = json.load(f)
                if "face_detection" in data and data["face_detection"].get(
                    "success", False
                ):
                    # Extract image path from sidecar filename
                    image_path = sidecar_file.stem
                    detection_results[image_path] = data["face_detection"]
        except Exception as e:
            _get_console().print(
                f"⚠️  Failed to load sidecar file {sidecar_file}: {e}", style="yellow"
            )

    if not detection_results:
        _get_console().print("❌ No valid face detection data found", style="red")
        return

    _get_console().print(
        f"📊 Loaded face detection data for {len(detection_results)} images",
        style="blue",
    )

    # Perform clustering
    try:
        clustering_result = core.cluster_faces(
            detection_results,
            similarity_threshold=similarity,
            min_cluster_size=min_cluster_size,
            algorithm=algorithm,
            max_faces=max_faces,
            save_sidecar=True,
        )

        if not clustering_result.get("success", False):
            _get_console().print(
                f"❌ Clustering failed: {clustering_result.get('error', 'Unknown error')}",
                style="red",
            )
            return

        # Display clustering results
        display_clustering_results(clustering_result)

        # Export results
        _get_console().print(f"📤 Exporting results to {output}...", style="blue")

        export_results = core.export_face_clusters(
            clustering_result,
            output,
            export_format=export_format,
            create_visualization=not no_visualization,
        )

        if export_results.get("success", False):
            _get_console().print("✅ Export completed successfully", style="green")

            # Display export summary
            files_created = export_results.get("files_created", [])
            if files_created:
                _get_console().print(
                    f"📄 Files created: {len(files_created)}", style="blue"
                )
                for file_path in files_created:
                    _get_console().print(f"  • {file_path}", style="dim")

            # Display visualization results
            viz_results = export_results.get("visualization", {})
            if viz_results.get("success", False):
                images_created = viz_results.get("images_created", [])
                if images_created:
                    _get_console().print(
                        f"🖼️  Visualization images: {len(images_created)}", style="blue"
                    )
                    for img_path in images_created:
                        _get_console().print(f"  • {img_path}", style="dim")
        else:
            _get_console().print(
                f"❌ Export failed: {export_results.get('error', 'Unknown error')}",
                style="red",
            )

    except Exception as e:
        _get_console().print(f"❌ Clustering failed: {e}", style="red")
        if verbose >= 1:
            import traceback

            _get_console().print(traceback.format_exc(), style="red")


def display_clustering_results(clustering_result: dict):
    """Display face clustering results in a formatted table."""
    # Lazy import: from rich.table import Table

    # Create results table
    table = Table(title="Face Clustering Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Add clustering statistics
    table.add_row("Total Faces", str(clustering_result.get("total_faces", 0)))
    table.add_row("Clusters Found", str(clustering_result.get("cluster_count", 0)))
    table.add_row(
        "Unclustered Faces", str(len(clustering_result.get("unclustered_faces", [])))
    )
    table.add_row("Algorithm", clustering_result.get("algorithm_used", "unknown"))
    table.add_row(
        "Processing Time", f"{clustering_result.get('processing_time', 0.0):.2f}s"
    )

    _get_console().print(table)

    # Display cluster details
    clusters = clustering_result.get("clusters", [])
    if clusters:
        cluster_table = Table(title="Cluster Details")
        cluster_table.add_column("Cluster ID", style="cyan")
        cluster_table.add_column("Face Count", style="green")
        cluster_table.add_column("Avg Confidence", style="yellow")
        cluster_table.add_column("Images", style="blue")

        for cluster in clusters:
            cluster_id = cluster.get("cluster_id", -1)
            face_count = cluster.get("face_count", 0)
            confidences = cluster.get("confidence_scores", [])
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            image_paths = cluster.get("image_paths", [])
            unique_images = len(set(image_paths))

            cluster_table.add_row(
                str(cluster_id),
                str(face_count),
                f"{avg_confidence:.3f}",
                str(unique_images),
            )

        _get_console().print(cluster_table)

    # Display unclustered faces if any
    unclustered = clustering_result.get("unclustered_faces", [])
    if unclustered:
        _get_console().print(
            f"\n⚠️  {len(unclustered)} faces could not be clustered", style="yellow"
        )
        if len(unclustered) <= 10:  # Show details for small numbers
            for face_id in unclustered:
                _get_console().print(f"  • {face_id}", style="dim")
        else:
            _get_console().print(
                f"  (showing first 10 of {len(unclustered)})", style="dim"
            )
            for face_id in unclustered[:10]:
                _get_console().print(f"  • {face_id}", style="dim")


def display_face_results(
    results: dict, extract_faces: bool, output_dir: Optional[Path]
):
    """Display face detection results."""

    # Create results table
    table = Table(title="Face Detection Results")
    table.add_column("Image", style="cyan")
    table.add_column("Faces Found", style="green", justify="right")
    table.add_column("Success", style="green")
    table.add_column("Error", style="red")

    total_faces = 0
    successful_images = 0

    for image_path, result in results.items():
        if result.get("success", False):
            face_count = len(result.get("faces", []))
            total_faces += face_count
            successful_images += 1

            table.add_row(Path(image_path).name, str(face_count), "✅", "")
        else:
            error_msg = result.get("error", "Unknown error")
            table.add_row(
                Path(image_path).name,
                "0",
                "❌",
                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg,
            )

    _get_console().print(table)
    _get_console().print(
        f"\n📊 Summary: {successful_images}/{len(results)} images processed, {total_faces} faces detected"
    )

    # Extract faces if requested
    if extract_faces and output_dir:
        _get_console().print(f"\n💾 Extracting faces to {output_dir}...", style="blue")
        # TODO: Implement face extraction
        _get_console().print("Face extraction not yet implemented", style="yellow")


def display_face_detection_results(
    results,
    total_images: int,
    total_faces_found: int,
    total_time: float,
    skipped_count: int = 0,
    suppress_empty: bool = True,
):
    """
    Display face detection results summary.

    Args:
        results: List of detection results
        total_images: Total number of images processed
        total_faces_found: Total number of faces found
        total_time: Total processing time
        skipped_count: Number of images skipped
        suppress_empty: If True, suppress output when no faces are found (default: True)
    """

    # If suppress mode is enabled and no faces were found, suppress output
    if suppress_empty and total_faces_found == 0:
        return

    _get_console().print("\n✅ Face detection complete!", style="green")
    _get_console().print(f"📊 Processed {total_images} images")
    _get_console().print(f"👥 Found {total_faces_found} faces")
    _get_console().print(f"⏱️  Total detection time: {total_time:.2f}s")
    if total_images > 0:
        _get_console().print(
            f"📈 Average time per image: {total_time / total_images:.2f}s"
        )

    # Show skipped count
    if skipped_count > 0:
        _get_console().print(
            f"⏭️  {skipped_count} images skipped (existing sidecar data)", style="blue"
        )

    # Show processing info
    if total_images > 1:
        _get_console().print(
            "🔄 Used sequential processing for optimal performance", style="blue"
        )


def display_benchmark_results(summary):
    """Display benchmark results summary."""
    # Lazy import: from rich.table import Table

    _get_console().print("\n🏆 Face Detection Benchmark Results", style="bold green")
    _get_console().print(
        f"📊 Tested {summary.total_images} images with {len(summary.detectors)} detectors"
    )

    # Performance table
    perf_table = Table(title="Performance Statistics")
    perf_table.add_column("Detector", style="cyan")
    perf_table.add_column("Avg Time (s)", style="green", justify="right")
    perf_table.add_column("Min Time (s)", style="green", justify="right")
    perf_table.add_column("Max Time (s)", style="green", justify="right")
    perf_table.add_column("Success Rate", style="green", justify="right")
    perf_table.add_column("Total Faces", style="green", justify="right")

    for detector_name in summary.detectors:
        if detector_name in summary.performance_stats:
            stats = summary.performance_stats[detector_name]
            perf_table.add_row(
                detector_name,
                f"{stats['avg_time']:.3f}",
                f"{stats['min_time']:.3f}",
                f"{stats['max_time']:.3f}",
                f"{stats['success_rate']:.1%}",
                str(stats["total_faces"]),
            )

    _get_console().print(perf_table)

    # Accuracy table
    acc_table = Table(title="Accuracy Statistics")
    acc_table.add_column("Detector", style="cyan")
    acc_table.add_column("Avg Confidence", style="green", justify="right")
    acc_table.add_column("Min Confidence", style="green", justify="right")
    acc_table.add_column("Max Confidence", style="green", justify="right")
    acc_table.add_column("Avg Face Size", style="green", justify="right")

    for detector_name in summary.detectors:
        if detector_name in summary.accuracy_stats:
            stats = summary.accuracy_stats[detector_name]
            acc_table.add_row(
                detector_name,
                f"{stats['avg_confidence']:.3f}",
                f"{stats['min_confidence']:.3f}",
                f"{stats['max_confidence']:.3f}",
                f"{stats['avg_face_size']:.0f}",
            )

    _get_console().print(acc_table)


def display_detector_comparison(comparison):
    """Display detector comparison and recommendations."""
    _get_console().print("\n🎯 Detector Comparison", style="bold blue")

    if comparison["fastest_detector"]:
        _get_console().print(
            f"⚡ Fastest: {comparison['fastest_detector']}", style="green"
        )

    if comparison["most_accurate_detector"]:
        _get_console().print(
            f"🎯 Most Accurate: {comparison['most_accurate_detector']}", style="green"
        )

    if comparison["most_reliable_detector"]:
        _get_console().print(
            f"🛡️  Most Reliable: {comparison['most_reliable_detector']}", style="green"
        )

    if comparison["recommendations"]:
        _get_console().print("\n💡 Recommendations:", style="bold yellow")
        for rec in comparison["recommendations"]:
            _get_console().print(f"  • {rec}", style="yellow")

    # Overall scores
    if comparison["detailed_comparison"]:
        _get_console().print("\n📊 Overall Scores:", style="bold blue")
        for detector_name, details in comparison["detailed_comparison"].items():
            score = details.get("overall_score", 0)
            _get_console().print(f"  {detector_name}: {score:.1f}/100", style="blue")


def display_face_extraction_results(
    extraction_results: dict, output_dir: Path, suppress_empty: bool = True
):
    """
    Display face extraction results summary.

    Args:
        extraction_results: Dictionary of extraction results
        output_dir: Output directory for extracted faces
        suppress_empty: If True, suppress output when no faces are extracted (default: True)
    """

    total_images = len(extraction_results)
    total_faces_extracted = 0
    successful_extractions = 0
    failed_extractions = 0

    # Count results
    for image_path, result in extraction_results.items():
        if result.get("success", False):
            successful_extractions += 1
            total_faces_extracted += result.get("faces_extracted", 0)
        else:
            failed_extractions += 1

    # If suppress mode is enabled and no faces were extracted, suppress output
    if suppress_empty and total_faces_extracted == 0:
        return

    _get_console().print("\n✅ Face extraction complete!", style="green")
    _get_console().print(f"📊 Processed {total_images} images")
    _get_console().print(f"👥 Extracted {total_faces_extracted} faces")
    _get_console().print(f"📁 Saved to: {output_dir}", style="blue")

    if successful_extractions > 0:
        _get_console().print(
            f"✅ {successful_extractions} images processed successfully", style="green"
        )

    if failed_extractions > 0:
        _get_console().print(
            f"❌ {failed_extractions} images failed to process", style="red"
        )

        # Show details for failed extractions
        _get_console().print("\n⚠️  Failed extractions:", style="yellow")
        for image_path, result in extraction_results.items():
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                _get_console().print(
                    f"  • {Path(image_path).name}: {error_msg}", style="red"
                )

    if total_faces_extracted > 0:
        avg_faces = (
            total_faces_extracted / successful_extractions
            if successful_extractions > 0
            else 0
        )
        _get_console().print(
            f"📈 Average faces per image: {avg_faces:.1f}", style="blue"
        )
