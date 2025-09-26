"""
Face Detection Commands

CLI commands for face detection and recognition operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
import json
import signal
import sys
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Lazy imports to avoid heavy dependencies at startup
# from ..utils import get_core, find_image_files, check_sidecar_files
# from ...sidecar import Sidecar, OperationType
# from ...detectors.face_benchmark import FaceDetectionBenchmark

console = Console()

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    console.print("\n🛑 Shutdown requested. Finishing current operations...", style="yellow")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@click.group()
def face_group():
    """Face detection and recognition commands."""
    pass


@face_group.command()
@click.argument('input_pattern', type=str)
@click.option('--border-padding', '-b', 
              default=0.25, 
              help='Border padding percentage (0.25 = 25%)')
@click.option('--max-images', '-m', 
              default=None, 
              type=int, 
              help='Maximum number of images to process')
@click.option('--gpu/--no-gpu', 
              default=True, 
              help='Use GPU acceleration if available')
@click.option('--force', '-f', 
              is_flag=True, 
              help='Force detection even if JSON sidecar exists')
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
@click.option('--verbose', '-v', 
              count=True, 
              help='Enable verbose logging (-v for info, -vv for debug)')
@click.option('--batch-size', 'batch_size',
              type=int,
              default=8,
              help='Processing batch size (legacy parameter, not used in sequential mode)')
@click.option('--auto-tune', 'auto_tune',
              is_flag=True,
              help='Automatically tune processing parameters for optimal performance')
@click.pass_context
def detect(ctx: click.Context, 
           input_pattern: str,
           border_padding: float,
           max_images: Optional[int],
           gpu: bool,
           force: bool,
           no_recursive: bool,
           verbose: int,
           batch_size: int,
           auto_tune: bool):
    """
    Detect faces in images and save comprehensive data to JSON sidecar files.
    
    INPUT_PATTERN can be a file pattern, directory path, or single image file.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        console.print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        console.print("ℹ️  Info logging enabled", style="blue")
    
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
        if input_pattern.startswith('/'):
            parent_dir = Path(input_pattern).parent
            pattern = Path(input_pattern).name
            image_files = list(parent_dir.glob(pattern))
        else:
            image_files = list(Path('.').glob(input_pattern))
    
    if not image_files:
        console.print("❌ No images found", style="red")
        return
    
    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]
    
    console.print(f"📊 Found {len(image_files)} images to analyze", style="blue")
    
    # Auto-tune processing parameters if requested
    if auto_tune and gpu:
        console.print("🔧 Auto-tuning processing parameters...", style="blue")
        try:
            # For sequential processing, we don't need batch size tuning
            # This is kept for compatibility but doesn't affect performance
            console.print("ℹ️  Sequential processing mode - batch size tuning not needed", style="blue")
        except Exception as e:
            console.print(f"⚠️  Auto-tuning failed: {e}", style="yellow")
    
    # Check for existing sidecar files
    console.print("🔍 Checking for existing sidecar files...", style="blue")
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import check_sidecar_files_parallel
    files_to_process, skipped_files = check_sidecar_files_parallel(
        image_files, 
        force, 
        operation_type="face_detection"
    )
    
    # Show skipping message after image discovery but before processing
    if skipped_files:
        console.print(f"⏭️  Skipping {len(skipped_files)} images - JSON sidecar already exists (use --force to override)", style="yellow")
    
    console.print(f"📊 Processing {len(files_to_process)} images ({len(skipped_files)} skipped)", style="blue")
    
    if not files_to_process:
        console.print("✅ All images already processed (use --force to reprocess)", style="green")
        return
    
    console.print(f"🔍 Starting face detection...", style="blue")
    
    # Use core's sequential processing for face detection
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core
    core = get_core(ctx)
    
    # Prepare detection parameters
    detection_kwargs = {
        'confidence': 0.5,
        'min_faces': 0,  # Allow 0 faces - "no face" is a valid result
        'face_size': 64
    }
    
    # Perform sequential detection with graceful shutdown handling
    try:
        results_dict = core.detect_faces(files_to_process, **detection_kwargs)
    except KeyboardInterrupt:
        console.print("\n🛑 Face detection interrupted by user", style="yellow")
        return
    except Exception as e:
        console.print(f"❌ Face detection failed: {e}", style="red")
        return
    
    # Convert results to list format for compatibility
    results = []
    total_faces_found = 0
    total_time = 0.0
    
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
                error="No result returned"
            )
            results.append(error_result)
    
    # Display final results
    display_face_detection_results(results, len(files_to_process), total_faces_found, total_time, len(skipped_files))


@face_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for extracted faces')
@click.option('--face-size', 'face_size',
              type=int,
              default=256,
              help='Size of extracted faces in pixels (default: 256)')
@click.option('--padding', '-p',
              type=int,
              default=20,
              help='Padding around faces in pixels (default: 20)')
@click.option('--workers', '-w',
              type=int,
              help='Number of parallel workers (default: auto)')
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
@click.option('--verbose', '-v',
              count=True,
              help='Enable verbose logging (-v for info, -vv for debug)')
@click.pass_context
def extract(ctx: click.Context, 
           input_path: Path, 
           output: Optional[Path],
           face_size: int,
           padding: int,
           workers: Optional[int],
           no_recursive: bool,
           verbose: int):
    """
    Extract detected faces to separate images.
    
    INPUT_PATH should be a directory containing images with face detection sidecar files.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        console.print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        console.print("ℹ️  Info logging enabled", style="blue")
    
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core
    core = get_core(ctx)
    
    # Determine output directory
    if output is None:
        output = input_path / f"{input_path.name}_faces"
    
    console.print(f"✂️  Extracting faces from {input_path}...", style="blue")
    console.print(f"📁 Output directory: {output}", style="blue")
    console.print(f"🖼️  Face size: {face_size}px", style="blue")
    console.print(f"📏 Padding: {padding}px", style="blue")
    
    # Find image files (recursive by default)
    recursive = not no_recursive
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import find_image_files
    image_files = find_image_files(input_path, recursive=recursive)
    
    if not image_files:
        console.print("❌ No image files found", style="red")
        return
    
    console.print(f"📊 Found {len(image_files)} images to process", style="blue")
    
    # Extract faces using core
    extraction_results = core.extract_faces(
        image_files,
        output,
        face_size=face_size,
        padding=padding,
        max_workers=workers
    )
    
    # Display extraction results
    display_face_extraction_results(extraction_results, output)


@face_group.command()
@click.argument('input_pattern', type=str)
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default='face_benchmark_results.json',
              help='Output file for benchmark results (default: face_benchmark_results.json)')
@click.option('--max-images', '-m',
              type=int,
              default=50,
              help='Maximum number of images to benchmark (default: 50)')
@click.option('--detectors', '-d',
              type=str,
              help='Comma-separated list of detectors to test (opencv_face_recognition,insightface)')
@click.option('--confidence', '-c',
              type=float,
              default=0.5,
              help='Confidence threshold for face detection (default: 0.5)')
@click.option('--min-face-size',
              type=int,
              default=64,
              help='Minimum face size in pixels (default: 64)')
@click.option('--gpu/--no-gpu',
              default=True,
              help='Use GPU acceleration if available')
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
@click.option('--verbose', '-v',
              count=True,
              help='Enable verbose logging (-v for info, -vv for debug)')
@click.pass_context
def benchmark(ctx: click.Context,
              input_pattern: str,
              output: Path,
              max_images: int,
              detectors: Optional[str],
              confidence: float,
              min_face_size: int,
              gpu: bool,
              no_recursive: bool,
              verbose: int):
    """
    Benchmark different face detection methods for speed and accuracy.
    
    INPUT_PATTERN can be a file pattern, directory path, or single image file.
    By default, directories are processed recursively. Use --no-recursive to disable.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        console.print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        console.print("ℹ️  Info logging enabled", style="blue")
    
    # Parse detector list
    detector_list = None
    if detectors:
        detector_list = [d.strip() for d in detectors.split(',')]
        console.print(f"🎯 Testing detectors: {', '.join(detector_list)}", style="blue")
    
    # Find image files
    input_path = Path(input_pattern)
    recursive = not no_recursive
    
    if input_path.is_dir():
        # Lazy import to avoid heavy dependencies at startup
        from ..utils import find_image_files
        image_files = find_image_files(input_path, recursive=recursive)
    else:
        # Pattern matching
        if input_pattern.startswith('/'):
            parent_dir = Path(input_pattern).parent
            pattern = Path(input_pattern).name
            image_files = list(parent_dir.glob(pattern))
        else:
            image_files = list(Path('.').glob(input_pattern))
    
    if not image_files:
        console.print("❌ No images found", style="red")
        return
    
    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]
    
    console.print(f"📊 Found {len(image_files)} images for benchmarking", style="blue")
    
    # Initialize benchmark (lazy import)
    from ...detectors.face_benchmark import FaceDetectionBenchmark
    benchmark = FaceDetectionBenchmark(
        enable_gpu=gpu,
        confidence_threshold=confidence,
        min_face_size=min_face_size
    )
    
    console.print("🚀 Starting face detection benchmark...", style="blue")
    
    # Run benchmark
    try:
        summary = benchmark.benchmark_batch(
            image_files,
            detectors=detector_list,
            max_images=max_images
        )
        
        # Display results
        display_benchmark_results(summary)
        
    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}", style="red")
        return


@face_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for clustering results')
@click.option('--similarity', '-s',
              type=float,
              default=0.6,
              help='Similarity threshold for clustering (0.0-1.0, default: 0.6)')
@click.option('--min-cluster-size', 'min_cluster_size',
              type=int,
              default=2,
              help='Minimum number of faces per cluster (default: 2)')
@click.option('--algorithm', '-a',
              type=click.Choice(['dbscan', 'agglomerative', 'kmeans']),
              default='dbscan',
              help='Clustering algorithm (default: dbscan)')
@click.option('--max-faces', 'max_faces',
              type=int,
              help='Maximum number of faces to cluster (default: all)')
@click.option('--export-format', 'export_format',
              type=click.Choice(['json', 'csv', 'both']),
              default='both',
              help='Export format (default: both)')
@click.option('--no-visualization', 'no_visualization',
              is_flag=True,
              help='Skip creating cluster visualization images')
@click.option('--no-recursive', 'no_recursive',
              is_flag=True,
              help='Disable recursive directory processing')
@click.option('--verbose', '-v',
              count=True,
              help='Enable verbose logging (-v for info, -vv for debug)')
@click.pass_context
def cluster(ctx: click.Context,
           input_path: Path,
           output: Optional[Path],
           similarity: float,
           min_cluster_size: int,
           algorithm: str,
           max_faces: Optional[int],
           export_format: str,
           no_visualization: bool,
           no_recursive: bool,
           verbose: int):
    """
    Cluster similar faces together.
    
    INPUT_PATH should be a directory containing face detection sidecar files.
    """
    
    # Setup logging based on verbose level
    if verbose >= 2:  # -vv: debug level
        console.print("🔍 Debug logging enabled", style="blue")
    elif verbose >= 1:  # -v: info level
        console.print("ℹ️  Info logging enabled", style="blue")
    
    # Lazy import to avoid heavy dependencies at startup
    from ..utils import get_core
    core = get_core(ctx)
    
    # Determine output directory
    if output is None:
        output = input_path / f"{input_path.name}_clusters"
    
    console.print(f"🔗 Clustering faces in {input_path}...", style="blue")
    console.print(f"📁 Output directory: {output}", style="blue")
    console.print(f"🎯 Similarity threshold: {similarity}", style="blue")
    console.print(f"👥 Min cluster size: {min_cluster_size}", style="blue")
    console.print(f"🧮 Algorithm: {algorithm}", style="blue")
    
    # Check if input path contains face detection sidecar files
    sidecar_files = list(input_path.glob("*_face_detection.json"))
    if not sidecar_files:
        console.print("❌ No face detection sidecar files found", style="red")
        console.print("💡 Run 'sportball face detect' first to detect faces", style="yellow")
        return
    
    console.print(f"📊 Found {len(sidecar_files)} face detection files", style="blue")
    
    # Perform clustering
    try:
        clustering_result = core.cluster_faces(
            input_path,
            similarity_threshold=similarity,
            min_cluster_size=min_cluster_size,
            algorithm=algorithm,
            max_faces=max_faces,
            save_sidecar=True
        )
        
        if not clustering_result.get('success', False):
            console.print(f"❌ Clustering failed: {clustering_result.get('error', 'Unknown error')}", style="red")
            return
        
        # Display clustering results
        display_clustering_results(clustering_result)
        
        # Export results
        console.print(f"📤 Exporting results to {output}...", style="blue")
        
        export_results = core.export_face_clusters(
            clustering_result,
            output,
            export_format=export_format,
            create_visualization=not no_visualization
        )
        
        if export_results.get('success', False):
            console.print("✅ Export completed successfully", style="green")
            
            # Display export summary
            files_created = export_results.get('files_created', [])
            if files_created:
                console.print(f"📄 Files created: {len(files_created)}", style="blue")
                for file_path in files_created:
                    console.print(f"  • {file_path}", style="dim")
            
            # Display visualization results
            viz_results = export_results.get('visualization', {})
            if viz_results.get('success', False):
                images_created = viz_results.get('images_created', [])
                if images_created:
                    console.print(f"🖼️  Visualization images: {len(images_created)}", style="blue")
                    for img_path in images_created:
                        console.print(f"  • {img_path}", style="dim")
        else:
            console.print(f"❌ Export failed: {export_results.get('error', 'Unknown error')}", style="red")
        
    except Exception as e:
        console.print(f"❌ Clustering failed: {e}", style="red")
        if verbose >= 1:
            import traceback
            console.print(traceback.format_exc(), style="red")


def display_clustering_results(clustering_result: dict):
    """Display face clustering results in a formatted table."""
    from rich.table import Table
    from rich.panel import Panel
    
    # Create results table
    table = Table(title="Face Clustering Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add clustering statistics
    table.add_row("Total Faces", str(clustering_result.get('total_faces', 0)))
    table.add_row("Clusters Found", str(clustering_result.get('cluster_count', 0)))
    table.add_row("Unclustered Faces", str(len(clustering_result.get('unclustered_faces', []))))
    table.add_row("Algorithm", clustering_result.get('algorithm_used', 'unknown'))
    table.add_row("Processing Time", f"{clustering_result.get('processing_time', 0.0):.2f}s")
    
    console.print(table)
    
    # Display cluster details
    clusters = clustering_result.get('clusters', [])
    if clusters:
        cluster_table = Table(title="Cluster Details")
        cluster_table.add_column("Cluster ID", style="cyan")
        cluster_table.add_column("Face Count", style="green")
        cluster_table.add_column("Avg Confidence", style="yellow")
        cluster_table.add_column("Images", style="blue")
        
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id', -1)
            face_count = cluster.get('face_count', 0)
            confidences = cluster.get('confidence_scores', [])
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            image_paths = cluster.get('image_paths', [])
            unique_images = len(set(image_paths))
            
            cluster_table.add_row(
                str(cluster_id),
                str(face_count),
                f"{avg_confidence:.3f}",
                str(unique_images)
            )
        
        console.print(cluster_table)
    
    # Display unclustered faces if any
    unclustered = clustering_result.get('unclustered_faces', [])
    if unclustered:
        console.print(f"\n⚠️  {len(unclustered)} faces could not be clustered", style="yellow")
        if len(unclustered) <= 10:  # Show details for small numbers
            for face_id in unclustered:
                console.print(f"  • {face_id}", style="dim")
        else:
            console.print(f"  (showing first 10 of {len(unclustered)})", style="dim")
            for face_id in unclustered[:10]:
                console.print(f"  • {face_id}", style="dim")


def display_face_results(results: dict, extract_faces: bool, output_dir: Optional[Path]):
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
        if result.get('success', False):
            face_count = len(result.get('faces', []))
            total_faces += face_count
            successful_images += 1
            
            table.add_row(
                Path(image_path).name,
                str(face_count),
                "✅",
                ""
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            table.add_row(
                Path(image_path).name,
                "0",
                "❌",
                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
            )
    
    console.print(table)
    console.print(f"\n📊 Summary: {successful_images}/{len(results)} images processed, {total_faces} faces detected")
    
    # Extract faces if requested
    if extract_faces and output_dir:
        console.print(f"\n💾 Extracting faces to {output_dir}...", style="blue")
        # TODO: Implement face extraction
        console.print("Face extraction not yet implemented", style="yellow")


def display_face_detection_results(results, total_images: int, total_faces_found: int, total_time: float, skipped_count: int = 0):
    """Display face detection results summary."""
    
    console.print(f"\n✅ Face detection complete!", style="green")
    console.print(f"📊 Processed {total_images} images")
    console.print(f"👥 Found {total_faces_found} faces")
    console.print(f"⏱️  Total detection time: {total_time:.2f}s")
    if total_images > 0:
        console.print(f"📈 Average time per image: {total_time/total_images:.2f}s")
    
    # Show skipped count
    if skipped_count > 0:
        console.print(f"⏭️  {skipped_count} images skipped (existing sidecar data)", style="blue")
    
    # Show processing info
    if total_images > 1:
        console.print(f"🔄 Used sequential processing for optimal performance", style="blue")


def display_benchmark_results(summary):
    """Display benchmark results summary."""
    from rich.table import Table
    
    console.print(f"\n🏆 Face Detection Benchmark Results", style="bold green")
    console.print(f"📊 Tested {summary.total_images} images with {len(summary.detectors)} detectors")
    
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
                str(stats['total_faces'])
            )
    
    console.print(perf_table)
    
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
                f"{stats['avg_face_size']:.0f}"
            )
    
    console.print(acc_table)


def display_detector_comparison(comparison):
    """Display detector comparison and recommendations."""
    console.print(f"\n🎯 Detector Comparison", style="bold blue")
    
    if comparison['fastest_detector']:
        console.print(f"⚡ Fastest: {comparison['fastest_detector']}", style="green")
    
    if comparison['most_accurate_detector']:
        console.print(f"🎯 Most Accurate: {comparison['most_accurate_detector']}", style="green")
    
    if comparison['most_reliable_detector']:
        console.print(f"🛡️  Most Reliable: {comparison['most_reliable_detector']}", style="green")
    
    if comparison['recommendations']:
        console.print(f"\n💡 Recommendations:", style="bold yellow")
        for rec in comparison['recommendations']:
            console.print(f"  • {rec}", style="yellow")
    
    # Overall scores
    if comparison['detailed_comparison']:
        console.print(f"\n📊 Overall Scores:", style="bold blue")
        for detector_name, details in comparison['detailed_comparison'].items():
            score = details.get('overall_score', 0)
            console.print(f"  {detector_name}: {score:.1f}/100", style="blue")


def display_face_extraction_results(extraction_results: dict, output_dir: Path):
    """Display face extraction results summary."""
    
    total_images = len(extraction_results)
    total_faces_extracted = 0
    successful_extractions = 0
    failed_extractions = 0
    
    # Count results
    for image_path, result in extraction_results.items():
        if result.get('success', False):
            successful_extractions += 1
            total_faces_extracted += result.get('faces_extracted', 0)
        else:
            failed_extractions += 1
    
    console.print(f"\n✅ Face extraction complete!", style="green")
    console.print(f"📊 Processed {total_images} images")
    console.print(f"👥 Extracted {total_faces_extracted} faces")
    console.print(f"📁 Saved to: {output_dir}", style="blue")
    
    if successful_extractions > 0:
        console.print(f"✅ {successful_extractions} images processed successfully", style="green")
    
    if failed_extractions > 0:
        console.print(f"❌ {failed_extractions} images failed to process", style="red")
        
        # Show details for failed extractions
        console.print(f"\n⚠️  Failed extractions:", style="yellow")
        for image_path, result in extraction_results.items():
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                console.print(f"  • {Path(image_path).name}: {error_msg}", style="red")
    
    if total_faces_extracted > 0:
        avg_faces = total_faces_extracted / successful_extractions if successful_extractions > 0 else 0
        console.print(f"📈 Average faces per image: {avg_faces:.1f}", style="blue")