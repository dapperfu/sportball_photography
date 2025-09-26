"""
Face Detection Commands

CLI commands for face detection and recognition operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
import json
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..utils import get_core, find_image_files
from ...sidecar import Sidecar, OperationType
from ...detectors.face_benchmark import FaceDetectionBenchmark

console = Console()


def check_sidecar_file(image_file: Path, force: bool) -> Tuple[Path, bool]:
    """
    Check if a sidecar file exists and contains face detection data.
    
    Args:
        image_file: Path to the image file
        force: Whether to force processing even if sidecar exists
        
    Returns:
        Tuple of (image_file, should_skip)
    """
    try:
        # Resolve symlink if needed
        original_image_path = image_file.resolve() if image_file.is_symlink() else image_file
        json_path = original_image_path.parent / f"{original_image_path.stem}.json"
        
        if json_path.exists() and not force:
            # Check if JSON contains face detection data
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if ("Face_detector" in data and 
                    "metadata" in data["Face_detector"] and
                    "extraction_timestamp" in data["Face_detector"]["metadata"]):
                    return (image_file, True)  # Should skip (already processed)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        
        return (image_file, False)  # Should process
        
    except Exception:
        return (image_file, False)  # Should process on error


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
              help='Batch size for processing multiple images (default: 8)')
@click.option('--auto-tune', 'auto_tune',
              is_flag=True,
              help='Automatically tune GPU batch size for optimal performance')
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
    
    # Set environment variable for progress bar control
    import os
    if verbose >= 1:
        os.environ['SPORTBALL_VERBOSE'] = 'true'
    else:
        os.environ.pop('SPORTBALL_VERBOSE', None)
    
    core = get_core(ctx)
    
    # Find image files
    input_path = Path(input_pattern)
    recursive = not no_recursive
    
    if input_path.is_dir():
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
    
    # Auto-tune GPU batch size if requested
    if auto_tune and gpu:
        console.print("🔧 Auto-tuning GPU batch size...", style="blue")
        try:
            face_detector = core.get_face_detector()
            optimal_batch_size = face_detector.tune_gpu_batch_size(
                max_test_images=min(20, len(image_files)),
                max_batch_size=min(32, batch_size * 4)
            )
            batch_size = optimal_batch_size
            console.print(f"🎯 Using optimized batch size: {batch_size}", style="green")
        except Exception as e:
            console.print(f"⚠️  Auto-tuning failed, using default batch size: {e}", style="yellow")
    
    # Check for existing sidecar files
    skipped_files = []
    files_to_process = []
    
    for image_file in image_files:
        image_file, should_skip = check_sidecar_file(image_file, force)
        if should_skip:
            skipped_files.append(image_file)
        else:
            files_to_process.append(image_file)
    
    console.print(f"📊 Processing {len(files_to_process)} images ({len(skipped_files)} skipped)", style="blue")
    
    if not files_to_process:
        console.print("✅ All images already processed (use --force to reprocess)", style="green")
        return
    
    console.print(f"🔍 Starting face detection with batch size {batch_size}...", style="blue")
    
    # Use core's batch processing for face detection
    core = get_core(ctx)
    
    # Prepare detection parameters
    detection_kwargs = {
        'confidence': 0.5,
        'min_faces': 1,
        'face_size': 64,
        'batch_size': batch_size
    }
    
    # Perform batch detection
    results_dict = core.detect_faces(files_to_process, **detection_kwargs)
    
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
@click.pass_context
def extract(ctx: click.Context, input_path: Path, output: Optional[Path]):
    """
    Extract detected faces to separate images.
    
    INPUT_PATH should be a directory containing images with face detection sidecar files.
    """
    
    core = get_core(ctx)
    
    console.print(f"✂️  Extracting faces from {input_path}...", style="blue")
    
    # TODO: Implement face extraction
    console.print("Face extraction not yet implemented", style="yellow")


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
    
    # Set environment variable for progress bar control
    import os
    if verbose >= 1:
        os.environ['SPORTBALL_VERBOSE'] = 'true'
    else:
        os.environ.pop('SPORTBALL_VERBOSE', None)
    
    # Parse detector list
    detector_list = None
    if detectors:
        detector_list = [d.strip() for d in detectors.split(',')]
        console.print(f"🎯 Testing detectors: {', '.join(detector_list)}", style="blue")
    
    # Find image files
    input_path = Path(input_pattern)
    recursive = not no_recursive
    
    if input_path.is_dir():
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
    
    # Initialize benchmark
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
        
        # Save results
        benchmark.save_benchmark_results(summary, output)
        console.print(f"💾 Benchmark results saved to {output}", style="green")
        
        # Generate comparison
        comparison = benchmark.compare_detectors(summary)
        display_detector_comparison(comparison)
        
    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}", style="red")
        return


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
    
    # Show batch processing info
    if total_images > 1:
        console.print(f"🔄 Used batch processing for efficiency", style="blue")


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