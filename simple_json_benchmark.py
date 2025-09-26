#!/usr/bin/env python3
"""
Simple JSON Validation Benchmark

A lightweight benchmark for JSON validation performance without heavy dependencies.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import time
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import statistics


def create_test_json_files(count: int, complexity: str = "medium") -> tuple[List[Path], Path]:
    """
    Create test JSON files for benchmarking.
    
    Args:
        count: Number of files to create
        complexity: Complexity level ("simple", "medium", "complex")
        
    Returns:
        Tuple of (created file paths, temp directory)
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="json_benchmark_"))
    files = []
    
    # Define complexity levels
    if complexity == "simple":
        # Simple detection result
        template = {
            "success": True,
            "faces": [
                {"bbox": [100, 100, 50, 50], "confidence": 0.95},
                {"bbox": [200, 200, 60, 60], "confidence": 0.87}
            ],
            "face_count": 2,
            "processing_time": 0.123
        }
    elif complexity == "medium":
        # Medium complexity with metadata
        template = {
            "success": True,
            "faces": [
                {
                    "face_id": 0,
                    "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                    "confidence": 0.95,
                    "landmarks": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    "encoding": [0.1] * 128
                },
                {
                    "face_id": 1,
                    "bbox": {"x": 0.5, "y": 0.6, "width": 0.2, "height": 0.3},
                    "confidence": 0.87,
                    "landmarks": [[0.7, 0.8], [0.9, 0.1], [0.2, 0.3]],
                    "encoding": [0.2] * 128
                }
            ],
            "metadata": {
                "image_path": "/path/to/image.jpg",
                "image_width": 1920,
                "image_height": 1080,
                "faces_found": 2,
                "processing_time": 0.123,
                "extraction_timestamp": "2024-12-19T10:30:00Z",
                "detector": "insightface",
                "model_name": "buffalo_l"
            }
        }
    else:  # complex
        # Complex detection result with multiple operations
        template = {
            "sidecar_info": {
                "operation_type": "face_detection",
                "created_at": "2024-12-19T10:30:00Z",
                "image_path": "/path/to/image.jpg"
            },
            "face_detection": {
                "success": True,
                "faces": [
                    {
                        "face_id": 0,
                        "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                        "confidence": 0.95,
                        "landmarks": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]],
                        "encoding": [0.1] * 512
                    },
                    {
                        "face_id": 1,
                        "bbox": {"x": 0.5, "y": 0.6, "width": 0.2, "height": 0.3},
                        "confidence": 0.87,
                        "landmarks": [[0.7, 0.8], [0.9, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
                        "encoding": [0.2] * 512
                    }
                ],
                "metadata": {
                    "image_path": "/path/to/image.jpg",
                    "image_width": 1920,
                    "image_height": 1080,
                    "faces_found": 2,
                    "processing_time": 0.123,
                    "extraction_timestamp": "2024-12-19T10:30:00Z",
                    "detector": "insightface",
                    "model_name": "buffalo_l",
                    "face_size_threshold": 64,
                    "confidence_threshold": 0.5
                }
            },
            "object_detection": {
                "success": True,
                "objects": [
                    {
                        "class": "person",
                        "confidence": 0.92,
                        "bbox": [100, 100, 200, 300]
                    },
                    {
                        "class": "sports ball",
                        "confidence": 0.78,
                        "bbox": [300, 400, 50, 50]
                    }
                ],
                "object_count": 2
            }
        }
    
    # Create files
    for i in range(count):
        # Add some variation to each file
        data = json.loads(json.dumps(template))  # Deep copy
        if "faces" in data:
            for face in data["faces"]:
                face["confidence"] += (i % 10) * 0.01  # Vary confidence
        elif "face_detection" in data and "faces" in data["face_detection"]:
            for face in data["face_detection"]["faces"]:
                face["confidence"] += (i % 10) * 0.01
        
        file_path = temp_dir / f"test_{i:06d}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        files.append(file_path)
    
    return files, temp_dir


def validate_json_file(file_path: Path) -> Dict[str, Any]:
    """Validate a single JSON file."""
    start_time = time.time()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract some information
        detection_count = 0
        if "faces" in data:
            detection_count = len(data["faces"])
        elif "face_detection" in data and "faces" in data["face_detection"]:
            detection_count = len(data["face_detection"]["faces"])
        elif "objects" in data:
            detection_count = len(data["objects"])
        
        return {
            'file_path': str(file_path),
            'is_valid': True,
            'processing_time': time.time() - start_time,
            'file_size': file_path.stat().st_size,
            'detection_count': detection_count
        }
    except Exception as e:
        return {
            'file_path': str(file_path),
            'is_valid': False,
            'error': str(e),
            'processing_time': time.time() - start_time,
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'detection_count': 0
        }


def benchmark_sequential(files: List[Path]) -> Dict[str, Any]:
    """Benchmark sequential JSON validation."""
    start_time = time.time()
    
    results = []
    for file_path in files:
        result = validate_json_file(file_path)
        results.append(result)
    
    total_time = time.time() - start_time
    
    return {
        'method': 'sequential',
        'total_time': total_time,
        'files_per_second': len(files) / total_time,
        'results_count': len(results),
        'valid_count': sum(1 for r in results if r['is_valid']),
        'invalid_count': sum(1 for r in results if not r['is_valid']),
        'avg_processing_time': statistics.mean([r['processing_time'] for r in results])
    }


def benchmark_threaded(files: List[Path], max_workers: int = None) -> Dict[str, Any]:
    """Benchmark threaded JSON validation."""
    if max_workers is None:
        max_workers = min(cpu_count(), len(files), 32)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(validate_json_file, files))
    
    total_time = time.time() - start_time
    
    return {
        'method': 'threaded',
        'max_workers': max_workers,
        'total_time': total_time,
        'files_per_second': len(files) / total_time,
        'results_count': len(results),
        'valid_count': sum(1 for r in results if r['is_valid']),
        'invalid_count': sum(1 for r in results if not r['is_valid']),
        'avg_processing_time': statistics.mean([r['processing_time'] for r in results])
    }


def benchmark_multiprocess(files: List[Path], max_workers: int = None) -> Dict[str, Any]:
    """Benchmark multiprocess JSON validation."""
    if max_workers is None:
        max_workers = min(cpu_count(), len(files), 32)
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(validate_json_file, files))
    
    total_time = time.time() - start_time
    
    return {
        'method': 'multiprocess',
        'max_workers': max_workers,
        'total_time': total_time,
        'files_per_second': len(files) / total_time,
        'results_count': len(results),
        'valid_count': sum(1 for r in results if r['is_valid']),
        'invalid_count': sum(1 for r in results if not r['is_valid']),
        'avg_processing_time': statistics.mean([r['processing_time'] for r in results])
    }


def run_benchmark(file_count: int, complexity: str, iterations: int = 3) -> Dict[str, Any]:
    """Run benchmark for a specific file count."""
    print(f"\n🔍 Benchmarking {file_count} files ({complexity} complexity)...")
    
    # Create test files
    files, temp_dir = create_test_json_files(file_count, complexity)
    
    try:
        results = {
            'file_count': file_count,
            'complexity': complexity,
            'iterations': iterations,
            'methods': {}
        }
        
        # Test different methods
        methods = [
            ('sequential', lambda: benchmark_sequential(files)),
            ('threaded', lambda: benchmark_threaded(files)),
            ('multiprocess', lambda: benchmark_multiprocess(files))
        ]
        
        for method_name, method_func in methods:
            print(f"  Testing {method_name}...")
            
            method_results = []
            for i in range(iterations):
                try:
                    result = method_func()
                    method_results.append(result)
                except Exception as e:
                    print(f"    Iteration {i+1} failed: {e}")
                    method_results.append({
                        'method': method_name,
                        'error': str(e),
                        'total_time': float('inf')
                    })
            
            # Calculate statistics
            valid_results = [r for r in method_results if 'error' not in r]
            if valid_results:
                times = [r['total_time'] for r in valid_results]
                files_per_sec = [r['files_per_second'] for r in valid_results]
                
                results['methods'][method_name] = {
                    'avg_time': statistics.mean(times),
                    'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                    'avg_files_per_second': statistics.mean(files_per_sec),
                    'std_files_per_second': statistics.stdev(files_per_sec) if len(files_per_sec) > 1 else 0,
                    'min_time': min(times),
                    'max_time': max(times),
                    'success_rate': len(valid_results) / len(method_results),
                    'iterations': method_results
                }
            else:
                results['methods'][method_name] = {
                    'error': 'All iterations failed',
                    'iterations': method_results
                }
        
        return results
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def print_results(results: Dict[str, Any]):
    """Print benchmark results."""
    print(f"\n📊 RESULTS FOR {results['file_count']} FILES ({results['complexity']} complexity)")
    print("=" * 60)
    
    # Sort methods by performance
    method_performance = []
    for method_name, method_data in results['methods'].items():
        if 'error' not in method_data:
            method_performance.append((
                method_name,
                method_data['avg_files_per_second'],
                method_data['avg_time']
            ))
    
    method_performance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Method':<15} {'Files/sec':<12} {'Time (s)':<12} {'Success Rate':<12}")
    print("-" * 60)
    
    for method_name, files_per_sec, avg_time in method_performance:
        method_data = results['methods'][method_name]
        success_rate = method_data['success_rate'] * 100
        
        print(f"{method_name:<15} {files_per_sec:<12.1f} {avg_time:<12.3f} {success_rate:<12.1f}%")
    
    # Show speedup
    if len(method_performance) > 1:
        fastest = method_performance[0]
        slowest = method_performance[-1]
        speedup = fastest[1] / slowest[1]
        print(f"\n🚀 {fastest[0]} is {speedup:.1f}x faster than {slowest[0]}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Simple JSON validation benchmark")
    parser.add_argument("--file-counts", nargs="+", type=int, default=[10, 100, 1000],
                       help="File counts to test (default: 10 100 1000)")
    parser.add_argument("--complexity", choices=["simple", "medium", "complex"], default="medium",
                       help="JSON complexity level (default: medium)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations per test (default: 3)")
    
    args = parser.parse_args()
    
    print("🔬 Starting simple JSON validation benchmark...")
    print(f"File counts: {args.file_counts}")
    print(f"Complexity: {args.complexity}")
    print(f"Iterations: {args.iterations}")
    print(f"CPU cores: {cpu_count()}")
    
    all_results = []
    
    for file_count in args.file_counts:
        result = run_benchmark(file_count, args.complexity, args.iterations)
        all_results.append(result)
        print_results(result)
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("=" * 60)
    print(f"{'Files':<8} {'Best Method':<15} {'Files/sec':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for result in all_results:
        # Find best method
        best_method = None
        best_performance = 0
        
        for method_name, method_data in result['methods'].items():
            if 'error' not in method_data:
                if method_data['avg_files_per_second'] > best_performance:
                    best_performance = method_data['avg_files_per_second']
                    best_method = method_name
        
        if best_method:
            # Calculate speedup vs sequential
            sequential_perf = result['methods'].get('sequential', {}).get('avg_files_per_second', 0)
            speedup = best_performance / sequential_perf if sequential_perf > 0 else 1.0
            
            print(f"{result['file_count']:<8} {best_method:<15} {best_performance:<12.1f} {speedup:<10.1f}x")
    
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()
