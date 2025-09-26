#!/usr/bin/env python3
"""
JSON Validation Benchmark Script

Benchmark Python vs Rust performance for JSON validation operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import time
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import statistics

# Import sportball modules
from sportball.detection.integration import DetectionIntegration
from sportball.detection.parallel_validator import ParallelJSONValidator
from sportball.detection.rust_performance import RustPerformanceModule, RustPerformanceConfig


def create_test_json_files(count: int, complexity: str = "medium") -> List[Path]:
    """
    Create test JSON files for benchmarking.
    
    Args:
        count: Number of files to create
        complexity: Complexity level ("simple", "medium", "complex")
        
    Returns:
        List of created file paths
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="sportball_json_benchmark_"))
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


def benchmark_python_sequential(files: List[Path]) -> Dict[str, Any]:
    """Benchmark Python sequential JSON validation."""
    start_time = time.time()
    
    results = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            results.append({
                'file_path': str(file_path),
                'is_valid': True,
                'file_size': file_path.stat().st_size
            })
        except Exception as e:
            results.append({
                'file_path': str(file_path),
                'is_valid': False,
                'error': str(e),
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            })
    
    total_time = time.time() - start_time
    
    return {
        'method': 'python_sequential',
        'total_time': total_time,
        'files_per_second': len(files) / total_time,
        'results_count': len(results),
        'valid_count': sum(1 for r in results if r['is_valid']),
        'invalid_count': sum(1 for r in results if not r['is_valid'])
    }


def benchmark_python_parallel(files: List[Path], max_workers: int = None) -> Dict[str, Any]:
    """Benchmark Python parallel JSON validation."""
    if max_workers is None:
        max_workers = min(cpu_count(), len(files), 32)
    
    def validate_single_file(file_path: Path) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return {
                'file_path': str(file_path),
                'is_valid': True,
                'file_size': file_path.stat().st_size
            }
        except Exception as e:
            return {
                'file_path': str(file_path),
                'is_valid': False,
                'error': str(e),
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            }
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(validate_single_file, files))
    
    total_time = time.time() - start_time
    
    return {
        'method': 'python_parallel',
        'max_workers': max_workers,
        'total_time': total_time,
        'files_per_second': len(files) / total_time,
        'results_count': len(results),
        'valid_count': sum(1 for r in results if r['is_valid']),
        'invalid_count': sum(1 for r in results if not r['is_valid'])
    }


def benchmark_sportball_validator(files: List[Path], max_workers: int = None) -> Dict[str, Any]:
    """Benchmark sportball's ParallelJSONValidator."""
    if max_workers is None:
        max_workers = min(cpu_count() * 2, 64)
    
    validator = ParallelJSONValidator(max_workers=max_workers, use_processes=True)
    
    start_time = time.time()
    results = validator.validate_json_files_parallel(files, show_progress=False)
    total_time = time.time() - start_time
    
    return {
        'method': 'sportball_validator',
        'max_workers': max_workers,
        'total_time': total_time,
        'files_per_second': len(files) / total_time,
        'results_count': len(results),
        'valid_count': sum(1 for r in results if r.is_valid),
        'invalid_count': sum(1 for r in results if not r.is_valid),
        'stats': validator.get_validation_statistics()
    }


def benchmark_rust_implementation(files: List[Path], max_workers: int = None) -> Dict[str, Any]:
    """Benchmark Rust implementation (if available)."""
    if max_workers is None:
        max_workers = 16
    
    rust_config = RustPerformanceConfig(
        enable_rust=True,
        max_workers=max_workers
    )
    rust_module = RustPerformanceModule(rust_config)
    
    if not rust_module.rust_available:
        return {
            'method': 'rust_implementation',
            'available': False,
            'error': 'Rust binary not available'
        }
    
    start_time = time.time()
    try:
        results = rust_module.parallel_json_validation(files)
        total_time = time.time() - start_time
        
        return {
            'method': 'rust_implementation',
            'max_workers': max_workers,
            'total_time': total_time,
            'files_per_second': len(files) / total_time,
            'results_count': len(results),
            'valid_count': sum(1 for r in results if r.get('is_valid', False)),
            'invalid_count': sum(1 for r in results if not r.get('is_valid', True)),
            'available': True
        }
    except Exception as e:
        return {
            'method': 'rust_implementation',
            'available': True,
            'error': str(e),
            'total_time': time.time() - start_time
        }


def run_comprehensive_benchmark(file_counts: List[int], complexity: str = "medium", 
                               iterations: int = 3) -> Dict[str, Any]:
    """
    Run comprehensive benchmark across different file counts and methods.
    
    Args:
        file_counts: List of file counts to test
        complexity: JSON complexity level
        iterations: Number of iterations per test
        
    Returns:
        Comprehensive benchmark results
    """
    results = {
        'file_counts': file_counts,
        'complexity': complexity,
        'iterations': iterations,
        'benchmarks': {}
    }
    
    for file_count in file_counts:
        print(f"\n🔍 Benchmarking {file_count} files ({complexity} complexity)...")
        
        # Create test files
        files, temp_dir = create_test_json_files(file_count, complexity)
        
        try:
            file_count_results = {
                'file_count': file_count,
                'iterations': {}
            }
            
            # Test different methods
            methods = [
                ('python_sequential', lambda: benchmark_python_sequential(files)),
                ('python_parallel', lambda: benchmark_python_parallel(files)),
                ('sportball_validator', lambda: benchmark_sportball_validator(files)),
                ('rust_implementation', lambda: benchmark_rust_implementation(files))
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
                    
                    file_count_results['iterations'][method_name] = {
                        'iterations': method_results,
                        'avg_time': statistics.mean(times),
                        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                        'avg_files_per_second': statistics.mean(files_per_sec),
                        'std_files_per_second': statistics.stdev(files_per_sec) if len(files_per_sec) > 1 else 0,
                        'min_time': min(times),
                        'max_time': max(times),
                        'success_rate': len(valid_results) / len(method_results)
                    }
                else:
                    file_count_results['iterations'][method_name] = {
                        'iterations': method_results,
                        'error': 'All iterations failed'
                    }
            
            results['benchmarks'][file_count] = file_count_results
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def print_benchmark_results(results: Dict[str, Any]):
    """Print benchmark results in a readable format."""
    print("\n" + "="*80)
    print("JSON VALIDATION BENCHMARK RESULTS")
    print("="*80)
    
    print(f"Complexity: {results['complexity']}")
    print(f"Iterations per test: {results['iterations']}")
    print(f"File counts tested: {results['file_counts']}")
    
    for file_count in results['file_counts']:
        if file_count not in results['benchmarks']:
            continue
            
        print(f"\n📊 {file_count} FILES")
        print("-" * 40)
        
        file_results = results['benchmarks'][file_count]
        
        # Sort methods by performance (files per second)
        method_performance = []
        for method_name, method_data in file_results['iterations'].items():
            if 'error' not in method_data:
                method_performance.append((
                    method_name,
                    method_data['avg_files_per_second'],
                    method_data['avg_time']
                ))
        
        method_performance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Method':<20} {'Files/sec':<12} {'Time (s)':<12} {'Success Rate':<12}")
        print("-" * 60)
        
        for method_name, files_per_sec, avg_time in method_performance:
            method_data = file_results['iterations'][method_name]
            success_rate = method_data['success_rate'] * 100
            
            print(f"{method_name:<20} {files_per_sec:<12.1f} {avg_time:<12.3f} {success_rate:<12.1f}%")
        
        # Show speedup comparison
        if len(method_performance) > 1:
            fastest = method_performance[0]
            print(f"\n🚀 Speedup vs slowest method:")
            for method_name, files_per_sec, avg_time in method_performance[1:]:
                speedup = files_per_sec / method_performance[-1][1]
                print(f"  {method_name}: {speedup:.1f}x faster")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark JSON validation performance")
    parser.add_argument("--file-counts", nargs="+", type=int, default=[10, 100, 1000, 5000],
                       help="File counts to test (default: 10 100 1000 5000)")
    parser.add_argument("--complexity", choices=["simple", "medium", "complex"], default="medium",
                       help="JSON complexity level (default: medium)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations per test (default: 3)")
    parser.add_argument("--output", type=str, help="Output file for results (JSON format)")
    
    args = parser.parse_args()
    
    print("🔬 Starting JSON validation benchmark...")
    print(f"File counts: {args.file_counts}")
    print(f"Complexity: {args.complexity}")
    print(f"Iterations: {args.iterations}")
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        file_counts=args.file_counts,
        complexity=args.complexity,
        iterations=args.iterations
    )
    
    # Print results
    print_benchmark_results(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()
