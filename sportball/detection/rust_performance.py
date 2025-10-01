"""
Rust Performance Module

High-performance Rust-based implementations for detection operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger



@dataclass
class RustPerformanceConfig:
    """Configuration for Rust performance module."""

    enable_rust: bool = True
    rust_binary_path: Optional[Path] = None
    fallback_to_python: bool = True
    max_workers: int = 16
    chunk_size: int = 1000


class RustPerformanceModule:
    """
    Rust-based performance module for detection operations.

    This module provides high-performance implementations of common
    detection operations using Rust, with Python fallbacks.
    """

    def __init__(self, config: Optional[RustPerformanceConfig] = None):
        """
        Initialize Rust performance module.

        Args:
            config: Configuration for the module
        """
        self.config = config or RustPerformanceConfig()
        self.logger = logger.bind(component="rust_performance")
        self.rust_available = False

        # Check if Rust is available
        self._check_rust_availability()

    def _check_rust_availability(self) -> None:
        """Check if Rust binary is available."""
        if not self.config.enable_rust:
            self.rust_available = False
            return

        # Check for Rust binary
        if self.config.rust_binary_path and self.config.rust_binary_path.exists():
            self.rust_available = True
            self.logger.info(f"Rust binary found: {self.config.rust_binary_path}")
            return

        # Try to find Rust binary in PATH
        try:
            result = subprocess.run(
                ["which", "sportball-rust"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.config.rust_binary_path = Path(result.stdout.strip())
                self.rust_available = True
                self.logger.info(
                    f"Rust binary found in PATH: {self.config.rust_binary_path}"
                )
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Rust not available
        self.rust_available = False
        if self.config.fallback_to_python:
            self.logger.warning(
                "Rust not available, falling back to Python implementations"
            )
        else:
            self.logger.error("Rust not available and fallback disabled")

    def parallel_json_validation(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Perform parallel JSON validation using Rust.

        Args:
            file_paths: List of JSON file paths to validate

        Returns:
            List of validation results
        """
        if not self.rust_available:
            if self.config.fallback_to_python:
                return self._python_json_validation(file_paths)
            else:
                raise RuntimeError("Rust not available and fallback disabled")

        try:
            # Create temporary file with file paths
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for file_path in file_paths:
                    f.write(f"{file_path}\n")
                temp_file = f.name

            try:
                # Run Rust binary
                cmd = [
                    str(self.config.rust_binary_path),
                    "validate-json",
                    "--input",
                    temp_file,
                    "--output",
                    "-",  # Output to stdout
                    "--workers",
                    str(self.config.max_workers),
                ]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Rust validation failed: {result.stderr}")

                # Parse results
                import json

                results = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                return results

            finally:
                # Clean up temporary file
                os.unlink(temp_file)

        except Exception as e:
            self.logger.error(f"Rust JSON validation failed: {e}")
            if self.config.fallback_to_python:
                return self._python_json_validation(file_paths)
            else:
                raise

    def parallel_file_processing(
        self, file_paths: List[Path], operation: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform parallel file processing using Rust.

        Args:
            file_paths: List of file paths to process
            operation: Operation to perform
            **kwargs: Additional operation parameters

        Returns:
            List of processing results
        """
        if not self.rust_available:
            if self.config.fallback_to_python:
                return self._python_file_processing(file_paths, operation, **kwargs)
            else:
                raise RuntimeError("Rust not available and fallback disabled")

        try:
            # Create temporary file with file paths
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for file_path in file_paths:
                    f.write(f"{file_path}\n")
                temp_file = f.name

            try:
                # Run Rust binary
                cmd = [
                    str(self.config.rust_binary_path),
                    operation,
                    "--input",
                    temp_file,
                    "--output",
                    "-",  # Output to stdout
                    "--workers",
                    str(self.config.max_workers),
                ]

                # Add additional parameters
                for key, value in kwargs.items():
                    cmd.extend([f"--{key}", str(value)])

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Rust processing failed: {result.stderr}")

                # Parse results
                import json

                results = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                return results

            finally:
                # Clean up temporary file
                os.unlink(temp_file)

        except Exception as e:
            self.logger.error(f"Rust file processing failed: {e}")
            if self.config.fallback_to_python:
                return self._python_file_processing(file_paths, operation, **kwargs)
            else:
                raise

    def batch_image_analysis(
        self, image_paths: List[Path], analysis_type: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform batch image analysis using Rust.

        Args:
            image_paths: List of image file paths
            analysis_type: Type of analysis to perform
            **kwargs: Additional analysis parameters

        Returns:
            List of analysis results
        """
        if not self.rust_available:
            if self.config.fallback_to_python:
                return self._python_image_analysis(image_paths, analysis_type, **kwargs)
            else:
                raise RuntimeError("Rust not available and fallback disabled")

        try:
            # Create temporary file with image paths
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for image_path in image_paths:
                    f.write(f"{image_path}\n")
                temp_file = f.name

            try:
                # Run Rust binary
                cmd = [
                    str(self.config.rust_binary_path),
                    "analyze-images",
                    "--input",
                    temp_file,
                    "--output",
                    "-",  # Output to stdout
                    "--analysis",
                    analysis_type,
                    "--workers",
                    str(self.config.max_workers),
                ]

                # Add additional parameters
                for key, value in kwargs.items():
                    cmd.extend([f"--{key}", str(value)])

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=1200
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Rust image analysis failed: {result.stderr}")

                # Parse results
                import json

                results = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                return results

            finally:
                # Clean up temporary file
                os.unlink(temp_file)

        except Exception as e:
            self.logger.error(f"Rust image analysis failed: {e}")
            if self.config.fallback_to_python:
                return self._python_image_analysis(image_paths, analysis_type, **kwargs)
            else:
                raise

    def _python_json_validation(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Python fallback for JSON validation."""
        from .parallel_validator import ParallelJSONValidator

        validator = ParallelJSONValidator(max_workers=self.config.max_workers)
        results = validator.validate_json_files_parallel(
            file_paths, show_progress=False
        )

        return [result.as_dict() for result in results]

    def _python_file_processing(
        self, file_paths: List[Path], operation: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Python fallback for file processing."""
        # This would implement Python-based file processing
        # For now, return empty results
        self.logger.warning(f"Python fallback for {operation} not implemented")
        return []

    def _python_image_analysis(
        self, image_paths: List[Path], analysis_type: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Python fallback for image analysis."""
        # This would implement Python-based image analysis
        # For now, return empty results
        self.logger.warning(f"Python fallback for {analysis_type} not implemented")
        return []

    def get_performance_info(self) -> Dict[str, Any]:
        """
        Get performance information about the Rust module.

        Returns:
            Dictionary with performance information
        """
        return {
            "rust_available": self.rust_available,
            "rust_binary_path": str(self.config.rust_binary_path)
            if self.config.rust_binary_path
            else None,
            "fallback_enabled": self.config.fallback_to_python,
            "max_workers": self.config.max_workers,
            "chunk_size": self.config.chunk_size,
        }

    def benchmark_performance(
        self, test_files: List[Path], iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark performance of Rust vs Python implementations.

        Args:
            test_files: List of test files to use
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with benchmark results
        """
        import time

        results = {
            "rust_available": self.rust_available,
            "test_files_count": len(test_files),
            "iterations": iterations,
            "rust_times": [],
            "python_times": [],
            "rust_avg": 0.0,
            "python_avg": 0.0,
            "speedup": 0.0,
        }

        if not test_files:
            return results

        # Benchmark Rust implementation
        if self.rust_available:
            for i in range(iterations):
                start_time = time.time()
                try:
                    self.parallel_json_validation(test_files)
                    rust_time = time.time() - start_time
                    results["rust_times"].append(rust_time)
                except Exception as e:
                    self.logger.error(f"Rust benchmark iteration {i} failed: {e}")
                    results["rust_times"].append(float("inf"))

        # Benchmark Python implementation
        for i in range(iterations):
            start_time = time.time()
            try:
                self._python_json_validation(test_files)
                python_time = time.time() - start_time
                results["python_times"].append(python_time)
            except Exception as e:
                self.logger.error(f"Python benchmark iteration {i} failed: {e}")
                results["python_times"].append(float("inf"))

        # Calculate averages
        if results["rust_times"]:
            results["rust_avg"] = sum(
                t for t in results["rust_times"] if t != float("inf")
            ) / len(results["rust_times"])

        if results["python_times"]:
            results["python_avg"] = sum(
                t for t in results["python_times"] if t != float("inf")
            ) / len(results["python_times"])

        # Calculate speedup
        if results["rust_avg"] > 0 and results["python_avg"] > 0:
            results["speedup"] = results["python_avg"] / results["rust_avg"]

        return results
