"""
Face Detection Benchmark Module

Benchmarking interface for comparing different face detection methods.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
import numpy as np

from .face import FaceDetector, InsightFaceDetector


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""

    detector_name: str
    image_path: str
    faces_detected: int
    processing_time: float
    success: bool
    error: Optional[str] = None
    confidence_scores: Optional[List[float]] = None
    face_sizes: Optional[List[Tuple[int, int]]] = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results across all detectors."""

    total_images: int
    detectors: List[str]
    results: Dict[str, List[BenchmarkResult]]
    performance_stats: Dict[str, Dict[str, float]]
    accuracy_stats: Dict[str, Dict[str, float]]


class FaceDetectionBenchmark:
    """
    Benchmark interface for comparing face detection methods.
    """

    def __init__(
        self,
        enable_gpu: bool = True,
        confidence_threshold: float = 0.5,
        min_face_size: int = 64,
    ):
        """
        Initialize face detection benchmark.

        Args:
            enable_gpu: Whether to enable GPU acceleration
            confidence_threshold: Detection confidence threshold
            min_face_size: Minimum face size in pixels
        """
        self.enable_gpu = enable_gpu
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size

        # Initialize logger
        self.logger = logger.bind(component="face_benchmark")

        # Initialize detectors
        self.detectors = {}
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize all available face detectors."""
        try:
            # Initialize flexible FaceDetector (InsightFace + face_recognition)
            self.detectors["flexible_detector"] = FaceDetector(
                enable_gpu=self.enable_gpu,
                confidence_threshold=self.confidence_threshold,
                min_face_size=self.min_face_size,
            )
            self.logger.info(
                "Initialized flexible detector (InsightFace + face_recognition)"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize flexible detector: {e}")

        try:
            # Initialize InsightFace detector
            self.detectors["insightface"] = InsightFaceDetector(
                enable_gpu=self.enable_gpu,
                confidence_threshold=self.confidence_threshold,
                min_face_size=self.min_face_size,
            )
            self.logger.info("Initialized InsightFace detector")
        except Exception as e:
            self.logger.warning(f"Failed to initialize InsightFace detector: {e}")

        self.logger.info(f"Initialized {len(self.detectors)} face detectors")

    def benchmark_single_image(
        self, image_path: Path, detectors: Optional[List[str]] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark face detection on a single image.

        Args:
            image_path: Path to the image file
            detectors: List of detector names to test (None for all)

        Returns:
            Dictionary mapping detector names to benchmark results
        """
        if detectors is None:
            detectors = list(self.detectors.keys())

        results = {}

        for detector_name in detectors:
            if detector_name not in self.detectors:
                self.logger.warning(f"Detector {detector_name} not available")
                continue

            detector = self.detectors[detector_name]

            try:
                # Measure processing time
                start_time = time.time()

                # Detect faces
                detection_result = detector.detect_faces(
                    image_path,
                    confidence=self.confidence_threshold,
                    min_faces=0,  # Allow 0 faces for benchmark
                    face_size=self.min_face_size,
                )

                processing_time = time.time() - start_time

                # Extract confidence scores and face sizes
                confidence_scores = []
                face_sizes = []

                if detection_result.success:
                    for face in detection_result.faces:
                        confidence_scores.append(face.confidence)
                        w, h = face.bbox[2], face.bbox[3]
                        face_sizes.append((w, h))

                # Create benchmark result
                benchmark_result = BenchmarkResult(
                    detector_name=detector_name,
                    image_path=str(image_path),
                    faces_detected=detection_result.face_count,
                    processing_time=processing_time,
                    success=detection_result.success,
                    error=detection_result.error,
                    confidence_scores=confidence_scores if confidence_scores else None,
                    face_sizes=face_sizes if face_sizes else None,
                )

                results[detector_name] = benchmark_result

            except Exception as e:
                self.logger.error(
                    f"Benchmark failed for {detector_name} on {image_path}: {e}"
                )
                results[detector_name] = BenchmarkResult(
                    detector_name=detector_name,
                    image_path=str(image_path),
                    faces_detected=0,
                    processing_time=0.0,
                    success=False,
                    error=str(e),
                )

        return results

    def benchmark_batch(
        self,
        image_paths: List[Path],
        detectors: Optional[List[str]] = None,
        max_images: Optional[int] = None,
    ) -> BenchmarkSummary:
        """
        Benchmark face detection on multiple images.

        Args:
            image_paths: List of image paths to test
            detectors: List of detector names to test (None for all)
            max_images: Maximum number of images to test

        Returns:
            BenchmarkSummary containing all results
        """
        if detectors is None:
            detectors = list(self.detectors.keys())

        if max_images:
            image_paths = image_paths[:max_images]

        self.logger.info(
            f"Starting benchmark on {len(image_paths)} images with {len(detectors)} detectors"
        )

        # Initialize results storage
        all_results = {detector_name: [] for detector_name in detectors}

        # Process each image
        for i, image_path in enumerate(image_paths):
            self.logger.info(
                f"Processing image {i + 1}/{len(image_paths)}: {image_path.name}"
            )

            image_results = self.benchmark_single_image(image_path, detectors)

            # Store results
            for detector_name, result in image_results.items():
                all_results[detector_name].append(result)

        # Calculate performance and accuracy statistics
        performance_stats = self._calculate_performance_stats(all_results)
        accuracy_stats = self._calculate_accuracy_stats(all_results)

        return BenchmarkSummary(
            total_images=len(image_paths),
            detectors=detectors,
            results=all_results,
            performance_stats=performance_stats,
            accuracy_stats=accuracy_stats,
        )

    def _calculate_performance_stats(
        self, results: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance statistics for each detector."""
        stats = {}

        for detector_name, detector_results in results.items():
            if not detector_results:
                continue

            # Filter successful results
            successful_results = [r for r in detector_results if r.success]

            if not successful_results:
                stats[detector_name] = {
                    "avg_time": 0.0,
                    "min_time": 0.0,
                    "max_time": 0.0,
                    "std_time": 0.0,
                    "success_rate": 0.0,
                    "total_faces": 0,
                    "avg_faces_per_image": 0.0,
                }
                continue

            # Calculate timing statistics
            times = [r.processing_time for r in successful_results]
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)

            # Calculate success rate
            success_rate = len(successful_results) / len(detector_results)

            # Calculate face detection statistics
            total_faces = sum(r.faces_detected for r in successful_results)
            avg_faces_per_image = (
                total_faces / len(successful_results) if successful_results else 0
            )

            stats[detector_name] = {
                "avg_time": float(avg_time),
                "min_time": float(min_time),
                "max_time": float(max_time),
                "std_time": float(std_time),
                "success_rate": float(success_rate),
                "total_faces": int(total_faces),
                "avg_faces_per_image": float(avg_faces_per_image),
            }

        return stats

    def _calculate_accuracy_stats(
        self, results: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate accuracy statistics for each detector."""
        stats = {}

        for detector_name, detector_results in results.items():
            if not detector_results:
                continue

            # Filter successful results
            successful_results = [r for r in detector_results if r.success]

            if not successful_results:
                stats[detector_name] = {
                    "avg_confidence": 0.0,
                    "min_confidence": 0.0,
                    "max_confidence": 0.0,
                    "std_confidence": 0.0,
                    "avg_face_size": 0.0,
                    "min_face_size": 0.0,
                    "max_face_size": 0.0,
                }
                continue

            # Calculate confidence statistics
            all_confidences = []
            all_face_sizes = []

            for result in successful_results:
                if result.confidence_scores:
                    all_confidences.extend(result.confidence_scores)
                if result.face_sizes:
                    # Calculate face area (width * height)
                    face_areas = [w * h for w, h in result.face_sizes]
                    all_face_sizes.extend(face_areas)

            if all_confidences:
                avg_confidence = np.mean(all_confidences)
                min_confidence = np.min(all_confidences)
                max_confidence = np.max(all_confidences)
                std_confidence = np.std(all_confidences)
            else:
                avg_confidence = min_confidence = max_confidence = std_confidence = 0.0

            if all_face_sizes:
                avg_face_size = np.mean(all_face_sizes)
                min_face_size = np.min(all_face_sizes)
                max_face_size = np.max(all_face_sizes)
            else:
                avg_face_size = min_face_size = max_face_size = 0.0

            stats[detector_name] = {
                "avg_confidence": float(avg_confidence),
                "min_confidence": float(min_confidence),
                "max_confidence": float(max_confidence),
                "std_confidence": float(std_confidence),
                "avg_face_size": float(avg_face_size),
                "min_face_size": float(min_face_size),
                "max_face_size": float(max_face_size),
            }

        return stats

    def save_benchmark_results(
        self, summary: BenchmarkSummary, output_path: Path
    ) -> None:
        """
        Save benchmark results to JSON file.

        Args:
            summary: Benchmark summary to save
            output_path: Path to save the results
        """
        try:
            # Convert dataclasses to dictionaries for JSON serialization
            results_dict = {}
            for detector_name, detector_results in summary.results.items():
                results_dict[detector_name] = [
                    asdict(result) for result in detector_results
                ]

            output_data = {
                "summary": {
                    "total_images": summary.total_images,
                    "detectors": summary.detectors,
                    "performance_stats": summary.performance_stats,
                    "accuracy_stats": summary.accuracy_stats,
                },
                "detailed_results": results_dict,
                "benchmark_info": {
                    "confidence_threshold": self.confidence_threshold,
                    "min_face_size": self.min_face_size,
                    "enable_gpu": self.enable_gpu,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            self.logger.info(f"Benchmark results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save benchmark results: {e}")

    def load_benchmark_results(self, input_path: Path) -> BenchmarkSummary:
        """
        Load benchmark results from JSON file.

        Args:
            input_path: Path to load the results from

        Returns:
            BenchmarkSummary containing loaded results
        """
        try:
            with open(input_path, "r") as f:
                data = json.load(f)

            # Reconstruct results
            results = {}
            for detector_name, detector_results in data["detailed_results"].items():
                results[detector_name] = [
                    BenchmarkResult(**result) for result in detector_results
                ]

            return BenchmarkSummary(
                total_images=data["summary"]["total_images"],
                detectors=data["summary"]["detectors"],
                results=results,
                performance_stats=data["summary"]["performance_stats"],
                accuracy_stats=data["summary"]["accuracy_stats"],
            )

        except Exception as e:
            self.logger.error(f"Failed to load benchmark results: {e}")
            raise

    def compare_detectors(self, summary: BenchmarkSummary) -> Dict[str, Any]:
        """
        Compare detectors and provide recommendations.

        Args:
            summary: Benchmark summary to analyze

        Returns:
            Dictionary containing comparison analysis
        """
        comparison = {
            "fastest_detector": None,
            "most_accurate_detector": None,
            "most_reliable_detector": None,
            "recommendations": [],
            "detailed_comparison": {},
        }

        if not summary.performance_stats:
            return comparison

        # Find fastest detector (lowest average time)
        fastest_time = float("inf")
        fastest_detector = None

        # Find most accurate detector (highest average confidence)
        highest_confidence = 0.0
        most_accurate_detector = None

        # Find most reliable detector (highest success rate)
        highest_success_rate = 0.0
        most_reliable_detector = None

        for detector_name, stats in summary.performance_stats.items():
            # Speed comparison
            if stats["avg_time"] < fastest_time:
                fastest_time = stats["avg_time"]
                fastest_detector = detector_name

            # Reliability comparison
            if stats["success_rate"] > highest_success_rate:
                highest_success_rate = stats["success_rate"]
                most_reliable_detector = detector_name

            # Accuracy comparison
            if detector_name in summary.accuracy_stats:
                accuracy_stats = summary.accuracy_stats[detector_name]
                if accuracy_stats["avg_confidence"] > highest_confidence:
                    highest_confidence = accuracy_stats["avg_confidence"]
                    most_accurate_detector = detector_name

        comparison["fastest_detector"] = fastest_detector
        comparison["most_accurate_detector"] = most_accurate_detector
        comparison["most_reliable_detector"] = most_reliable_detector

        # Generate recommendations
        recommendations = []

        if fastest_detector:
            recommendations.append(
                f"Fastest detector: {fastest_detector} ({fastest_time:.3f}s avg)"
            )

        if most_accurate_detector:
            recommendations.append(
                f"Most accurate detector: {most_accurate_detector} ({highest_confidence:.3f} avg confidence)"
            )

        if most_reliable_detector:
            recommendations.append(
                f"Most reliable detector: {most_reliable_detector} ({highest_success_rate:.1%} success rate)"
            )

        # Detailed comparison
        for detector_name in summary.detectors:
            if detector_name in summary.performance_stats:
                perf_stats = summary.performance_stats[detector_name]
                acc_stats = summary.accuracy_stats.get(detector_name, {})

                comparison["detailed_comparison"][detector_name] = {
                    "performance": perf_stats,
                    "accuracy": acc_stats,
                    "overall_score": self._calculate_overall_score(
                        perf_stats, acc_stats
                    ),
                }

        comparison["recommendations"] = recommendations

        return comparison

    def _calculate_overall_score(
        self, perf_stats: Dict[str, float], acc_stats: Dict[str, float]
    ) -> float:
        """
        Calculate overall score for a detector based on performance and accuracy.

        Args:
            perf_stats: Performance statistics
            acc_stats: Accuracy statistics

        Returns:
            Overall score (0-100)
        """
        # Weighted scoring system
        speed_score = max(
            0, 100 - (perf_stats["avg_time"] * 100)
        )  # Lower time = higher score
        reliability_score = (
            perf_stats["success_rate"] * 100
        )  # Higher success rate = higher score
        accuracy_score = (
            acc_stats.get("avg_confidence", 0) * 100
        )  # Higher confidence = higher score

        # Weighted average (40% speed, 30% reliability, 30% accuracy)
        overall_score = (
            speed_score * 0.4 + reliability_score * 0.3 + accuracy_score * 0.3
        )

        return min(100, max(0, overall_score))
