"""
Detection Integration Module

Integration layer between the tool-agnostic detection framework and existing sportball modules.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from loguru import logger

from .base import DetectionConfig, DetectionResult
from .registry import DetectionRegistry
from .adapters import (
    FaceDetectionAdapter,
    ObjectDetectionAdapter,
    QualityAssessmentAdapter,
)
from .parallel_validator import ParallelJSONValidator
from .rust_performance import RustPerformanceModule, RustPerformanceConfig


class DetectionIntegration:
    """
    Integration layer for tool-agnostic detection.

    This class provides a bridge between the existing sportball modules
    and the new tool-agnostic detection framework.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        enable_rust: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize detection integration.

        Args:
            base_dir: Base directory for operations
            enable_rust: Whether to enable Rust performance improvements
            max_workers: Maximum number of parallel workers
        """
        self.base_dir = base_dir or Path.cwd()
        self.logger = logger.bind(component="detection_integration")

        # Initialize detection registry
        self.registry = DetectionRegistry()

        # Initialize parallel validator
        self.validator = ParallelJSONValidator(max_workers=max_workers)

        # Initialize Rust performance module
        self.rust_module = None
        if enable_rust:
            rust_config = RustPerformanceConfig(
                enable_rust=True, max_workers=max_workers or 16
            )
            self.rust_module = RustPerformanceModule(rust_config)

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default detection tools."""
        # Register face detection tools
        self.registry.register_tool(
            "face_detection_insightface",
            FaceDetectionAdapter,
            DetectionConfig(
                tool_name="face_detection_insightface",
                tool_version="1.0.0",
                tool_params={"detector_type": "insightface", "model_name": "buffalo_l"},
            ),
        )

        self.registry.register_tool(
            "face_detection_opencv",
            FaceDetectionAdapter,
            DetectionConfig(
                tool_name="face_detection_opencv",
                tool_version="1.0.0",
                tool_params={"detector_type": "opencv"},
            ),
        )

        # Register object detection tool
        self.registry.register_tool(
            "object_detection_yolov8",
            ObjectDetectionAdapter,
            DetectionConfig(
                tool_name="object_detection_yolov8",
                tool_version="1.0.0",
                tool_params={"model_path": "yolov8n.pt", "border_padding": 0.25},
            ),
        )

        # Register quality assessment tool
        self.registry.register_tool(
            "quality_assessment",
            QualityAssessmentAdapter,
            DetectionConfig(tool_name="quality_assessment", tool_version="1.0.0"),
        )

        self.logger.info("Registered default detection tools")

    def detect_with_tool(
        self,
        tool_name: str,
        image_paths: Union[Path, List[Path]],
        config_override: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[DetectionResult, Dict[str, DetectionResult]]:
        """
        Perform detection using a specific tool.

        Args:
            tool_name: Name of the detection tool
            image_paths: Single image path or list of image paths
            config_override: Optional configuration overrides
            **kwargs: Additional detection parameters

        Returns:
            Detection result(s)
        """
        # Normalize input
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
            single_image = True
        else:
            single_image = False

        # Get tool instance
        tool = self.registry.get_tool(tool_name, config_override)
        if not tool:
            error_result = DetectionResult(
                success=False, error=f"Tool not found: {tool_name}", tool_name=tool_name
            )
            return (
                error_result
                if single_image
                else {str(path): error_result for path in image_paths}
            )

        try:
            if single_image:
                return tool.detect(image_paths[0], **kwargs)
            else:
                return tool.detect_batch(image_paths, **kwargs)

        except Exception as e:
            self.logger.error(f"Detection failed with tool {tool_name}: {e}")
            error_result = DetectionResult(
                success=False, error=str(e), tool_name=tool_name
            )
            return (
                error_result
                if single_image
                else {str(path): error_result for path in image_paths}
            )

    def validate_sidecar_files(
        self,
        directory: Path,
        operation_type: Optional[str] = None,
        use_rust: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Validate sidecar files in a directory.

        Args:
            directory: Directory to search for sidecar files
            operation_type: Optional operation type filter
            use_rust: Whether to use Rust implementation if available

        Returns:
            List of validation results
        """
        if use_rust and self.rust_module and self.rust_module.rust_available:
            try:
                # Find JSON files
                json_files = list(directory.glob("*.json"))

                if operation_type:
                    # Filter by operation type
                    filtered_files = []
                    for json_file in json_files:
                        try:
                            import json

                            with open(json_file, "r") as f:
                                data = json.load(f)

                            # Check if this file contains the specified operation type
                            if self._contains_operation_type(data, operation_type):
                                filtered_files.append(json_file)

                        except Exception:
                            # If we can't read the file, include it for validation
                            filtered_files.append(json_file)

                    json_files = filtered_files

                # Use Rust implementation
                return self.rust_module.parallel_json_validation(json_files)

            except Exception as e:
                self.logger.warning(
                    f"Rust validation failed, falling back to Python: {e}"
                )

        # Fall back to Python implementation
        results = self.validator.validate_sidecar_files(directory, operation_type)
        return [result.as_dict() for result in results]

    def validate_detection_results(
        self, results: Dict[str, DetectionResult], use_rust: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Validate detection results.

        Args:
            results: Dictionary of detection results
            use_rust: Whether to use Rust implementation if available

        Returns:
            List of validation results
        """
        if use_rust and self.rust_module and self.rust_module.rust_available:
            try:
                # Convert DetectionResult objects to file paths for Rust processing
                # This is a simplified approach - in practice, you might want to
                # serialize the results to temporary files
                file_paths = [Path(image_path) for image_path in results.keys()]
                return self.rust_module.parallel_json_validation(file_paths)

            except Exception as e:
                self.logger.warning(
                    f"Rust validation failed, falling back to Python: {e}"
                )

        # Fall back to Python implementation
        validation_results = self.validator.validate_detection_results(results)
        return [result.as_dict() for result in validation_results]

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available detection tools.

        Returns:
            Dictionary mapping tool names to their information
        """
        return self.registry.get_available_tools()

    def register_custom_tool(
        self, tool_name: str, tool_class, config: Optional[DetectionConfig] = None
    ) -> None:
        """
        Register a custom detection tool.

        Args:
            tool_name: Name of the tool
            tool_class: DetectionTool subclass
            config: Optional configuration for the tool
        """
        self.registry.register_tool(tool_name, tool_class, config)
        self.logger.info(f"Registered custom detection tool: {tool_name}")

    def update_tool_config(
        self, tool_name: str, config_updates: Dict[str, Any]
    ) -> bool:
        """
        Update configuration for a detection tool.

        Args:
            tool_name: Name of the tool
            config_updates: Configuration updates

        Returns:
            True if successful, False otherwise
        """
        return self.registry.update_tool_config(tool_name, config_updates)

    def get_performance_info(self) -> Dict[str, Any]:
        """
        Get performance information about the detection system.

        Returns:
            Dictionary with performance information
        """
        info = {
            "available_tools": list(self.get_available_tools().keys()),
            "validator_workers": self.validator.max_workers,
            "validator_use_processes": self.validator.use_processes,
        }

        if self.rust_module:
            info["rust_performance"] = self.rust_module.get_performance_info()

        return info

    def benchmark_performance(
        self, test_files: List[Path], iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark performance of the detection system.

        Args:
            test_files: List of test files to use
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with benchmark results
        """
        if self.rust_module:
            return self.rust_module.benchmark_performance(test_files, iterations)
        else:
            return {
                "rust_available": False,
                "message": "Rust performance module not available",
            }

    def _contains_operation_type(
        self, data: Dict[str, Any], operation_type: str
    ) -> bool:
        """Check if JSON data contains a specific operation type."""
        # Check direct keys
        if operation_type in data:
            return True

        # Check sidecar_info structure
        if "sidecar_info" in data and isinstance(data["sidecar_info"], dict):
            if data["sidecar_info"].get("operation_type") == operation_type:
                return True

        # Check nested structures
        for key in ["data", "result"]:
            if key in data and isinstance(data[key], dict):
                if self._contains_operation_type(data[key], operation_type):
                    return True

        return False

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.registry.clear_cache()
        self.validator.reset_statistics()
        self.logger.info("Cleared all detection caches")

    def reload_tool(self, tool_name: str) -> bool:
        """
        Reload a specific detection tool.

        Args:
            tool_name: Name of the tool to reload

        Returns:
            True if successful, False otherwise
        """
        return self.registry.reload_tool(tool_name)
