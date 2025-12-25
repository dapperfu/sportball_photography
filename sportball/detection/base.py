"""
Base Detection Framework

Tool-agnostic detection interface and base classes.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from loguru import logger


@dataclass
class DetectionResult:
    """
    Generic detection result that can be used by any detection tool.

    This provides a standardized format for all detection operations,
    making the system tool-agnostic.
    """

    # Core result data
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # Metadata
    tool_name: str = ""
    processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Tool-specific data (flexible structure)
    tool_data: Dict[str, Any] = field(default_factory=dict)

    # Image information
    image_path: Optional[Path] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tool_name": self.tool_name,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp,
            "tool_data": self.tool_data,
            "image_path": str(self.image_path) if self.image_path else None,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }

    def get_detection_count(self) -> int:
        """Get the number of detections (tool-agnostic)."""
        # Try common detection count fields
        if "count" in self.data:
            return self.data["count"]
        elif "faces" in self.data and isinstance(self.data["faces"], list):
            return len(self.data["faces"])
        elif "objects" in self.data and isinstance(self.data["objects"], list):
            return len(self.data["objects"])
        elif "detections" in self.data and isinstance(self.data["detections"], list):
            return len(self.data["detections"])
        return 0

    def get_confidence_scores(self) -> List[float]:
        """Get confidence scores from detections (tool-agnostic)."""
        scores = []

        # Try to extract confidence scores from various data structures
        for key in ["faces", "objects", "detections"]:
            if key in self.data and isinstance(self.data[key], list):
                for item in self.data[key]:
                    if isinstance(item, dict) and "confidence" in item:
                        scores.append(float(item["confidence"]))

        return scores


@dataclass
class DetectionConfig:
    """
    Configuration for detection tools.

    This provides a standardized way to configure any detection tool,
    making the system tool-agnostic and configurable.
    """

    # Tool identification
    tool_name: str = ""
    tool_version: str = "1.0.0"

    # Performance settings
    enable_gpu: bool = True
    batch_size: int = 8
    max_workers: Optional[int] = None

    # Detection parameters
    confidence_threshold: float = 0.5
    min_size: int = 64
    max_detections: Optional[int] = None

    # Processing options
    cache_enabled: bool = True
    save_sidecar: bool = True
    force_reprocess: bool = False

    # Tool-specific parameters (flexible)
    tool_params: Dict[str, Any] = field(default_factory=dict)

    # Validation settings
    validate_input: bool = True
    validate_output: bool = True

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Store unknown parameters in tool_params
                self.tool_params[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_name != "tool_params":
                result[field_name] = field_value
        result.update(self.tool_params)
        return result


class DetectionTool(ABC):
    """
    Abstract base class for all detection tools.

    This provides a tool-agnostic interface that any detection tool
    can implement, making the system flexible and extensible.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize detection tool.

        Args:
            config: Configuration for the tool
        """
        self.config = config or DetectionConfig()
        self.logger = logger.bind(component=f"detection_tool_{self.config.tool_name}")
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the detection tool.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def detect(self, image_path: Path, **kwargs) -> DetectionResult:
        """
        Perform detection on a single image.

        Args:
            image_path: Path to the image file
            **kwargs: Additional detection parameters

        Returns:
            DetectionResult with detection data
        """
        pass

    @abstractmethod
    def detect_batch(
        self, image_paths: List[Path], **kwargs
    ) -> Dict[str, DetectionResult]:
        """
        Perform detection on multiple images.

        Args:
            image_paths: List of image file paths
            **kwargs: Additional detection parameters

        Returns:
            Dictionary mapping image paths to DetectionResult objects
        """
        pass

    def validate_input(self, image_path: Path) -> bool:
        """
        Validate input image.

        Args:
            image_path: Path to the image file

        Returns:
            True if valid, False otherwise
        """
        if not self.config.validate_input:
            return True

        try:
            if not image_path.exists():
                self.logger.error(f"Image file does not exist: {image_path}")
                return False

            if not image_path.is_file():
                self.logger.error(f"Path is not a file: {image_path}")
                return False

            # Check file extension
            valid_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
            if image_path.suffix.lower() not in valid_extensions:
                self.logger.error(f"Unsupported image format: {image_path.suffix}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    def validate_output(self, result: DetectionResult) -> bool:
        """
        Validate detection result.

        Args:
            result: Detection result to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.config.validate_output:
            return True

        try:
            # Basic validation
            if not isinstance(result, DetectionResult):
                self.logger.error("Result is not a DetectionResult instance")
                return False

            if not isinstance(result.success, bool):
                self.logger.error("Result.success must be a boolean")
                return False

            if not isinstance(result.data, dict):
                self.logger.error("Result.data must be a dictionary")
                return False

            # Check for required fields based on success status
            if result.success:
                if not result.data:
                    self.logger.warning("Successful result has empty data")

            return True

        except Exception as e:
            self.logger.error(f"Output validation failed: {e}")
            return False

    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about the detection tool.

        Returns:
            Dictionary with tool information
        """
        return {
            "name": self.config.tool_name,
            "version": self.config.tool_version,
            "initialized": self._initialized,
            "config": self.config.to_dict(),
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update tool configuration.

        Args:
            new_config: New configuration parameters
        """
        self.config.update_from_dict(new_config)
        self.logger.info(f"Updated configuration for {self.config.tool_name}")

    def _ensure_initialized(self) -> bool:
        """
        Ensure tool is initialized.

        Returns:
            True if initialized, False otherwise
        """
        if not self._initialized:
            self._initialized = self.initialize()
            if not self._initialized:
                self.logger.error(f"Failed to initialize {self.config.tool_name}")
        return self._initialized


class DetectionToolFactory:
    """
    Factory for creating detection tools.

    This provides a centralized way to create and configure
    detection tools, making the system more modular.
    """

    _tool_registry: Dict[str, Type[DetectionTool]] = {}

    @classmethod
    def register_tool(cls, tool_name: str, tool_class: Type[DetectionTool]) -> None:
        """
        Register a detection tool class.

        Args:
            tool_name: Name of the tool
            tool_class: DetectionTool subclass
        """
        cls._tool_registry[tool_name] = tool_class
        logger.info(f"Registered detection tool: {tool_name}")

    @classmethod
    def create_tool(
        cls, tool_name: str, config: Optional[DetectionConfig] = None
    ) -> Optional[DetectionTool]:
        """
        Create a detection tool instance.

        Args:
            tool_name: Name of the tool to create
            config: Configuration for the tool

        Returns:
            DetectionTool instance or None if not found
        """
        if tool_name not in cls._tool_registry:
            logger.error(f"Unknown detection tool: {tool_name}")
            return None

        tool_class = cls._tool_registry[tool_name]
        tool_instance = tool_class(config)

        logger.info(f"Created detection tool: {tool_name}")
        return tool_instance

    @classmethod
    def list_tools(cls) -> List[str]:
        """
        List all registered detection tools.

        Returns:
            List of tool names
        """
        return list(cls._tool_registry.keys())

    @classmethod
    def get_tool_info(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information or None if not found
        """
        if tool_name not in cls._tool_registry:
            return None

        tool_class = cls._tool_registry[tool_name]
        return {
            "name": tool_name,
            "class": tool_class.__name__,
            "module": tool_class.__module__,
            "docstring": tool_class.__doc__,
        }
