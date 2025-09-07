"""
Configuration settings for soccer photo sorter.

This module defines the configuration structure using Pydantic models
for type safety and validation.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator
import json


class ColorConfig(BaseModel):
    """Configuration for color detection."""
    
    # Color categories with RGB ranges and tolerances
    colors: Dict[str, Dict[str, Any]] = Field(
        default={
            "red": {"rgb_range": [200, 0, 0], "tolerance": 50},
            "blue": {"rgb_range": [0, 0, 200], "tolerance": 50},
            "green": {"rgb_range": [0, 200, 0], "tolerance": 50},
            "yellow": {"rgb_range": [255, 255, 0], "tolerance": 50},
            "white": {"rgb_range": [255, 255, 255], "tolerance": 30},
            "black": {"rgb_range": [0, 0, 0], "tolerance": 30},
            "orange": {"rgb_range": [255, 165, 0], "tolerance": 50},
            "purple": {"rgb_range": [128, 0, 128], "tolerance": 50},
        },
        description="Color definitions with RGB ranges and tolerances"
    )
    
    # Detection parameters
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for color detection"
    )
    
    # Image processing parameters
    blur_kernel_size: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Kernel size for image blurring"
    )
    
    # Color space conversion
    color_space: str = Field(
        default="RGB",
        description="Color space for analysis (RGB, HSV, LAB)"
    )


class DetectionConfig(BaseModel):
    """Configuration for general detection parameters."""
    
    # Confidence thresholds
    color_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for color detection"
    )
    
    number_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for number detection"
    )
    
    face_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for face detection"
    )
    
    # Processing parameters
    max_image_size: Tuple[int, int] = Field(
        default=(1920, 1080),
        description="Maximum image size for processing"
    )
    
    min_image_size: Tuple[int, int] = Field(
        default=(100, 100),
        description="Minimum image size for processing"
    )
    
    # Supported file formats
    supported_formats: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
        description="Supported image file formats"
    )


class ProcessingConfig(BaseModel):
    """Configuration for processing parameters."""
    
    # Threading and parallel processing
    max_threads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum number of processing threads"
    )
    
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for processing images"
    )
    
    # Memory management
    max_memory_usage: int = Field(
        default=4096,
        ge=512,
        le=16384,
        description="Maximum memory usage in MB"
    )
    
    # CUDA settings
    use_cuda: bool = Field(
        default=True,
        description="Enable CUDA acceleration if available"
    )
    
    gpu_memory_limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="GPU memory limit in GB"
    )
    
    # Output settings
    preserve_originals: bool = Field(
        default=True,
        description="Preserve original files without modification"
    )
    
    create_symlinks: bool = Field(
        default=True,
        description="Create symbolic links instead of copying files"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    verbose: bool = Field(
        default=False,
        description="Enable verbose output"
    )


class Settings(BaseModel):
    """Main configuration settings for soccer photo sorter."""
    
    # Configuration sections
    color: ColorConfig = Field(default_factory=ColorConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    # Paths
    input_path: Optional[Path] = Field(
        default=None,
        description="Input directory path"
    )
    
    output_path: Optional[Path] = Field(
        default=None,
        description="Output directory path"
    )
    
    config_path: Optional[Path] = Field(
        default=None,
        description="Configuration file path"
    )
    
    # Processing modes
    enable_color_detection: bool = Field(
        default=True,
        description="Enable jersey color detection"
    )
    
    enable_number_detection: bool = Field(
        default=True,
        description="Enable jersey number detection"
    )
    
    enable_face_detection: bool = Field(
        default=True,
        description="Enable face detection and recognition"
    )
    
    # Dry run mode
    dry_run: bool = Field(
        default=False,
        description="Preview changes without creating directories"
    )
    
    @validator('input_path', 'output_path', 'config_path')
    def validate_paths(cls, v):
        """Validate path fields."""
        if v is not None:
            return Path(v)
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict()
    
    def to_json(self, file_path: Optional[Path] = None) -> str:
        """Convert settings to JSON string or save to file."""
        json_str = self.json(indent=2, exclude_none=True)
        if file_path:
            file_path.write_text(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Settings':
        """Create settings from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'Settings':
        """Create settings from JSON file."""
        json_str = file_path.read_text()
        return cls.from_json(json_str)
    
    def save_to_file(self, file_path: Path) -> None:
        """Save settings to JSON file."""
        self.to_json(file_path)
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update settings from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_color_config(self) -> ColorConfig:
        """Get color configuration."""
        return self.color
    
    def get_detection_config(self) -> DetectionConfig:
        """Get detection configuration."""
        return self.detection
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration."""
        return self.processing
