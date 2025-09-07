"""Configuration management for soccer photo sorter."""

from .settings import Settings, ColorConfig, DetectionConfig, ProcessingConfig
from .config_loader import ConfigLoader

__all__ = [
    "Settings",
    "ColorConfig", 
    "DetectionConfig",
    "ProcessingConfig",
    "ConfigLoader",
]
