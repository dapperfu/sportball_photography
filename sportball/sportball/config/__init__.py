"""Configuration management for sportball."""

from .settings import Settings, ColorConfig, DetectionConfig, ProcessingConfig
from .config_loader import ConfigLoader
from .paths import (
    get_config_dir,
    get_models_dir,
    get_model_path,
    ensure_model_downloaded,
    get_cache_dir,
    get_data_dir,
)

__all__ = [
    "Settings",
    "ColorConfig", 
    "DetectionConfig",
    "ProcessingConfig",
    "ConfigLoader",
    "get_config_dir",
    "get_models_dir",
    "get_model_path",
    "ensure_model_downloaded",
    "get_cache_dir",
    "get_data_dir",
]
