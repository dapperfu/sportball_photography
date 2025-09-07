"""
Configuration loader for soccer photo sorter.

This module handles loading configuration from various sources including
files, environment variables, and command-line arguments.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger

from .settings import Settings


class ConfigLoader:
    """Configuration loader with support for multiple sources."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self._default_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "color": {
                "colors": {
                    "red": {"rgb_range": [200, 0, 0], "tolerance": 50},
                    "blue": {"rgb_range": [0, 0, 200], "tolerance": 50},
                    "green": {"rgb_range": [0, 200, 0], "tolerance": 50},
                    "yellow": {"rgb_range": [255, 255, 0], "tolerance": 50},
                    "white": {"rgb_range": [255, 255, 255], "tolerance": 30},
                    "black": {"rgb_range": [0, 0, 0], "tolerance": 30},
                    "orange": {"rgb_range": [255, 165, 0], "tolerance": 50},
                    "purple": {"rgb_range": [128, 0, 128], "tolerance": 50},
                },
                "confidence_threshold": 0.8,
                "blur_kernel_size": 5,
                "color_space": "RGB",
            },
            "detection": {
                "color_confidence": 0.8,
                "number_confidence": 0.7,
                "face_confidence": 0.75,
                "max_image_size": [1920, 1080],
                "min_image_size": [100, 100],
                "supported_formats": [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
            },
            "processing": {
                "max_threads": 4,
                "batch_size": 10,
                "max_memory_usage": 4096,
                "use_cuda": True,
                "gpu_memory_limit": None,
                "preserve_originals": True,
                "create_symlinks": True,
                "log_level": "INFO",
                "verbose": False,
            },
            "enable_color_detection": True,
            "enable_number_detection": True,
            "enable_face_detection": True,
            "dry_run": False,
        }
    
    def load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {file_path}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {file_path}: {e}")
            raise
    
    def load_from_env(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary with environment overrides
        """
        config = {}
        
        # Environment variable mappings
        env_mappings = {
            'SOCCER_PHOTO_SORTER_INPUT_PATH': 'input_path',
            'SOCCER_PHOTO_SORTER_OUTPUT_PATH': 'output_path',
            'SOCCER_PHOTO_SORTER_LOG_LEVEL': 'processing.log_level',
            'SOCCER_PHOTO_SORTER_VERBOSE': 'processing.verbose',
            'SOCCER_PHOTO_SORTER_USE_CUDA': 'processing.use_cuda',
            'SOCCER_PHOTO_SORTER_MAX_THREADS': 'processing.max_threads',
            'SOCCER_PHOTO_SORTER_COLOR_CONFIDENCE': 'detection.color_confidence',
            'SOCCER_PHOTO_SORTER_NUMBER_CONFIDENCE': 'detection.number_confidence',
            'SOCCER_PHOTO_SORTER_FACE_CONFIDENCE': 'detection.face_confidence',
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key.endswith('_confidence'):
                    value = float(value)
                elif config_key.endswith('_threads') or config_key.endswith('_usage'):
                    value = int(value)
                elif config_key.endswith('_verbose') or config_key.endswith('_cuda'):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                # Set nested configuration values
                keys = config_key.split('.')
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
        
        if config:
            logger.info("Loaded configuration from environment variables")
        
        return config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_config(self, 
                   file_path: Optional[Path] = None,
                   env_overrides: bool = True,
                   cli_overrides: Optional[Dict[str, Any]] = None) -> Settings:
        """
        Load configuration from multiple sources.
        
        Args:
            file_path: Optional path to configuration file
            env_overrides: Whether to apply environment variable overrides
            cli_overrides: Optional CLI argument overrides
            
        Returns:
            Settings object with merged configuration
        """
        configs = [self._default_config]
        
        # Load from file if specified
        if file_path and file_path.exists():
            file_config = self.load_from_file(file_path)
            configs.append(file_config)
        
        # Load from environment variables
        if env_overrides:
            env_config = self.load_from_env()
            if env_config:
                configs.append(env_config)
        
        # Apply CLI overrides
        if cli_overrides:
            configs.append(cli_overrides)
        
        # Merge all configurations
        merged_config = self.merge_configs(*configs)
        
        # Create Settings object
        settings = Settings.from_dict(merged_config)
        
        logger.info("Configuration loaded successfully")
        return settings
    
    def save_config(self, settings: Settings, file_path: Path) -> None:
        """
        Save settings to configuration file.
        
        Args:
            settings: Settings object to save
            file_path: Path to save configuration file
        """
        settings.save_to_file(file_path)
        logger.info(f"Configuration saved to {file_path}")
    
    def create_default_config_file(self, file_path: Path) -> None:
        """
        Create default configuration file.
        
        Args:
            file_path: Path to create configuration file
        """
        settings = Settings.from_dict(self._default_config)
        settings.save_to_file(file_path)
        logger.info(f"Default configuration file created at {file_path}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            Settings.from_dict(config)
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
