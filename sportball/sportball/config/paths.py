"""
Configuration Paths Utility

Provides utilities for managing configuration and model file paths,
including the ~/.config/sportball/ directory structure.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import os
from pathlib import Path
from typing import Optional


def get_config_dir() -> Path:
    """
    Get the sportball configuration directory.
    
    Returns:
        Path to ~/.config/sportball/ directory
    """
    config_home = os.environ.get('XDG_CONFIG_HOME')
    if config_home:
        config_dir = Path(config_home) / 'sportball'
    else:
        config_dir = Path.home() / '.config' / 'sportball'
    
    # Ensure directory exists
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_models_dir() -> Path:
    """
    Get the sportball models directory.
    
    Models are stored in ~/.config/sportball/models/ for cross-project sharing.
    
    Returns:
        Path to ~/.config/sportball/models/ directory
    """
    models_dir = get_config_dir() / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_model_path(model_name: str) -> Path:
    """
    Get the full path to a model file.
    
    Args:
        model_name: Name of the model file (e.g., 'yolov8n.pt')
        
    Returns:
        Path to the model file in ~/.config/sportball/models/
    """
    return get_models_dir() / model_name


def ensure_model_downloaded(model_name: str, download_if_missing: bool = True) -> Path:
    """
    Ensure a model file exists, downloading it if necessary.
    
    This function checks if a model exists at the config path. If not and
    download_if_missing is True, it will use Ultralytics' auto-download
    feature to download the model, then copy it to the config directory.
    
    Args:
        model_name: Name of the model file (e.g., 'yolov8n.pt')
        download_if_missing: Whether to download the model if it doesn't exist
        
    Returns:
        Path to the model file (guaranteed to exist if download_if_missing is True)
        
    Raises:
        FileNotFoundError: If model doesn't exist and download_if_missing is False
        ImportError: If ultralytics is not available
    """
    model_path = get_model_path(model_name)
    
    # If model exists, return the path
    if model_path.exists():
        return model_path
    
    # If download is disabled, raise an error
    if not download_if_missing:
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Set download_if_missing=True to automatically download it."
        )
    
    # Try to import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package is required to download models. "
            "Install with: pip install ultralytics"
        )
    
    # Use Ultralytics' auto-download by passing just the model name
    # Ultralytics downloads to its cache directory, then we copy to our config dir
    import shutil
    import logging as std_logging
    
    # Suppress Ultralytics download messages temporarily
    original_level = std_logging.getLogger('ultralytics').level
    std_logging.getLogger('ultralytics').setLevel(std_logging.WARNING)
    
    try:
        # This will trigger Ultralytics' auto-download
        # Ultralytics downloads to current directory when given just a model name
        original_cwd = os.getcwd()
        downloaded_path = None
        
        try:
            # Change to a temp directory to download there, then we'll move it
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                # This triggers Ultralytics' auto-download
                model = YOLO(model_name)
                
                # Ultralytics downloads to current directory (tmpdir in this case)
                downloaded_path = Path(tmpdir) / model_name
                
                if downloaded_path.exists():
                    # Copy to config directory
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(downloaded_path), str(model_path))
                    return model_path
                else:
                    # Fallback: check if model has path attributes
                    if hasattr(model, 'ckpt_path') and model.ckpt_path:
                        downloaded_path = Path(model.ckpt_path)
                    elif hasattr(model, 'weights') and model.weights:
                        downloaded_path = Path(model.weights)
                    
                    if downloaded_path and downloaded_path.exists():
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(downloaded_path), str(model_path))
                        return model_path
        finally:
            os.chdir(original_cwd)
        
        # If we get here, the download didn't work as expected
        raise FileNotFoundError(
            f"Model {model_name} was downloaded but not found at expected location. "
            f"Please check Ultralytics cache or download manually."
        )
    finally:
        std_logging.getLogger('ultralytics').setLevel(original_level)


def get_cache_dir() -> Path:
    """
    Get the sportball cache directory.
    
    Returns:
        Path to ~/.config/sportball/cache/ directory
    """
    cache_dir = get_config_dir() / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_data_dir() -> Path:
    """
    Get the sportball data directory.
    
    Returns:
        Path to ~/.config/sportball/data/ directory
    """
    data_dir = get_config_dir() / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

