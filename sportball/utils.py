"""Utility functions for the sportball package."""

from pathlib import Path
from typing import List
from PIL import Image, ImageOps
import logging

logger = logging.getLogger(__name__)


def load_image_with_exif_rotation(image_path: Path) -> Image.Image:
    """
    Load an image and apply EXIF rotation to ensure it's displayed correctly.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image with EXIF rotation applied
        
    Raises:
        Exception: If image cannot be loaded
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Apply EXIF rotation using ImageOps - this handles Orientation tag
        # This ensures the image is rotated according to EXIF metadata
        image = ImageOps.exif_transpose(image)
        
        # Convert to RGB for consistency
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
    except Exception as e:
        logger.error(f"Failed to load image with EXIF rotation {image_path}: {e}")
        raise


def find_image_files(input_path: Path, recursive: bool = True) -> List[Path]:
    """
    Find all image files in the given path.
    
    Args:
        input_path: Path to search (file or directory)
        recursive: Whether to search recursively (default: True)
        
    Returns:
        List of image file paths
    """
    # Supported image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"]
    
    # Check if input is a single file
    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            return [input_path]
        else:
            return []
    
    # Find images in directory
    image_files = []
    for ext in image_extensions:
        if recursive:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    return sorted(list(set(image_files)))

