"""
File utilities for soccer photo sorter.

This module provides file handling, validation, and organization utilities
for image processing operations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set, Optional, Dict, Any, Tuple
from PIL import Image
import mimetypes
from loguru import logger


class ImageValidator:
    """Validator for image files."""
    
    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize image validator.
        
        Args:
            supported_formats: List of supported file extensions
        """
        self.supported_formats = supported_formats or [
            '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'
        ]
        self._supported_mime_types = self._get_supported_mime_types()
    
    def _get_supported_mime_types(self) -> Set[str]:
        """Get supported MIME types."""
        mime_types = set()
        for ext in self.supported_formats:
            mime_type, _ = mimetypes.guess_type(f"file{ext}")
            if mime_type:
                mime_types.add(mime_type)
        return mime_types
    
    def is_valid_image_file(self, file_path: Path) -> bool:
        """
        Check if file is a valid image.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is a valid image
        """
        if not file_path.exists():
            return False
        
        # Check file extension
        if file_path.suffix.lower() not in self.supported_formats:
            return False
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type not in self._supported_mime_types:
            return False
        
        # Try to open with PIL
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def get_image_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get image information.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Image information dictionary or None if invalid
        """
        if not self.is_valid_image_file(file_path):
            return None
        
        try:
            with Image.open(file_path) as img:
                return {
                    'path': file_path,
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'width': img.width,
                    'height': img.height,
                    'file_size': file_path.stat().st_size,
                }
        except Exception as e:
            logger.error(f"Error getting image info for {file_path}: {e}")
            return None
    
    def validate_image_size(self, 
                          file_path: Path, 
                          min_size: Tuple[int, int] = (100, 100),
                          max_size: Tuple[int, int] = (8000, 8000)) -> bool:
        """
        Validate image size.
        
        Args:
            file_path: Path to image file
            min_size: Minimum size (width, height)
            max_size: Maximum size (width, height)
            
        Returns:
            True if image size is valid
        """
        info = self.get_image_info(file_path)
        if not info:
            return False
        
        width, height = info['size']
        min_width, min_height = min_size
        max_width, max_height = max_size
        
        return (min_width <= width <= max_width and 
                min_height <= height <= max_height)


class FileUtils:
    """Utility class for file operations."""
    
    def __init__(self, validator: Optional[ImageValidator] = None):
        """
        Initialize file utils.
        
        Args:
            validator: Optional image validator
        """
        self.validator = validator or ImageValidator()
    
    def find_image_files(self, 
                        directory: Path, 
                        recursive: bool = True) -> List[Path]:
        """
        Find all image files in directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            
        Returns:
            List of image file paths
        """
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory not found: {directory}")
            return []
        
        image_files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and self.validator.is_valid_image_file(file_path):
                image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} image files in {directory}")
        return image_files
    
    def create_directory(self, directory: Path, parents: bool = True) -> bool:
        """
        Create directory if it doesn't exist.
        
        Args:
            directory: Directory path to create
            parents: Whether to create parent directories
            
        Returns:
            True if directory was created or already exists
        """
        try:
            directory.mkdir(parents=parents, exist_ok=True)
            logger.debug(f"Directory created/verified: {directory}")
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False
    
    def copy_file(self, 
                 source: Path, 
                 destination: Path, 
                 create_dirs: bool = True) -> bool:
        """
        Copy file to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            create_dirs: Whether to create destination directories
            
        Returns:
            True if copy was successful
        """
        try:
            if create_dirs:
                self.create_directory(destination.parent)
            
            shutil.copy2(source, destination)
            logger.debug(f"File copied: {source} -> {destination}")
            return True
        except Exception as e:
            logger.error(f"Error copying file {source} to {destination}: {e}")
            return False
    
    def create_symlink(self, 
                      source: Path, 
                      destination: Path, 
                      create_dirs: bool = True) -> bool:
        """
        Create symbolic link to source file.
        
        Args:
            source: Source file path
            destination: Destination link path
            create_dirs: Whether to create destination directories
            
        Returns:
            True if symlink was created successfully
        """
        try:
            if create_dirs:
                self.create_directory(destination.parent)
            
            # Remove existing file/link if it exists
            if destination.exists():
                destination.unlink()
            
            destination.symlink_to(source.resolve())
            logger.debug(f"Symlink created: {destination} -> {source}")
            return True
        except Exception as e:
            logger.error(f"Error creating symlink {destination} -> {source}: {e}")
            return False
    
    def organize_files(self, 
                      files: List[Path], 
                      output_dir: Path,
                      categories: Dict[str, List[Path]],
                      use_symlinks: bool = True,
                      dry_run: bool = False) -> Dict[str, int]:
        """
        Organize files into categorized directories.
        
        Args:
            files: List of files to organize
            output_dir: Base output directory
            categories: Dictionary mapping category names to file lists
            use_symlinks: Whether to use symbolic links
            dry_run: Whether to preview changes without creating files
            
        Returns:
            Dictionary with organization statistics
        """
        stats = {'total_files': len(files), 'organized': 0, 'errors': 0}
        
        for category, category_files in categories.items():
            if not category_files:
                continue
            
            category_dir = output_dir / category
            logger.info(f"Organizing {len(category_files)} files into {category_dir}")
            
            if not dry_run:
                self.create_directory(category_dir)
            
            for file_path in category_files:
                if file_path not in files:
                    continue
                
                destination = category_dir / file_path.name
                
                if dry_run:
                    logger.info(f"Would {'link' if use_symlinks else 'copy'} {file_path} -> {destination}")
                    stats['organized'] += 1
                else:
                    if use_symlinks:
                        success = self.create_symlink(file_path, destination)
                    else:
                        success = self.copy_file(file_path, destination)
                    
                    if success:
                        stats['organized'] += 1
                    else:
                        stats['errors'] += 1
        
        logger.info(f"Organization complete: {stats['organized']} files organized, {stats['errors']} errors")
        return stats
    
    def get_file_stats(self, directory: Path) -> Dict[str, Any]:
        """
        Get statistics about files in directory.
        
        Args:
            directory: Directory to analyze
            
        Returns:
            File statistics dictionary
        """
        if not directory.exists():
            return {'error': 'Directory not found'}
        
        stats = {
            'total_files': 0,
            'image_files': 0,
            'total_size': 0,
            'image_size': 0,
            'formats': {},
            'sizes': [],
        }
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                stats['total_files'] += 1
                stats['total_size'] += file_path.stat().st_size
                
                if self.validator.is_valid_image_file(file_path):
                    stats['image_files'] += 1
                    stats['image_size'] += file_path.stat().st_size
                    
                    # Track formats
                    ext = file_path.suffix.lower()
                    stats['formats'][ext] = stats['formats'].get(ext, 0) + 1
                    
                    # Track image sizes
                    info = self.validator.get_image_info(file_path)
                    if info:
                        stats['sizes'].append(info['size'])
        
        return stats
    
    def cleanup_empty_directories(self, directory: Path) -> int:
        """
        Remove empty directories.
        
        Args:
            directory: Directory to clean up
            
        Returns:
            Number of directories removed
        """
        removed_count = 0
        
        for dir_path in sorted(directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    removed_count += 1
                    logger.debug(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error removing directory {dir_path}: {e}")
        
        logger.info(f"Removed {removed_count} empty directories")
        return removed_count
    
    def backup_files(self, 
                    files: List[Path], 
                    backup_dir: Path,
                    create_timestamp: bool = True) -> bool:
        """
        Create backup of files.
        
        Args:
            files: List of files to backup
            backup_dir: Backup directory
            create_timestamp: Whether to create timestamped subdirectory
            
        Returns:
            True if backup was successful
        """
        try:
            if create_timestamp:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = backup_dir / f"backup_{timestamp}"
            
            self.create_directory(backup_dir)
            
            for file_path in files:
                destination = backup_dir / file_path.name
                self.copy_file(file_path, destination)
            
            logger.info(f"Backup created: {len(files)} files backed up to {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
