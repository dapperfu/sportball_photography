"""
CLI Utilities

Utility functions for the sportball CLI.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import List
from ..core import SportballCore


def get_core(ctx: click.Context) -> SportballCore:
    """
    Get or create SportballCore instance from context.
    
    Args:
        ctx: Click context
        
    Returns:
        SportballCore instance
    """
    if 'core' not in ctx.obj:
        ctx.obj['core'] = SportballCore(
            base_dir=ctx.obj.get('base_dir'),
            enable_gpu=ctx.obj.get('gpu', True),
            max_workers=ctx.obj.get('workers'),
            cache_enabled=ctx.obj.get('cache', True)
        )
    
    return ctx.obj['core']


def find_image_files(input_path: Path, recursive: bool = True) -> List[Path]:
    """
    Find image files in a directory or return single file.
    
    Args:
        input_path: Path to file or directory
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    if input_path.is_file():
        return [input_path]
    
    if not input_path.is_dir():
        raise click.BadParameter(f"Path does not exist: {input_path}")
    
    image_files = []
    
    if recursive:
        # Recursive search
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    else:
        # Non-recursive search
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    return sorted(list(set(image_files)))
