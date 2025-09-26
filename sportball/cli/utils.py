"""
CLI Utilities

Utility functions for the sportball CLI.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
from pathlib import Path
from typing import List
# Lazy import SportballCore to avoid heavy imports at startup
# from ..core import SportballCore


def get_core(ctx: click.Context):
    """
    Get or create SportballCore instance from context.
    
    Args:
        ctx: Click context
        
    Returns:
        SportballCore instance
    """
    if 'core' not in ctx.obj:
        # Lazy import to avoid heavy imports at startup
        from ..core import SportballCore
        ctx.obj['core'] = SportballCore(
            base_dir=ctx.obj.get('base_dir'),
            enable_gpu=ctx.obj.get('gpu', True),
            max_workers=ctx.obj.get('workers'),
            cache_enabled=ctx.obj.get('cache', True),
            verbose=ctx.obj.get('verbose', False)
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
    
    # Directories to exclude from search
    exclude_dirs = {'venv', '__pycache__', '.git', '.sportball_cache', 'node_modules'}
    
    if recursive:
        # Recursive search with exclusions
        for ext in image_extensions:
            for pattern in [f'*{ext}', f'*{ext.upper()}']:
                for file_path in input_path.rglob(pattern):
                    # Skip files in excluded directories
                    if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                        continue
                    image_files.append(file_path)
    else:
        # Non-recursive search
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    return sorted(list(set(image_files)))
