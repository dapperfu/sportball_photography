"""
CLI Utilities

Utility functions for the sportball CLI.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
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
