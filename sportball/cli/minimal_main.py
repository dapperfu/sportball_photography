"""
Ultra-minimal CLI entry point to avoid any heavy imports at startup.

This module provides the absolute minimum imports needed to start the CLI,
with all heavy dependencies loaded only when specific commands are used.
"""

import click
import warnings
from pathlib import Path
from typing import Optional

# Suppress annoying deprecation warnings
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)


# Minimal CLI group with no heavy imports
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--base-dir",
    "-d",
    type=click.Path(path_type=Path),
    help="Base directory for operations",
)
@click.option("--gpu/--no-gpu", default=True, help="Enable/disable GPU acceleration")
@click.option("--workers", "-w", type=int, help="Number of parallel workers")
@click.option("--cache/--no-cache", default=True, help="Enable/disable result caching")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output except errors")
@click.version_option(version=__import__("sportball").__version__)
@click.pass_context
def cli(
    ctx: click.Context,
    base_dir: Optional[Path],
    gpu: bool,
    workers: Optional[int],
    cache: bool,
    verbose: bool,
    quiet: bool,
):
    """
    Sportball - Unified Sports Photo Analysis Package

    Split a dump of sports photos into game folders using EXIF times.

    Neural-net face/object/pose detection lives in other projects.

    Examples:

    \b
    # Split photos into games
    sportball split 04_Apr 05_May

    Bash Completion:
    To enable bash completion, add this to your ~/.bashrc or ~/.bash_profile:

        # For virtual environment (recommended):
        eval "$(python -m sportball.cli.main --completion-script-bash)"

        # Or if sportball is in your PATH:
        eval "$(sportball --completion-script-bash)"

    Then restart your shell or run: source ~/.bashrc
    """

    # Configure logging only when needed (lazy import)
    if verbose or quiet:
        from loguru import logger

        if verbose:
            logger.add("sportball.log", level="DEBUG", rotation="10 MB")
            logger.info("Verbose logging enabled")
        elif quiet:
            logger.remove()
            logger.add(lambda msg: None, level="ERROR")
        else:
            # Default: INFO level, suppress DEBUG messages
            logger.remove()
            logger.add(lambda msg: None, level="INFO")

    # Store configuration in context
    ctx.ensure_object(dict)
    ctx.obj["base_dir"] = base_dir
    ctx.obj["gpu"] = gpu
    ctx.obj["workers"] = workers
    ctx.obj["cache"] = cache
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


# Ultra-minimal command loading - only load when accessed
def _get_command(self, ctx, name):
    """Load commands only when accessed to avoid heavy imports."""
    if name in ("analyze", "split", "animate"):
        from .commands import game_commands

        return getattr(game_commands, name)
    elif name == "util":
        from .commands import utility_commands

        return utility_commands.utility_group

    return None


# Override command resolution for lazy loading
cli.get_command = _get_command.__get__(cli, type(cli))


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
