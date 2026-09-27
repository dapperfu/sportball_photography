"""
Ultra-minimal CLI entry point with zero heavy imports at startup.

This module provides the absolute minimum needed to start the CLI,
with ALL heavy dependencies loaded only when specific commands are used.
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


class LazyCommandGroup(click.Group):
    """Custom Click group that loads commands lazily."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands_loaded = False

    def get_command(self, ctx, cmd_name):
        """Load commands lazily when a command is requested."""
        if not self.commands_loaded:
            _load_commands()
            self.commands_loaded = True
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx):
        """Load commands lazily when listing commands."""
        if not self.commands_loaded:
            _load_commands()
            self.commands_loaded = True
        return super().list_commands(ctx)


# Ultra-minimal CLI group with zero heavy imports
@click.group(
    cls=LazyCommandGroup, context_settings={"help_option_names": ["-h", "--help"]}
)
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
    sportball split --output SpringGames *

    Bash Completion:
    To enable bash completion, add this to your ~/.bashrc or ~/.bash_profile:

        # For virtual environment (recommended):
        eval "$(python -m sportball.cli.main completion --bash)"

        # Or if sportball is in your PATH:
        eval "$(sportball completion --bash)"

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


# Lazy command loading to avoid heavy imports at startup
def _load_commands():
    """Load command groups lazily to avoid heavy imports at startup."""
    from .commands import (
        game_commands,
        pano_commands,
        utility_commands,
    )

    cli.add_command(game_commands.split)
    cli.add_command(game_commands.analyze)
    cli.add_command(game_commands.animate)
    cli.add_command(pano_commands.pano)
    cli.add_command(utility_commands.utility_group, name="util")


@cli.command()
@click.option(
    "--bash",
    "shell",
    flag_value="bash",
    default=True,
    help="Generate bash completion script",
)
@click.option("--zsh", "shell", flag_value="zsh", help="Generate zsh completion script")
def completion(shell: str):
    """Generate shell completion script."""
    import sys
    import os

    # Get the current script name (sportball or python -m sportball.cli.main)
    script_name = os.path.basename(sys.argv[0])
    if script_name in ["python", "python3"] or (
        len(sys.argv) > 1 and "sportball.cli.main" in sys.argv[1]
    ):
        # Running as python -m sportball.cli.main
        script_name = "sportball"
    elif script_name == "main.py":
        # Running as python -m sportball.cli.main
        script_name = "sportball"

    if shell == "bash":
        # Generate bash completion script
        completion_script = f"""# Bash completion for {script_name}
_{script_name}_completion() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    
    # Main commands
    if [[ $COMP_CWORD -eq 1 ]]; then
        opts="analyze split animate pano util completion --help --version --base-dir --gpu --no-gpu --workers --cache --no-cache --verbose --quiet"
        COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
        return 0
    fi
    
    # Sub-commands based on main command
    case "${{COMP_WORDS[1]}}" in
        util)
            if [[ $COMP_CWORD -eq 2 ]]; then
                opts="cache-clear cache-stats system-info"
                COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
            fi
            ;;
        completion)
            if [[ $COMP_CWORD -eq 2 ]]; then
                opts="--bash --zsh"
                COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
            fi
            ;;
    esac
    
    # File completion for paths
    if [[ $cur == */* ]]; then
        COMPREPLY=( $(compgen -f -- "${{cur}}") )
    fi
}}

complete -F _{script_name}_completion {script_name}
"""
        print(completion_script)
    elif shell == "zsh":
        # Generate zsh completion script
        completion_script = f"""# Zsh completion for {script_name}
#compdef {script_name}

_{script_name}() {{
    local context state line
    typeset -A opt_args
    
    _arguments -C \\
        '1: :->command' \\
        '*::arg:->args' \\
        '--help[Show help message]' \\
        '--version[Show version]' \\
        '--base-dir[Base directory for operations]:directory:_files' \\
        '--gpu[Enable GPU acceleration]' \\
        '--no-gpu[Disable GPU acceleration]' \\
        '--workers[Number of parallel workers]:number' \\
        '--cache[Enable result caching]' \\
        '--no-cache[Disable result caching]' \\
        '--verbose[Enable verbose logging]' \\
        '--quiet[Suppress output except errors]'
    
    case $state in
        command)
            local commands
            commands=(
                'analyze:Show how photos would be split into games'
                'split:Split photos into game folders'
                'animate:Encode each game folder to an MP4'
                'pano:Find action panoramas and write Hugin projects'
                'util:Utility commands for cache management and system operations'
                'completion:Generate shell completion script'
            )
            _describe 'command' commands
            ;;
        args)
            case $line[1] in
                games)
                    _arguments '1: :(split detect analyze)'
                    ;;
                util)
                    _arguments '1: :(cache-clear cache-stats system-info)'
                    ;;
                completion)
                    _arguments '1: :(--bash --zsh)'
                    ;;
            esac
            ;;
    esac
}}

_{script_name} "$@"
"""
        print(completion_script)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
