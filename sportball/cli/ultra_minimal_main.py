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
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

# Ultra-minimal CLI group with zero heavy imports
@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--base-dir', '-d', 
              type=click.Path(path_type=Path),
              help='Base directory for operations')
@click.option('--gpu/--no-gpu', 
              default=True,
              help='Enable/disable GPU acceleration')
@click.option('--workers', '-w',
              type=int,
              help='Number of parallel workers')
@click.option('--cache/--no-cache',
              default=True,
              help='Enable/disable result caching')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose logging')
@click.option('--quiet', '-q',
              is_flag=True,
              help='Suppress output except errors')
@click.version_option(version=__import__('sportball').__version__)
@click.pass_context
def cli(ctx: click.Context, 
        base_dir: Optional[Path], 
        gpu: bool, 
        workers: Optional[int], 
        cache: bool,
        verbose: bool, 
        quiet: bool):
    """
    Sportball - Unified Sports Photo Analysis Package
    
    A comprehensive tool for analyzing and organizing sports photographs
    using computer vision, machine learning, and AI techniques.
    
    Features:
    - Face detection and recognition
    - Object detection and extraction (including balls)
    - Game boundary detection
    - Photo quality assessment
    - Sidecar file management and statistics
    - Parallel processing with GPU support
    
    Examples:
    
    \b
    # Detect faces in images
    sportball face detect /path/to/images
    
    \b
    # Extract objects from images
    sportball object extract /path/to/images --output /path/to/output
    
    \b
    # Split photos into games
    sportball games split /path/to/photos --output /path/to/games
    
    \b
    # Detect balls specifically
    sportball object detect /path/to/images --classes "sports ball"
    
    \b
    # Assess photo quality
    sportball quality assess /path/to/images
    
    # Analyze sidecar files
    sportball sidecar stats /path/to/images
    
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
    ctx.obj['base_dir'] = base_dir
    ctx.obj['gpu'] = gpu
    ctx.obj['workers'] = workers
    ctx.obj['cache'] = cache
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet

# Add command groups (import here to avoid circular imports)
from .commands import (
    face_commands,
    object_commands,
    game_commands,
    quality_commands,
    utility_commands,
    sidecar_commands
)

cli.add_command(face_commands.face_group, name='face')
cli.add_command(object_commands.object_group, name='object')
cli.add_command(game_commands.game_group, name='games')
cli.add_command(quality_commands.quality_group, name='quality')
cli.add_command(utility_commands.utility_group, name='util')
cli.add_command(sidecar_commands.sidecar_group, name='sidecar')


@cli.command()
@click.option('--bash', 'shell', flag_value='bash', default=True, help='Generate bash completion script')
@click.option('--zsh', 'shell', flag_value='zsh', help='Generate zsh completion script')
def completion(shell: str):
    """Generate shell completion script."""
    import sys
    import os
    
    # Get the current script name (sportball or python -m sportball.cli.main)
    script_name = os.path.basename(sys.argv[0])
    if script_name in ['python', 'python3'] or (len(sys.argv) > 1 and 'sportball.cli.main' in sys.argv[1]):
        # Running as python -m sportball.cli.main
        script_name = 'sportball'
    elif script_name == 'main.py':
        # Running as python -m sportball.cli.main
        script_name = 'sportball'
    
    if shell == 'bash':
        # Generate bash completion script
        completion_script = f"""# Bash completion for {script_name}
_{script_name}_completion() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    
    # Main commands
    if [[ $COMP_CWORD -eq 1 ]]; then
        opts="face games object quality sidecar util completion --help --version --base-dir --gpu --no-gpu --workers --cache --no-cache --verbose --quiet"
        COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
        return 0
    fi
    
    # Sub-commands based on main command
    case "${{COMP_WORDS[1]}}" in
        face)
            if [[ $COMP_CWORD -eq 2 ]]; then
                opts="detect recognize cluster benchmark"
                COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
            fi
            ;;
        games)
            if [[ $COMP_CWORD -eq 2 ]]; then
                opts="split detect analyze"
                COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
            fi
            ;;
        object)
            if [[ $COMP_CWORD -eq 2 ]]; then
                opts="detect extract analyze"
                COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
            fi
            ;;
        quality)
            if [[ $COMP_CWORD -eq 2 ]]; then
                opts="assess analyze filter"
                COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
            fi
            ;;
        sidecar)
            if [[ $COMP_CWORD -eq 2 ]]; then
                opts="stats analyze validate"
                COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
            fi
            ;;
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
    elif shell == 'zsh':
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
                'face:Face detection and recognition commands'
                'games:Game detection and splitting commands'
                'object:Object detection and extraction commands'
                'quality:Photo quality assessment commands'
                'sidecar:Sidecar file management and statistics commands'
                'util:Utility commands for cache management and system operations'
                'completion:Generate shell completion script'
            )
            _describe 'command' commands
            ;;
        args)
            case $line[1] in
                face)
                    _arguments '1: :(detect recognize cluster benchmark)'
                    ;;
                games)
                    _arguments '1: :(split detect analyze)'
                    ;;
                object)
                    _arguments '1: :(detect extract analyze)'
                    ;;
                quality)
                    _arguments '1: :(assess analyze filter)'
                    ;;
                sidecar)
                    _arguments '1: :(stats analyze validate)'
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

if __name__ == '__main__':
    main()
