"""
Game Detection Commands

CLI commands for game boundary detection and splitting operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click

from sportball.detectors.game import (
    DEFAULT_MIN_PHOTOS,
    DEFAULT_MIN_PHOTOS_PER_HOUR,
    resolve_session_floors,
)

from ..utils import get_core

# Lazy import: from rich.console import Console
# Lazy import: from rich.table import Table
# Lazy import: from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn



def _session_floors_from_cli(
    ctx: click.Context, min_photos: int, min_rate: int
) -> Tuple[Optional[int], Optional[int]]:
    """
    Apply mutually exclusive --min-photos / --min-rate from Click.

    Parameters
    ----------
    ctx : click.Context
        Current command context, used to see which flags were passed.
    min_photos : int
        Absolute photo floor from the option default or CLI.
    min_rate : int
        Photos-per-hour floor from the option default or CLI.

    Returns
    -------
    tuple of (int or None, int or None)
        ``(min_photos, min_rate)`` for game detection.

    Raises
    ------
    click.UsageError
        If both flags were passed.
    """
    from click.core import ParameterSource

    photos_explicit = (
        ctx.get_parameter_source("min_photos") == ParameterSource.COMMANDLINE
    )
    rate_explicit = ctx.get_parameter_source("min_rate") == ParameterSource.COMMANDLINE
    try:
        return resolve_session_floors(
            min_photos,
            min_rate,
            photos_explicit=photos_explicit,
            rate_explicit=rate_explicit,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _get_console():
    """Lazy import of Console to avoid heavy imports at startup."""
    from rich.console import Console

    return Console()


def _get_progress():
    """Lazy import of Progress components to avoid heavy imports at startup."""
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    return Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


def _get_table():
    """Lazy import of Table to avoid heavy imports at startup."""
    from rich.table import Table

    return Table


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("Games"),
    show_default=True,
    help="Directory for numbered game folders",
)
@click.option(
    "--split-file",
    "-s",
    type=click.Path(path_type=Path),
    help="Text file with manual split points (one timestamp per line)",
)
@click.option(
    "--pattern",
    "-p",
    default="*",
    help='File pattern to match (e.g., "202509*" for Sep 2025)',
)
@click.option(
    "--min-duration",
    "min_duration",
    type=int,
    default=0,
    help="Minimum game duration in minutes (0 keeps short bursts)",
)
@click.option(
    "--min-gap",
    "min_gap",
    type=float,
    default=10.0,
    show_default=True,
    help="Minimum gap to separate games, in minutes (fractions allowed, e.g. 0.5)",
)
@click.option(
    "--min-photos",
    "min_photos",
    type=int,
    default=DEFAULT_MIN_PHOTOS,
    show_default=True,
    help=(
        "Minimum photos in a game (absolute count). "
        "Mutually exclusive with --min-rate. Wrestling: a 7-minute match "
        "with 10 frames still counts."
    ),
)
@click.option(
    "--min-rate",
    "min_rate",
    type=int,
    default=DEFAULT_MIN_PHOTOS_PER_HOUR,
    show_default=True,
    help=(
        "Minimum photos per hour in a session. "
        "Mutually exclusive with --min-photos. "
        "20 shots in 12 min is 100/hour."
    ),
)
@click.option(
    "--copy/--symlink", default=False, help="Copy files instead of creating symlinks"
)
@click.pass_context
def split(
    ctx: click.Context,
    inputs: Tuple[str, ...],
    output_dir: Path,
    split_file: Optional[Path],
    pattern: str,
    min_duration: int,
    min_gap: float,
    min_photos: int,
    min_rate: int,
    copy: bool,
) -> None:
    """
    Split photos from one or more directories into numbered game folders.

    INPUTS are dump directories or globs (``04_Apr 05_May`` or ``*``).
    Capture times come from EXIF (DateTimeOriginal). All inputs are
    pooled so games are numbered across a whole season.
    Output defaults to ``./Games`` unless ``-o`` / ``--output`` is set.

    Examples:

    \b
    sb split 04_Apr 05_May

    \b
    sb split --output SpringGames 04_Apr 05_May

    \b
    sb split --output SpringGames *
    """
    from sportball.detectors.game import expand_input_directories

    try:
        input_dirs = expand_input_directories(inputs, output_dir=output_dir)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    min_photos, min_rate = _session_floors_from_cli(ctx, min_photos, min_rate)

    core = get_core(ctx)
    dir_label = ", ".join(str(path) for path in input_dirs)

    _get_console().print(
        f"✂️  Splitting photos in {dir_label} into games...", style="blue"
    )
    _get_console().print(f"Output: {output_dir}")

    _get_console().print(f"Pattern: {pattern}")

    # Load manual splits if provided
    manual_splits = []
    if split_file and split_file.exists():
        _get_console().print(
            f"📄 Loading manual splits from {split_file}...", style="blue"
        )
        manual_splits = core.game_detector.load_split_file(split_file)
        if manual_splits:
            _get_console().print(
                f"✅ Loaded {len(manual_splits)} manual splits", style="green"
            )
        else:
            _get_console().print("⚠️  No valid splits found in file", style="yellow")
    elif split_file:
        _get_console().print(f"⚠️  Split file not found: {split_file}", style="yellow")

    # Perform game detection
    import time

    start_time = time.time()

    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _get_progress()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=_get_console(),
    ) as progress:
        task = progress.add_task("Processing photos...", total=None)

        # Detect games automatically
        results = core.detect_games(
            input_dirs,
            pattern=pattern,
            min_duration=min_duration,
            min_gap=min_gap,
            min_photos=min_photos,
            min_rate=min_rate,
            output_dir=output_dir,
        )

        progress.update(task, completed=True, description="Game detection complete")

    end_time = time.time()
    processing_time = end_time - start_time

    # Apply manual splits if provided
    if manual_splits and results.get("success", False):
        _get_console().print(
            f"🔧 Applying {len(manual_splits)} manual splits...", style="blue"
        )
        # Get the games from the detector
        games = core.game_detector.games
        if games:
            final_games = core.game_detector.apply_manual_splits(manual_splits)
            # Update results with final games
            results["games"] = core.game_detector._format_games_for_output(final_games)
            _get_console().print(
                f"✅ Applied manual splits. Final games: {len(final_games)}",
                style="green",
            )

    # Display results
    display_game_results(results, output_dir, copy, processing_time)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("Games"),
    show_default=True,
    help="Existing game album to skip when scanning (not written)",
)
@click.option(
    "--pattern",
    "-p",
    default="*",
    help='File pattern to match (e.g., "202509*" for Sep 2025)',
)
@click.option(
    "--min-duration",
    "min_duration",
    type=int,
    default=0,
    help="Minimum game duration in minutes (0 keeps short bursts)",
)
@click.option(
    "--min-gap",
    "min_gap",
    type=float,
    default=10.0,
    show_default=True,
    help="Minimum gap to separate games, in minutes (fractions allowed, e.g. 0.5)",
)
@click.option(
    "--min-photos",
    "min_photos",
    type=int,
    default=DEFAULT_MIN_PHOTOS,
    show_default=True,
    help=(
        "Minimum photos in a game (absolute count). "
        "Mutually exclusive with --min-rate."
    ),
)
@click.option(
    "--min-rate",
    "min_rate",
    type=int,
    default=DEFAULT_MIN_PHOTOS_PER_HOUR,
    show_default=True,
    help=(
        "Minimum photos per hour in a session. " "Mutually exclusive with --min-photos."
    ),
)
@click.option(
    "--bin-minutes",
    "bin_minutes",
    type=int,
    default=None,
    help="Histogram bin width in minutes (chosen from the span if omitted)",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    inputs: Tuple[str, ...],
    output_dir: Path,
    pattern: str,
    min_duration: int,
    min_gap: float,
    min_photos: int,
    min_rate: int,
    bin_minutes: Optional[int],
) -> None:
    """
    Show how photos would be split: albums, breaks, and a density histogram.

    Does not create folders. INPUTS are dump directories or globs.

    Examples:

    \b
    sb analyze 04_Apr 05_May
    sb analyze --output SpringGames *
    """
    from sportball.detectors.game import expand_input_directories

    try:
        input_dirs = expand_input_directories(inputs, output_dir=output_dir)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    min_photos, min_rate = _session_floors_from_cli(ctx, min_photos, min_rate)

    core = get_core(ctx)
    dir_label = ", ".join(str(path) for path in input_dirs)
    _get_console().print(f"📊 Analyzing games in {dir_label}...", style="blue")
    _get_console().print(f"Pattern: {pattern}")

    import time

    start_time = time.time()
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _get_progress()
    kwargs: Dict[str, object] = {
        "pattern": pattern,
        "min_duration": min_duration,
        "min_gap": min_gap,
        "min_photos": min_photos,
        "min_rate": min_rate,
        "output_dir": output_dir,
    }
    if bin_minutes is not None:
        kwargs["bin_minutes"] = bin_minutes

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=_get_console(),
    ) as progress:
        task = progress.add_task("Reading EXIF and clustering...", total=None)
        results = core.analyze_games(input_dirs, **kwargs)
        progress.update(task, completed=True, description="Analysis complete")

    display_game_analysis(results, time.time() - start_time)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "--duration",
    "duration_seconds",
    type=float,
    default=None,
    help="Target video length in seconds per game (FPS = photos / duration). Default 60 when --fps is omitted.",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Frames per second (fractional allowed). Mutually exclusive with --duration.",
)
@click.option(
    "--size",
    "size_spec",
    type=str,
    default=None,
    help=(
        "Output frame size: WIDTHxHEIGHT (5568x3712), WIDTHx (5568x), "
        "or xHEIGHT (x1080 for 1080p). An omitted axis keeps aspect ratio."
    ),
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=2,
    show_default=True,
    help="Parallel ffmpeg encodes",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing .mp4 files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show planned encodes without running ffmpeg",
)
@click.option(
    "--ffmpeg",
    "ffmpeg_bin",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to ffmpeg (default: look up on PATH)",
)
def animate(
    inputs: Tuple[str, ...],
    duration_seconds: Optional[float],
    fps: Optional[float],
    size_spec: Optional[str],
    workers: int,
    force: bool,
    dry_run: bool,
    ffmpeg_bin: Optional[Path],
) -> None:
    """
    Encode each Game## album into a matching .mp4 with ffmpeg.

    INPUTS are game folders (``Game44_12Oct2025_144301-144303``) and/or
    a parent that contains them (``Games``). Each video is written next to
    its album as ``Game44_12Oct2025_144301-144303.mp4``.

    Pass ``--duration`` to fit the whole album into that many seconds
    (FPS is derived). Pass ``--fps`` for an explicit fractional rate.
    Default is ``--duration 60`` (one minute per game).

    ``--size`` resizes each frame. ``5568x3712`` is exact,
    ``5568x`` sets width and keeps aspect ratio, and ``x1080`` sets
    height (1080p).

    Examples:

    \b
    sb animate Games
    sb animate --duration 60 Games
    sb animate --fps 12.5 Games
    sb animate --fps 0.5 Game44_12Oct2025_144301-144303
    sb animate --size 5568x3712 Games
    sb animate --size x1080 Games
    """
    from sportball.detectors.animate import (
        VideoSize,
        animate_games,
        find_ffmpeg,
        parse_video_size,
    )

    if fps is not None and duration_seconds is not None:
        raise click.UsageError("Use either --fps or --duration, not both")
    if fps is None and duration_seconds is None:
        duration_seconds = 60.0
    if fps is not None and fps <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--fps")
    if duration_seconds is not None and duration_seconds <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--duration")

    size: Optional[VideoSize] = None
    if size_spec is not None:
        try:
            size = parse_video_size(size_spec)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--size") from exc

    input_paths = [Path(item) for item in inputs]
    ffmpeg_path: Optional[str] = None
    if ffmpeg_bin is not None:
        ffmpeg_path = str(ffmpeg_bin)
    elif not dry_run:
        try:
            ffmpeg_path = find_ffmpeg()
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc

    mode = f"--fps {fps:g}" if fps is not None else f"--duration {duration_seconds:g}s"
    size_note = f", size={size.label}" if size is not None else ""
    _get_console().print(
        f"🎬 Animating game albums ({mode}{size_note}, workers={workers})...",
        style="blue",
    )

    def _on_progress(name: str, status: str) -> None:
        style = (
            "green"
            if "wrote" in status or "would write" in status
            else ("yellow" if "exists" in status else "red")
        )
        _get_console().print(f"  {name}: {status}", style=style)

    try:
        results = animate_games(
            input_paths,
            fps=fps,
            duration_seconds=duration_seconds,
            size=size,
            workers=workers,
            force=force,
            dry_run=dry_run,
            ffmpeg_bin=ffmpeg_path,
            on_progress=_on_progress,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    ok = sum(1 for item in results if item.success and not item.skipped)
    skipped = sum(1 for item in results if item.skipped)
    failed = sum(1 for item in results if not item.success)
    _get_console().print(
        f"\n✅ {ok} encoded, {skipped} skipped, {failed} failed "
        f"({len(results)} albums)",
        style="green" if failed == 0 else "yellow",
    )
    if failed:
        raise SystemExit(1)


def _format_gap_minutes(minutes: float) -> str:
    """
    Format a gap for the analyze timeline.

    Parameters
    ----------
    minutes : float
        Gap length in minutes.

    Returns
    -------
    str
        Human-readable duration.
    """
    if minutes < 1.0:
        return f"{max(minutes * 60.0, 0.0):.0f} sec"
    if minutes < 120.0:
        return f"{minutes:.1f} min"
    hours = minutes / 60.0
    if hours < 48.0:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} days"


def display_game_analysis(
    results: dict, processing_time: Optional[float] = None
) -> None:
    """
    Print albums, breaks, leftover clusters, and an inter-shot gap histogram.

    Parameters
    ----------
    results : dict
        Payload from ``GameDetector.analyze_games``.
    processing_time : float, optional
        Seconds spent clustering.
    """
    if not results.get("success", False):
        _get_console().print(
            f"❌ Game analysis failed: {results.get('error', 'Unknown error')}",
            style="red",
        )
        return

    summary = results.get("summary", {})
    Table = _get_table()

    from sportball.detectors.game import format_fractional_minutes

    current_gap = float(summary.get("min_gap_minutes", 10))
    min_photos = summary.get("min_photos")
    min_rate = summary.get("min_photos_per_hour")
    if min_photos is None:
        photos_floor = "no --min-photos"
    else:
        photos_floor = f"--min-photos {min_photos}"
    if min_rate is None:
        rate_floor = "no --min-rate"
    else:
        rate_floor = f"--min-rate {min_rate}/h"
    _get_console().print(
        f"\n{summary.get('total_photos', 0)} photos, "
        f"{summary.get('total_games', 0)} games, "
        f"{summary.get('unsorted_photos', 0)} unsorted, "
        f"span {_format_gap_minutes(float(summary.get('span_minutes', 0)))}, "
        f"{photos_floor}, {rate_floor}, "
        f"--min-gap {format_fractional_minutes(current_gap * 60.0)} "
        f"({current_gap * 60.0:g}s)"
    )
    if processing_time is not None:
        _get_console().print(f"Clustered in {processing_time:.1f}s")

    timeline_table = Table(title="How photos will be sorted")
    timeline_table.add_column("", style="dim", width=8)
    timeline_table.add_column("Album", style="cyan")
    timeline_table.add_column("When", style="green")
    timeline_table.add_column("Duration", style="yellow", justify="right")
    timeline_table.add_column("Photos", style="magenta", justify="right")
    timeline_table.add_column("Rate", style="blue", justify="right")
    timeline_table.add_column("Notes", style="white")

    for event in results.get("timeline", []):
        kind = event.get("kind")
        if kind == "break":
            timeline_table.add_row(
                "break",
                "",
                "",
                _format_gap_minutes(float(event.get("gap_minutes", 0))),
                "",
                "",
                f"{event.get('after', '')} → {event.get('before', '')}",
            )
            continue
        notes = event.get("reason", "") if kind == "unsorted" else ""
        timeline_table.add_row(
            "game" if kind == "game" else "skip",
            str(event.get("label", "")),
            f"{event.get('start_label', '')}–{event.get('end_label', '')}",
            f"{event.get('duration_minutes', 0):.1f} min",
            str(event.get("photo_count", 0)),
            f"{event.get('photos_per_hour', 0):.0f}/h",
            notes,
        )

    _get_console().print()
    _get_console().print(timeline_table)

    gap_histogram = results.get("gap_histogram", [])
    if gap_histogram:
        gap_table = Table(
            title=(
                "Seconds between consecutive shots "
                f"(current --min-gap {format_fractional_minutes(current_gap * 60.0)}; "
                "keep = same game, split = new game)"
            )
        )
        gap_table.add_column("Gap", style="green", no_wrap=True)
        gap_table.add_column("--min-gap", style="yellow", justify="right")
        gap_table.add_column("Pairs", style="magenta", justify="right")
        gap_table.add_column("Histogram", style="cyan")
        gap_table.add_column("If set here", style="white")
        for row in gap_histogram:
            gap_table.add_row(
                str(row.get("label", "")),
                str(row.get("min_gap_minutes", "")),
                str(row.get("count", 0)),
                str(row.get("bar", "")),
                str(row.get("effect", "")),
            )
        _get_console().print()
        _get_console().print(gap_table)
        _get_console().print(
            "Set --min-gap to a bin's minutes value to start splitting at that gap.",
            style="dim",
        )

    unsorted = results.get("unsorted", [])
    if unsorted:
        leftover_table = Table(title="Unsorted clusters (will stay in the dump)")
        leftover_table.add_column("When", style="green")
        leftover_table.add_column("Photos", justify="right", style="magenta")
        leftover_table.add_column("Duration", justify="right", style="yellow")
        leftover_table.add_column("Rate", justify="right", style="blue")
        leftover_table.add_column("Why", style="white")
        for cluster in unsorted:
            leftover_table.add_row(
                f"{cluster.get('start_label', '')}–{cluster.get('end_label', '')}",
                str(cluster.get("photo_count", 0)),
                f"{cluster.get('duration_minutes', 0):.1f} min",
                f"{cluster.get('photos_per_hour', 0):.0f}/h",
                str(cluster.get("reason", "")),
            )
        _get_console().print()
        _get_console().print(leftover_table)
    elif not results.get("games"):
        _get_console().print("\nNo games and no leftover clusters.", style="yellow")


def display_game_results(
    results: dict,
    output_dir: Path,
    copy_files: bool,
    processing_time: Optional[float] = None,
) -> None:
    """
    Display game detection results and write album folders.

    Parameters
    ----------
    results : dict
        Payload from ``detect_games``.
    output_dir : Path
        Destination for numbered game folders.
    copy_files : bool
        Copy files when True; symlink otherwise.
    processing_time : float, optional
        Seconds spent clustering.
    """

    if not results.get("success", False):
        _get_console().print(
            f"❌ Game detection failed: {results.get('error', 'Unknown error')}",
            style="red",
        )
        return

    games = results.get("games", [])
    if not games:
        _get_console().print("❌ No games detected", style="red")
        return

    # Create results table
    Table = _get_table()
    table = Table(title="Game Detection Results")
    table.add_column("Game ID", style="cyan", justify="right")
    table.add_column("Start Time", style="green")
    table.add_column("End Time", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Photos", style="magenta", justify="right")
    table.add_column("Gap Before", style="blue")
    table.add_column("Gap After", style="blue")

    total_photos = 0
    total_duration = 0

    from sportball.detectors.game import format_game_id

    total_games = len(games)
    for game in games:
        duration_minutes = game.get("duration_minutes", 0)
        photo_count = game.get("photo_count", 0)
        gap_before = game.get("gap_before_minutes")
        gap_after = game.get("gap_after_minutes")

        total_photos += photo_count
        total_duration += duration_minutes

        table.add_row(
            format_game_id(int(game.get("game_id", 1)), total_games),
            game.get("start_time_formatted", "N/A"),
            game.get("end_time_formatted", "N/A"),
            f"{duration_minutes:.1f} min",
            str(photo_count),
            f"{gap_before:.1f} min" if gap_before else "N/A",
            f"{gap_after:.1f} min" if gap_after else "N/A",
        )

    _get_console().print(table)

    # Format timing information
    if processing_time is not None:
        hours = int(processing_time // 3600)
        minutes = int((processing_time % 3600) // 60)
        seconds = int(processing_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Calculate photos per second
        photos_per_second = total_photos / processing_time if processing_time > 0 else 0

        _get_console().print(
            f"\n📊 Summary: {len(games)} games detected in {time_str}, {photos_per_second:.1f} photos/sec, {total_photos} photos, {total_duration:.1f} minutes total"
        )
    else:
        _get_console().print(
            f"\n📊 Summary: {len(games)} games detected, {total_photos} photos, {total_duration:.1f} minutes total"
        )

    _get_console().print(
        f"\n📁 Creating organized folders in {output_dir}...", style="blue"
    )
    create_organized_folders(games, output_dir, copy_files)


def create_organized_folders(
    games: List[Dict], output_dir: Path, copy_files: bool = False
) -> Dict[str, Path]:
    """
    Create organized folders for games.

    Args:
        games: List of game dictionaries from detection results
        output_dir: Output directory for organized games
        copy_files: Whether to copy files (True) or create symlinks (False)

    Returns:
        Dictionary mapping game IDs to folder paths
    """
    if not games:
        _get_console().print("❌ No games to organize", style="red")
        return {}

    _get_console().print(
        f"📁 Creating organized folders for {len(games)} games...", style="blue"
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    game_folders = {}

    from sportball.detectors.game import format_game_id

    total_games = len(games)
    for game in games:
        # Create game folder name
        game_id = int(game.get("game_id", 1))
        game_label = format_game_id(game_id, total_games)
        start_time = game.get("start_time_formatted", "00:00:00")
        end_time = game.get("end_time_formatted", "00:00:00")

        # Get date from start_time ISO string
        start_time_iso = game.get("start_time", "")
        if start_time_iso:
            try:
                from datetime import datetime

                start_datetime = datetime.fromisoformat(
                    start_time_iso.replace("Z", "+00:00")
                )
                date_str = start_datetime.strftime("%d%b%Y")  # e.g., "24Sep2025"
            except (ValueError, AttributeError):
                # Fallback if parsing fails
                date_str = "UnknownDate"
        else:
            date_str = "UnknownDate"

        game_folder_name = f"Game{game_label}_{date_str}_{start_time.replace(':', '')}-{end_time.replace(':', '')}"
        game_folder = output_dir / game_folder_name
        game_folder.mkdir(exist_ok=True)

        photo_count = game.get("photo_count", 0)
        _get_console().print(
            f"📂 Creating {game_folder_name} with {photo_count} photos", style="green"
        )

        # Copy or symlink photos
        photo_files = game.get("photo_files", [])
        for photo_path_str in photo_files:
            photo_path = Path(photo_path_str)
            if photo_path.exists():
                dest_path = game_folder / photo_path.name

                if copy_files:
                    # Copy file
                    if dest_path.exists():
                        dest_path.unlink()  # Remove existing file
                    shutil.copy2(photo_path, dest_path)
                else:
                    # Create symlink with absolute path
                    if dest_path.exists():
                        dest_path.unlink()  # Remove existing symlink/file
                    try:
                        # Ensure we use absolute path for the symlink target
                        absolute_photo_path = photo_path.resolve()
                        dest_path.symlink_to(absolute_photo_path)
                    except Exception as e:
                        _get_console().print(
                            f"⚠️  Warning: Could not create symlink {dest_path}: {e}",
                            style="yellow",
                        )
                        # Fallback to copying
                        shutil.copy2(photo_path, dest_path)

        game_folders[f"Game{game_label}"] = game_folder

    _get_console().print(
        f"✅ Created {len(game_folders)} organized game folders", style="green"
    )
    return game_folders
