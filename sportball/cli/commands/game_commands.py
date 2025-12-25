"""
Game Detection Commands

CLI commands for game boundary detection and splitting operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import click
import shutil
from pathlib import Path
from typing import Optional, List, Dict
# Lazy import: from rich.console import Console
# Lazy import: from rich.table import Table
# Lazy import: from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..utils import get_core

console = None  # Will be initialized lazily


def _get_console():
    """Lazy import of Console to avoid heavy imports at startup."""
    from rich.console import Console

    return Console()


def _get_progress():
    """Lazy import of Progress components to avoid heavy imports at startup."""
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
    )

    return Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


def _get_table():
    """Lazy import of Table to avoid heavy imports at startup."""
    from rich.table import Table

    return Table


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def game_group():
    """Game detection and splitting commands."""
    pass


@game_group.command()
@click.argument(
    "input_path", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--split-file",
    "-s",
    type=click.Path(path_type=Path),
    help="Text file with manual split points (one timestamp per line)",
)
@click.option(
    "--pattern",
    "-p",
    default="*_*",
    help='File pattern to match (e.g., "202509*_*" for Sep 2025)',
)
@click.option(
    "--min-duration",
    "min_duration",
    type=int,
    default=30,
    help="Minimum game duration in minutes",
)
@click.option(
    "--min-gap",
    "min_gap",
    type=int,
    default=10,
    help="Minimum gap to separate games in minutes",
)
@click.option(
    "--min-photos", "min_photos", type=int, default=50, help="Minimum photos per game"
)
@click.option(
    "--copy/--symlink", default=False, help="Copy files instead of creating symlinks"
)
@click.option(
    "--save-sidecar/--no-sidecar", default=True, help="Save results to sidecar files"
)
@click.option(
    "--analyze-only",
    "analyze_only",
    is_flag=True,
    help="Only analyze and display results without creating folders",
)
@click.option(
    "--split-by-jersey",
    "split_by_jersey",
    is_flag=True,
    help="Split games based on jersey colors (requires pose detection)",
)
@click.option(
    "--pose-confidence",
    "pose_confidence",
    type=float,
    default=0.7,
    help="Minimum confidence for pose detections (jersey splitting)",
)
@click.option(
    "--color-similarity",
    "color_similarity",
    type=float,
    default=0.15,
    help="Threshold for grouping similar jersey colors (jersey splitting)",
)
@click.option(
    "--min-team-photos",
    "min_team_photos",
    type=int,
    default=5,
    help="Minimum photos required to form a team (jersey splitting)",
)
@click.pass_context
def split(
    ctx: click.Context,
    input_path: Path,
    output_dir: Path,
    split_file: Optional[Path],
    pattern: str,
    min_duration: int,
    min_gap: int,
    min_photos: int,
    copy: bool,
    save_sidecar: bool,
    analyze_only: bool,
    split_by_jersey: bool,
    pose_confidence: float,
    color_similarity: float,
    min_team_photos: int,
):
    """
    Split photos into games with optional manual split points.

    This is the main game organization command that detects game boundaries
    based on photo timestamps and organizes them into folders.

    INPUT_PATH should be a directory containing photos with timestamp filenames.
    OUTPUT_DIR is where organized games will be created (unless --analyze-only is used).

    Examples:

    \b
    # Basic game splitting
    sb games split /path/to/photos /path/to/games

    \b
    # With manual splits
    sb games split /path/to/photos /path/to/games --split-file splits.txt

    \b
    # Only analyze without creating folders
    sb games split /path/to/photos /path/to/games --analyze-only

    \b
    # Copy files instead of symlinks
    sb games split /path/to/photos /path/to/games --copy

    \b
    # Split by jersey colors (requires pose detection)
    sb games split /path/to/photos /path/to/games --split-by-jersey

    \b
    # Jersey splitting with custom parameters
    sb games split /path/to/photos /path/to/games --split-by-jersey --pose-confidence 0.8 --color-similarity 0.1
    """

    core = get_core(ctx)

    # Handle jersey-based splitting
    if split_by_jersey:
        _handle_jersey_splitting(
            core, input_path, output_dir, analyze_only, copy, save_sidecar,
            pose_confidence, color_similarity, min_team_photos
        )
        return

    if analyze_only:
        _get_console().print(f"📊 Analyzing games in {input_path}...", style="blue")
    else:
        _get_console().print(
            f"✂️  Splitting photos in {input_path} into games...", style="blue"
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
            input_path,
            pattern=pattern,
            save_sidecar=save_sidecar,
            min_duration=min_duration,
            min_gap=min_gap,
            min_photos=min_photos,
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
    display_game_results(results, output_dir, copy, analyze_only, processing_time)


def display_game_results(
    results: dict, output_dir: Path, copy_files: bool, analyze_only: bool = False, processing_time: Optional[float] = None
):
    """Display game detection results."""

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

    for game in games:
        duration_minutes = game.get("duration_minutes", 0)
        photo_count = game.get("photo_count", 0)
        gap_before = game.get("gap_before_minutes")
        gap_after = game.get("gap_after_minutes")

        total_photos += photo_count
        total_duration += duration_minutes

        table.add_row(
            str(game.get("game_id", "N/A")),
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

    # Create organized folders (unless analyze-only mode)
    if not analyze_only:
        _get_console().print(
            f"\n📁 Creating organized folders in {output_dir}...", style="blue"
        )
        create_organized_folders(games, output_dir, copy_files)
    else:
        _get_console().print(
            "\n📊 Analysis complete - no folders created (use without --analyze-only to create folders)",
            style="blue",
        )


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

    for game in games:
        # Create game folder name
        game_id = game.get("game_id", 1)
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

        game_folder_name = f"Game{game_id}_{date_str}_{start_time.replace(':', '')}-{end_time.replace(':', '')}"
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

        game_folders[f"Game{game_id}"] = game_folder

    _get_console().print(
        f"✅ Created {len(game_folders)} organized game folders", style="green"
    )
    return game_folders


def _handle_jersey_splitting(
    core,
    input_path: Path,
    output_dir: Path,
    analyze_only: bool,
    copy_files: bool,
    save_sidecar: bool,
    pose_confidence: float,
    color_similarity: float,
    min_team_photos: int,
):
    """Handle jersey-based game splitting."""
    console = _get_console()
    
    if analyze_only:
        console.print(f"📊 Analyzing jersey colors in {input_path}...", style="blue")
    else:
        console.print(
            f"🎨 Splitting games by jersey colors in {input_path}...", style="blue"
        )
        console.print(f"Output: {output_dir}")
    
    console.print(f"Pose confidence threshold: {pose_confidence}")
    console.print(f"Color similarity threshold: {color_similarity}")
    console.print(f"Minimum team photos: {min_team_photos}")
    
    # Find image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    if not image_files:
        console.print("❌ No image files found", style="red")
        return
    
    console.print(f"Found {len(image_files)} images to process")
    
    # Check for existing pose detection data
    missing_pose_data = []
    for image_file in image_files:
        sidecar_data = core.sidecar.load_data(image_file, "pose_detection")
        if not sidecar_data:
            missing_pose_data.append(image_file)
    
    if missing_pose_data:
        console.print(
            f"⚠️  Warning: {len(missing_pose_data)} images missing pose detection data",
            style="yellow"
        )
        console.print(
            "Consider running pose detection first: sb pose detect /path/to/images",
            style="yellow"
        )
    
    # Perform jersey splitting
    import time
    start_time = time.time()
    
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _get_progress()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing jersey colors...", total=None)
        
        # Perform jersey-based game splitting
        results = core.split_games_by_jersey_color(
            image_files,
            output_dir=output_dir if not analyze_only else None,
            save_sidecar=save_sidecar,
            pose_confidence_threshold=pose_confidence,
            color_similarity_threshold=color_similarity,
            min_team_photos=min_team_photos,
        )
        
        progress.update(task, completed=True, description="Jersey splitting complete")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Display results
    _display_jersey_splitting_results(results, output_dir, copy_files, analyze_only, processing_time)


def _display_jersey_splitting_results(
    results: dict, 
    output_dir: Path, 
    copy_files: bool, 
    analyze_only: bool = False, 
    processing_time: Optional[float] = None
):
    """Display jersey splitting results."""
    console = _get_console()
    
    if not results.get("success", False):
        console.print(
            f"❌ Jersey splitting failed: {results.get('error', 'Unknown error')}",
            style="red",
        )
        return
    
    summary = results.get("summary", {})
    detected_teams = results.get("detected_teams", [])
    split_decisions = results.get("split_decisions", [])
    
    if not detected_teams:
        console.print("❌ No teams detected", style="red")
        return
    
    # Display detected teams
    Table = _get_table()
    teams_table = Table(title="Detected Teams")
    teams_table.add_column("Team Name", style="cyan")
    teams_table.add_column("Dominant Color", style="green")
    teams_table.add_column("Photo Count", style="magenta", justify="right")
    teams_table.add_column("Confidence", style="yellow", justify="right")
    
    for team in detected_teams:
        color_rgb = team.get("dominant_color", {}).get("rgb_color", (0, 0, 0))
        color_str = f"RGB{color_rgb}"
        
        teams_table.add_row(
            team.get("team_name", "Unknown"),
            color_str,
            str(team.get("photo_count", 0)),
            f"{team.get('confidence', 0.0):.2f}",
        )
    
    console.print(teams_table)
    
    # Display splitting statistics
    total_photos = summary.get("total_photos", 0)
    split_photos = summary.get("split_photos", 0)
    single_team_photos = summary.get("single_team_photos", 0)
    multi_team_photos = summary.get("multi_team_photos", 0)
    no_team_photos = summary.get("no_team_photos", 0)
    avg_confidence = summary.get("average_confidence", 0.0)
    
    # Format timing information
    if processing_time is not None:
        hours = int(processing_time // 3600)
        minutes = int((processing_time % 3600) // 60)
        seconds = int(processing_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Calculate photos per second
        photos_per_second = total_photos / processing_time if processing_time > 0 else 0
        
        console.print(
            f"\n📊 Summary: {len(detected_teams)} teams detected in {time_str}, "
            f"{photos_per_second:.1f} photos/sec, {total_photos} photos total"
        )
    else:
        console.print(
            f"\n📊 Summary: {len(detected_teams)} teams detected, {total_photos} photos total"
        )
    
    console.print(f"Split photos: {split_photos}")
    console.print(f"Single team photos: {single_team_photos}")
    console.print(f"Multi-team photos: {multi_team_photos}")
    console.print(f"No team photos: {no_team_photos}")
    console.print(f"Average confidence: {avg_confidence:.2f}")
    
    # Create organized folders (unless analyze-only mode)
    if not analyze_only and output_dir:
        console.print(
            f"\n📁 Creating organized folders in {output_dir}...", style="blue"
        )
        _create_jersey_organized_folders(split_decisions, detected_teams, output_dir, copy_files)
    else:
        console.print(
            "\n📊 Analysis complete - no folders created (use without --analyze-only to create folders)",
            style="blue",
        )


def _create_jersey_organized_folders(
    split_decisions: List[Dict], 
    detected_teams: List[Dict], 
    output_dir: Path, 
    copy_files: bool = False
) -> Dict[str, Path]:
    """
    Create organized folders for jersey-based game splitting.
    
    Args:
        split_decisions: List of splitting decisions
        detected_teams: List of detected teams
        output_dir: Output directory for organized games
        copy_files: Whether to copy files (True) or create symlinks (False)
        
    Returns:
        Dictionary mapping team names to folder paths
    """
    console = _get_console()
    
    if not detected_teams:
        console.print("❌ No teams to organize", style="red")
        return {}
    
    console.print(
        f"📁 Creating organized folders for {len(detected_teams)} teams...", style="blue"
    )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    team_folders = {}
    
    # Create folders for each team
    for team in detected_teams:
        team_name = team.get("team_name", "Unknown")
        team_folder = output_dir / team_name
        team_folder.mkdir(exist_ok=True)
        team_folders[team_name] = team_folder
        
        console.print(f"📂 Created folder: {team_name}", style="green")
    
    # Create folders for special categories
    multi_team_folder = output_dir / "MultiTeam"
    multi_team_folder.mkdir(exist_ok=True)
    
    single_team_folder = output_dir / "SingleTeam"
    single_team_folder.mkdir(exist_ok=True)
    
    no_team_folder = output_dir / "NoTeam"
    no_team_folder.mkdir(exist_ok=True)
    
    # Organize photos
    for decision in split_decisions:
        photo_path = Path(decision.get("photo_path", ""))
        if not photo_path.exists():
            continue
        
        should_split = decision.get("should_split", False)
        split_games = decision.get("split_games", [])
        
        if should_split and len(split_games) > 1:
            # Multi-team photo
            dest_path = multi_team_folder / photo_path.name
            _copy_or_symlink_photo(photo_path, dest_path, copy_files)
        elif len(split_games) == 1:
            # Single team photo
            team_name = split_games[0]
            if team_name in team_folders:
                dest_path = team_folders[team_name] / photo_path.name
                _copy_or_symlink_photo(photo_path, dest_path, copy_files)
        else:
            # No clear team
            dest_path = no_team_folder / photo_path.name
            _copy_or_symlink_photo(photo_path, dest_path, copy_files)
    
    console.print(
        "✅ Created organized folders for jersey-based splitting", style="green"
    )
    return team_folders


def _copy_or_symlink_photo(src_path: Path, dest_path: Path, copy_files: bool):
    """Copy or symlink photo from source to destination."""
    try:
        import shutil
        
        if copy_files:
            # Copy file
            if dest_path.exists():
                dest_path.unlink()
            shutil.copy2(src_path, dest_path)
        else:
            # Create symlink
            if dest_path.exists():
                dest_path.unlink()
            dest_path.symlink_to(src_path.resolve())
    except Exception as e:
        console = _get_console()
        console.print(
            f"⚠️  Warning: Could not copy/symlink {src_path}: {e}",
            style="yellow",
        )
