"""
Action-panorama commands.

``sb pano`` finds known panoramas (ended by uniform frames) and guessed
panoramas (neighbor overlap, no overlap a few frames away), then writes
symlink folders and a Hugin project optimized for yaw, pitch, roll, and
field of view. Stitching and a black-canvas crop are on by default.
``--no-stitch`` skips the inline stitch. ``--no-crop`` skips
``<pano_name>_cropped.jpg``.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from datetime import datetime
from typing import Any, Tuple

import click

from sportball.detectors.pano import (
    DEFAULT_MARKER_COUNT,
    DEFAULT_MARKER_SAMPLE,
    DEFAULT_MARKER_VAR,
    DEFAULT_CP_EDGE,
    DEFAULT_DISCONTINUITY_SECONDS,
    DEFAULT_FAR_POINTS,
    DEFAULT_MIN_FRAMES,
    DEFAULT_OVERLAP_POINTS,
    DEFAULT_SPLIT_POINTS,
    DEFAULT_STRIDE,
    LinkReport,
    PanoConfig,
    PanoGroup,
    PanoResult,
    anchor_index,
)


def _get_console() -> Any:
    """
    Lazy import of Console to avoid heavy imports at startup.

    Returns
    -------
    Console
        A Rich console.
    """
    from rich.console import Console

    return Console()


def _get_table() -> Any:
    """
    Lazy import of Table to avoid heavy imports at startup.

    Returns
    -------
    type
        The Rich Table class.
    """
    from rich.table import Table

    return Table


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "--pattern",
    "-p",
    default="*",
    show_default=True,
    help="File pattern to match",
)
@click.option(
    "--marker-var",
    type=float,
    default=DEFAULT_MARKER_VAR,
    show_default=True,
    help=(
        "Max variance of the grayscale thumbnail around its mean. "
        "At or below this, a frame is a marker (lens covered, sky, or ground)."
    ),
)
@click.option(
    "--marker-sample",
    type=int,
    default=DEFAULT_MARKER_SAMPLE,
    show_default=True,
    help="Edge of the grayscale thumbnail used for the mean and the variance",
)
@click.option(
    "--marker-count",
    type=int,
    default=DEFAULT_MARKER_COUNT,
    show_default=True,
    help="Consecutive uniform frames that mark a known panorama",
)
@click.option(
    "--discontinuity",
    type=float,
    default=DEFAULT_DISCONTINUITY_SECONDS,
    show_default=True,
    help="Seconds. A larger gap is two different shots",
)
@click.option(
    "--split-points",
    type=int,
    default=DEFAULT_SPLIT_POINTS,
    show_default=True,
    help=(
        "Walking back from a uniform marker, a pair with this many "
        "control points or fewer is a cut"
    ),
)
@click.option(
    "--boundary",
    type=click.Choice(["largest-gap", "first"], case_sensitive=False),
    default="largest-gap",
    show_default=True,
    help=(
        "largest-gap cuts the known panorama at the weak link with the "
        "biggest time gap. first cuts at the weak link nearest the marker frames."
    ),
)
@click.option(
    "--guess/--no-guess",
    default=True,
    show_default=True,
    help="Also search for panoramas that were not marked with uniform frames",
)
@click.option(
    "--stride",
    type=int,
    default=DEFAULT_STRIDE,
    show_default=True,
    help="Frame distance that must not overlap in a guessed panorama",
)
@click.option(
    "--overlap-points",
    type=int,
    default=DEFAULT_OVERLAP_POINTS,
    show_default=True,
    help="Minimum control points between neighbors to count as overlap",
)
@click.option(
    "--far-points",
    type=int,
    default=DEFAULT_FAR_POINTS,
    show_default=True,
    help="Maximum control points allowed --stride frames apart",
)
@click.option(
    "--min-frames",
    type=int,
    default=DEFAULT_MIN_FRAMES,
    show_default=True,
    help=(
        "Minimum frames in a panorama. Guessed runs also need stride+1 "
        "frames so the stride test has a pair."
    ),
)
@click.option(
    "--cp-edge",
    type=int,
    default=DEFAULT_CP_EDGE,
    show_default=True,
    help="Long-edge size of the JPEG previews passed to cpfind",
)
@click.option(
    "--copy/--symlink",
    "copy_files",
    default=False,
    help="Copy files instead of creating symlinks",
)
@click.option(
    "--pto/--no-pto",
    "write_pto",
    default=True,
    show_default=True,
    help="Write a Hugin .pto project inside each panorama folder",
)
@click.option(
    "--stitch/--no-stitch",
    default=True,
    show_default=True,
    help=(
        "After every project is optimized, stitch it to a JPEG with "
        "hugin_executor. No batch window is opened."
    ),
)
@click.option(
    "--crop/--no-crop",
    default=True,
    show_default=True,
    help=(
        "Write <pano_name>_cropped.jpg beside each stitched panorama, "
        "with the black canvas removed."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print grayscale mean, variance, and control points without writing folders",
)
@click.pass_context
def pano(
    ctx: click.Context,
    inputs: Tuple[str, ...],
    pattern: str,
    marker_var: float,
    marker_sample: int,
    marker_count: int,
    discontinuity: float,
    split_points: int,
    boundary: str,
    guess: bool,
    stride: int,
    overlap_points: int,
    far_points: int,
    min_frames: int,
    cp_edge: int,
    copy_files: bool,
    write_pto: bool,
    stitch: bool,
    crop: bool,
    dry_run: bool,
) -> None:
    """
    Find action panoramas and write Hugin projects.

    INPUTS are dump directories, game folders, or globs. A known panorama
    is the run of frames before two uniform frames (lens covered, sky, or
    a zoom into the ground). A guessed panorama has no marker: neighbors
    share control points, and frames ``--stride`` apart (default 3) do not.

    Five frames in 2.5 seconds and five frames in 10 seconds are the same
    test. A gap longer than ``--discontinuity`` seconds (default 10) is
    two different shots.

    Marker frames stay out of the folder. Each accepted run becomes
    ``known_pano01_20Sep2025_090012-090018`` or ``guessed_pano02_...``
    inside a sibling of that input named ``{input}-panos``. Five input
    folders produce five ``-panos`` folders, each numbered from 01.
    A panorama needs at least 5 photos. The ``.pto`` inside references
    those files. The median photo is the position and exposure anchor
    (5 photos: the 3rd; 4 photos: the 2nd). The project is optimized
    for yaw, pitch, roll, and field of view, and nothing else.
    Stitching is on unless ``--no-stitch`` is passed. Each project is
    stitched to a JPEG in process with ``hugin_executor``, named like
    the ``.pto``, with no batch window. A project Hugin cannot stitch
    is skipped and listed at the end, and the others still run.
    Cropping is on unless ``--no-crop`` is passed. After those images
    exist it writes ``<pano_name>_cropped.jpg`` beside the full stitch,
    in the same panorama folder, with the black canvas removed. The
    full image stays, so the black border can still be cleaned up by
    hand.

    Examples:

    \b
    sb pano Games/Game03_19Sep2026_120915-132501
    sb pano Game01 Game02 Game03 Game04 Game05
    sb pano --no-stitch --no-crop Games/Game03_19Sep2026_120915-132501
    sb pano --dry-run Games/Game03_19Sep2026_120915-132501
    sb pano --marker-var 80 --split-points 0 --overlap-points 15 Games
    """
    from sportball.detectors.pano import find_action_panos

    quiet = bool(ctx.obj and ctx.obj.get("quiet"))
    config = PanoConfig(
        marker_var=marker_var,
        marker_sample=marker_sample,
        marker_count=marker_count,
        discontinuity=discontinuity,
        split_points=split_points,
        boundary=boundary.lower(),
        guess=guess,
        stride=stride,
        overlap_points=overlap_points,
        far_points=far_points,
        min_frames=min_frames,
        cp_edge=cp_edge,
        copy_files=copy_files,
        write_pto=write_pto,
        stitch=stitch,
        crop=crop,
        dry_run=dry_run,
        progress=not quiet,
    )
    try:
        result = find_action_panos(
            inputs,
            config,
            pattern=pattern,
        )
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    if not quiet:
        display_pano_result(result, config)
    else:
        accepted = [group for group in result.groups if group.accepted]
        summary = f"{len(accepted)} panoramas, {len(result.frames)} photos"
        if result.unstitched:
            summary += f", {len(result.unstitched)} not stitched"
        _get_console().print(summary)


def display_pano_result(result: PanoResult, config: PanoConfig) -> None:
    """
    Print grayscale mean, variance, marker runs, and per-link control points.

    Parameters
    ----------
    result : PanoResult
        Detection output.
    config : PanoConfig
        Thresholds, used to label uniform frames and to note a dry run.
    """
    console = _get_console()
    Table = _get_table()
    accepted = [group for group in result.groups if group.accepted]
    known = sum(1 for group in accepted if group.kind == "known")
    guessed = sum(1 for group in accepted if group.kind == "guessed")
    held = sum(1 for group in result.groups if group.kind == "held")
    mode = "dry run" if config.dry_run else "wrote folders"
    console.print(
        f"\n{len(result.frames)} photos, "
        f"{result.skipped_no_exif} skipped (no EXIF), "
        f"{result.skipped_undecodable} skipped (undecodable), "
        f"{known} known, {guessed} guessed, {held} held ({mode})"
    )

    photo_table = Table(title="Uniformity (grayscale mean and variance)")
    photo_table.add_column("File", style="cyan")
    photo_table.add_column("When", style="green")
    photo_table.add_column("Mean", justify="right", style="magenta")
    photo_table.add_column("Var", justify="right", style="magenta")
    photo_table.add_column("Marker", style="yellow")
    for frame in result.frames:
        photo_table.add_row(
            frame.path.name,
            _format_timestamp(frame.timestamp),
            f"{frame.mean:.1f}",
            f"{frame.variation:.1f}",
            "yes" if frame.variation <= config.marker_var else "",
        )
    console.print()
    console.print(photo_table)

    if result.marker_runs:
        console.print("\nUniform markers (not included in a folder):")
        for start, end in result.marker_runs:
            names = ", ".join(
                result.frames[index].path.name for index in range(start, end + 1)
            )
            console.print(f"  {names}")
    else:
        console.print("\nNo uniform-frame markers.")

    if result.cropped:
        console.print("\nCropped panoramas (black canvas removed):")
        for path in result.cropped:
            console.print(f"  {path}")

    if result.unstitched:
        console.print("\nNot stitched (Hugin failed; the folder and .pto are kept):")
        for path in result.unstitched:
            console.print(f"  {path}", style="yellow")

    if not result.groups:
        console.print("\nNo panorama candidates.")
        return

    console.print()
    for group in result.groups:
        _print_group(group)


def _print_group(group: PanoGroup) -> None:
    """
    Print one accepted or rejected run and its control-point links.

    Parameters
    ----------
    group : PanoGroup
        Run to display.
    """
    console = _get_console()
    label = group.kind
    if group.number is not None:
        label = f"{group.kind} {group.number}"
    status = "write" if group.accepted else "skip"
    span = _span_seconds(group)
    console.print(
        f"{status}  {label}  {len(group.frames)} frames  {span:.1f}s",
        style="green" if group.accepted else "yellow",
    )
    console.print(f"  {group.reason}")
    if group.folder is not None:
        console.print(f"  {group.folder}")
    if group.accepted and group.frames:
        index = anchor_index(len(group.frames))
        console.print(
            f"  anchor {group.frames[index].path.name} "
            f"({index + 1} of {len(group.frames)}, position and exposure)"
        )
    if group.cut_link is not None:
        console.print(f"  cut  {_format_link(group.cut_link)}")
    for link in group.neighbor_links:
        console.print(f"  {_format_link(link)}")
    for link in group.stride_links:
        console.print(
            f"  stride  {link.older} --{link.control_points}cp "
            f"{link.gap_seconds:.1f}s-- {link.newer}"
        )


def _format_link(link: LinkReport) -> str:
    """
    Format one consecutive pair for the report.

    Parameters
    ----------
    link : LinkReport
        Pair to format.

    Returns
    -------
    str
        ``older --Ncp 0.4s-- newer``.
    """
    return (
        f"{link.older} --{link.control_points}cp {link.gap_seconds:.1f}s-- {link.newer}"
    )


def _format_timestamp(moment: datetime) -> str:
    """
    Format a capture time, keeping milliseconds when EXIF has them.

    Parameters
    ----------
    moment : datetime
        Capture time.

    Returns
    -------
    str
        ``HH:MM:SS`` or ``HH:MM:SS.mmm``.
    """
    if moment.microsecond:
        millis = moment.microsecond // 1000
        return moment.strftime("%H:%M:%S.") + f"{millis:03d}"
    return moment.strftime("%H:%M:%S")


def _span_seconds(group: PanoGroup) -> float:
    """
    Return the capture span of a run in seconds.

    Parameters
    ----------
    group : PanoGroup
        Run with at least one frame.

    Returns
    -------
    float
        Seconds from the first frame to the last. Zero for a single frame.
    """
    if len(group.frames) < 2:
        return 0.0
    start = group.frames[0].timestamp
    end = group.frames[-1].timestamp
    return (end - start).total_seconds()
