"""
Game folder animation via ffmpeg.

Turns each ``Game##_<date>_<start>-<end>`` album into a matching ``.mp4``
using either a target video duration (FPS derived) or an explicit FPS.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from .game import _IMAGE_SUFFIXES

# Matches Game01_12Oct2025_140118-140119 and wider pads (Game001_...).
_GAME_FOLDER_RE = re.compile(r"^Game\d+_\d{1,2}[A-Za-z]{3}\d{4}_\d{6}-\d{6}$")

ProgressCallback = Callable[[str, str], None]

_SIZE_RE = re.compile(r"^(\d*)[xX](\d*)$")


@dataclass(frozen=True)
class VideoSize:
    """
    Target frame size for the encoded video.

    One axis may be omitted. ffmpeg then keeps the source aspect ratio
    and rounds the computed side to a multiple of 2 (``yuv420p``).

    Parameters
    ----------
    width : int, optional
        Output width in pixels. ``None`` means derive from height.
    height : int, optional
        Output height in pixels. ``None`` means derive from width.
    """

    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def label(self) -> str:
        """Return the size in ``WIDTHxHEIGHT`` form, omitting a free axis."""
        width = "" if self.width is None else str(self.width)
        height = "" if self.height is None else str(self.height)
        return f"{width}x{height}"

    def scale_filter(self) -> str:
        """
        Return an ffmpeg ``scale`` filter for this size.

        Returns
        -------
        str
            Filter such as ``scale=5568:3712``, ``scale=5568:-2``, or
            ``scale=-2:1080``.
        """
        width = -2 if self.width is None else self.width
        height = -2 if self.height is None else self.height
        return f"scale={width}:{height}"


def parse_video_size(spec: str) -> VideoSize:
    """
    Parse a frame-size string.

    Accepted forms:

    - ``5568x3712`` — exact width and height
    - ``5568x`` — width only; height follows the photo aspect ratio
    - ``x3712`` or ``x1080`` — height only (``x1080`` is 1080p)

    Parameters
    ----------
    spec : str
        Size text from ``--size``.

    Returns
    -------
    VideoSize
        Parsed axes. A missing axis is ``None``.

    Raises
    ------
    ValueError
        If the text is not a size, a number is not a positive even integer,
        or both axes are missing.
    """
    text = spec.strip()
    match = _SIZE_RE.match(text)
    if match is None or (match.group(1) == "" and match.group(2) == ""):
        raise ValueError(
            f"Invalid size {spec!r}. Use WIDTHxHEIGHT (5568x3712), "
            "WIDTHx (5568x), or xHEIGHT (x1080)."
        )
    width = _parse_size_axis(match.group(1), "width")
    height = _parse_size_axis(match.group(2), "height")
    return VideoSize(width=width, height=height)


def _parse_size_axis(raw: str, name: str) -> Optional[int]:
    """
    Parse one side of a size spec.

    Parameters
    ----------
    raw : str
        Digits, or empty when that axis is omitted.
    name : str
        ``width`` or ``height``, used in error text.

    Returns
    -------
    int or None
        Pixel count, or ``None`` when omitted.

    Raises
    ------
    ValueError
        If the value is zero or odd. libx264 ``yuv420p`` needs even sides.
    """
    if raw == "":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if value % 2 != 0:
        raise ValueError(f"{name} must be even for yuv420p video, got {value}")
    return value


@dataclass(frozen=True)
class AnimatePlan:
    """
    One game album ready to encode.

    Parameters
    ----------
    game_dir : Path
        Album folder named ``Game##_<date>_<start>-<end>``.
    photos : list of Path
        Image files in capture/name order.
    fps : float
        Frames per second for ffmpeg.
    output_path : Path
        Destination ``.mp4`` next to the album folder (same stem as the folder).
    size : VideoSize, optional
        Output frame size. ``None`` keeps each photo's native resolution.
    """

    game_dir: Path
    photos: List[Path]
    fps: float
    output_path: Path
    size: Optional[VideoSize] = None

    @property
    def duration_seconds(self) -> float:
        """Planned video length from photo count and FPS."""
        if self.fps <= 0.0 or not self.photos:
            return 0.0
        return float(len(self.photos)) / float(self.fps)


@dataclass(frozen=True)
class AnimateResult:
    """
    Outcome of one game encode.

    Parameters
    ----------
    game_dir : Path
        Album that was processed.
    output_path : Path
        Written (or skipped) video path.
    success : bool
        True when encoding finished or was skipped intentionally.
    skipped : bool
        True when an existing video was left alone.
    message : str
        Short status for the CLI.
    """

    game_dir: Path
    output_path: Path
    success: bool
    skipped: bool
    message: str


def is_game_folder(path: Path) -> bool:
    """
    Return whether ``path`` looks like a split game album directory.

    Parameters
    ----------
    path : Path
        Candidate directory.

    Returns
    -------
    bool
        True when the name matches ``Game##_<date>_<HHMMSS>-<HHMMSS>``.
    """
    return path.is_dir() and _GAME_FOLDER_RE.match(path.name) is not None


def discover_game_folders(inputs: Sequence[Path]) -> List[Path]:
    """
    Resolve CLI inputs into game album directories.

    A path that is itself a game folder is kept. Otherwise immediate
    child directories matching the game naming pattern are collected.

    Parameters
    ----------
    inputs : sequence of Path
        Game folders and/or parent folders that contain them.

    Returns
    -------
    list of Path
        Unique game albums, sorted by folder name.

    Raises
    ------
    ValueError
        If nothing matching the game folder pattern is found.
    """
    found: List[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        candidates: List[Path] = []
        if is_game_folder(path):
            candidates.append(path.resolve())
        elif path.is_dir():
            candidates.extend(
                child.resolve()
                for child in sorted(path.iterdir())
                if is_game_folder(child)
            )
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            found.append(candidate)

    if not found:
        raise ValueError(
            "No game folders found. Expected names like "
            "Game01_12Oct2025_140118-140119"
        )
    return sorted(found, key=lambda item: item.name)


def list_game_photos(game_dir: Path) -> List[Path]:
    """
    List image files in a game album in filename order.

    Symlinks are followed. Non-image files and nested directories are
    ignored.

    Parameters
    ----------
    game_dir : Path
        Game album directory.

    Returns
    -------
    list of Path
        Resolved image paths sorted by basename.
    """
    photos: List[Path] = []
    for entry in game_dir.iterdir():
        if not entry.is_file() and not entry.is_symlink():
            continue
        if entry.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if entry.name.startswith("."):
            continue
        photos.append(entry)
    photos.sort(key=lambda item: item.name.lower())
    return photos


def resolve_fps(
    photo_count: int,
    *,
    fps: Optional[float] = None,
    duration_seconds: Optional[float] = None,
) -> float:
    """
    Choose an encode frame rate from ``--fps`` or ``--duration``.

    Duration mode sets FPS so ``photo_count / fps == duration_seconds``.

    Parameters
    ----------
    photo_count : int
        Number of frames in the album.
    fps : float, optional
        Explicit frames per second.
    duration_seconds : float, optional
        Target video length in seconds.

    Returns
    -------
    float
        Positive frames per second.

    Raises
    ------
    ValueError
        If inputs are inconsistent or yield a non-positive rate.
    """
    if fps is not None and duration_seconds is not None:
        raise ValueError("Pass only one of fps or duration_seconds")
    if fps is None and duration_seconds is None:
        raise ValueError("Pass fps or duration_seconds")
    if photo_count < 1:
        raise ValueError("Need at least one photo to animate")

    if fps is not None:
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        return float(fps)

    assert duration_seconds is not None
    if duration_seconds <= 0.0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")
    return float(photo_count) / float(duration_seconds)


def plan_animation(
    game_dir: Path,
    *,
    fps: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    size: Optional[VideoSize] = None,
) -> AnimatePlan:
    """
    Build an encode plan for one game album.

    Parameters
    ----------
    game_dir : Path
        Album directory.
    fps : float, optional
        Explicit frame rate.
    duration_seconds : float, optional
        Target video length; FPS is derived.
    size : VideoSize, optional
        Output frame size. Omitted axes keep the photo aspect ratio.

    Returns
    -------
    AnimatePlan
        Paths, frame rate, and optional scale for ffmpeg.

    Raises
    ------
    ValueError
        If the folder has no images or rate options are invalid.
    """
    photos = list_game_photos(game_dir)
    if not photos:
        raise ValueError(f"No images in {game_dir}")
    rate = resolve_fps(len(photos), fps=fps, duration_seconds=duration_seconds)
    output_path = game_dir.parent / f"{game_dir.name}.mp4"
    return AnimatePlan(
        game_dir=game_dir,
        photos=photos,
        fps=rate,
        output_path=output_path,
        size=size,
    )


def find_ffmpeg(ffmpeg_bin: Optional[str] = None) -> str:
    """
    Locate an ffmpeg executable.

    Parameters
    ----------
    ffmpeg_bin : str, optional
        Explicit path or command name.

    Returns
    -------
    str
        Absolute path or command name that exists on PATH.

    Raises
    ------
    RuntimeError
        If ffmpeg cannot be found.
    """
    candidate = ffmpeg_bin or "ffmpeg"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise RuntimeError(
            "ffmpeg is required for game animation. Install ffmpeg and "
            "ensure it is on PATH, or pass --ffmpeg /path/to/ffmpeg."
        )
    return resolved


def _write_concat_list(photos: Sequence[Path], fps: float, list_path: Path) -> None:
    """
    Write an ffmpeg concat demuxer list with per-frame duration.

    Parameters
    ----------
    photos : sequence of Path
        Frames in display order.
    fps : float
        Frames per second (sets ``duration`` to ``1/fps``).
    list_path : Path
        Destination text file.
    """
    frame_duration = 1.0 / float(fps)
    lines: List[str] = []
    for photo in photos:
        # Concat demuxer needs escaped single quotes in paths.
        escaped = str(photo.resolve()).replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {frame_duration:.10f}")
    # Repeat the last file once without duration so the final frame holds.
    last = str(photos[-1].resolve()).replace("'", r"'\''")
    lines.append(f"file '{last}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frame_summary(plan: AnimatePlan) -> str:
    """
    Describe frame count, rate, length, and optional output size.

    Parameters
    ----------
    plan : AnimatePlan
        Encode plan.

    Returns
    -------
    str
        Text such as ``3 frames @ 0.5 fps, 6.00s, x1080``.
    """
    size_note = f", {plan.size.label}" if plan.size is not None else ""
    return (
        f"{len(plan.photos)} frames @ {plan.fps:.4g} fps, "
        f"{plan.duration_seconds:.2f}s{size_note}"
    )


def build_ffmpeg_command(ffmpeg: str, list_path: Path, plan: AnimatePlan) -> List[str]:
    """
    Build the ffmpeg argv for one game album.

    Parameters
    ----------
    ffmpeg : str
        ffmpeg executable.
    list_path : Path
        Concat demuxer list.
    plan : AnimatePlan
        Photos, rate, destination, and optional frame size.

    Returns
    -------
    list of str
        Argument vector, including the executable.
    """
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
    ]
    if plan.size is not None:
        cmd.extend(["-vf", plan.size.scale_filter()])
    cmd.extend(
        [
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(plan.output_path),
        ]
    )
    return cmd


def encode_game_video(
    plan: AnimatePlan,
    *,
    ffmpeg_bin: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> AnimateResult:
    """
    Encode one game album to MP4 with ffmpeg.

    Parameters
    ----------
    plan : AnimatePlan
        Folder, photos, and FPS.
    ffmpeg_bin : str, optional
        ffmpeg executable override.
    force : bool
        Overwrite an existing ``.mp4``.
    dry_run : bool
        Plan only; do not invoke ffmpeg.

    Returns
    -------
    AnimateResult
        Success, skip, or failure details.
    """
    if plan.output_path.exists() and not force:
        return AnimateResult(
            game_dir=plan.game_dir,
            output_path=plan.output_path,
            success=True,
            skipped=True,
            message=f"exists (use --force): {plan.output_path.name}",
        )

    if dry_run:
        return AnimateResult(
            game_dir=plan.game_dir,
            output_path=plan.output_path,
            success=True,
            skipped=False,
            message=(f"would write {plan.output_path.name} ({_frame_summary(plan)})"),
        )

    try:
        ffmpeg = find_ffmpeg(ffmpeg_bin)
    except RuntimeError as exc:
        return AnimateResult(
            game_dir=plan.game_dir,
            output_path=plan.output_path,
            success=False,
            skipped=False,
            message=str(exc),
        )

    with tempfile.TemporaryDirectory(prefix="sportball_animate_") as tmp:
        list_path = Path(tmp) / "concat.txt"
        _write_concat_list(plan.photos, plan.fps, list_path)
        cmd = build_ffmpeg_command(ffmpeg, list_path, plan)
        logger.debug("Running {}", " ".join(cmd))
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return AnimateResult(
                game_dir=plan.game_dir,
                output_path=plan.output_path,
                success=False,
                skipped=False,
                message=f"ffmpeg failed to start: {exc}",
            )

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip().splitlines()
            tail = stderr[-8:] if stderr else ["unknown ffmpeg error"]
            return AnimateResult(
                game_dir=plan.game_dir,
                output_path=plan.output_path,
                success=False,
                skipped=False,
                message="ffmpeg error: " + " | ".join(tail),
            )

    return AnimateResult(
        game_dir=plan.game_dir,
        output_path=plan.output_path,
        success=True,
        skipped=False,
        message=f"wrote {plan.output_path.name} ({_frame_summary(plan)})",
    )


def animate_games(
    inputs: Sequence[Path],
    *,
    fps: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    size: Optional[VideoSize] = None,
    workers: int = 2,
    force: bool = False,
    dry_run: bool = False,
    ffmpeg_bin: Optional[str] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> List[AnimateResult]:
    """
    Animate one or more game albums concurrently with ffmpeg.

    Parameters
    ----------
    inputs : sequence of Path
        Game folders and/or parents that contain them.
    fps : float, optional
        Explicit frames per second (fractional allowed).
    duration_seconds : float, optional
        Target length per game; FPS is ``photos / duration``.
    size : VideoSize, optional
        Output frame size (``5568x3712``, ``5568x``, ``x1080``).
    workers : int
        Parallel ffmpeg processes.
    force : bool
        Overwrite existing videos.
    dry_run : bool
        Report plans without encoding.
    ffmpeg_bin : str, optional
        ffmpeg path override.
    on_progress : callable, optional
        Called as ``(game_name, status)`` when each job finishes.

    Returns
    -------
    list of AnimateResult
        One result per discovered game folder, in folder-name order.
    """
    folders = discover_game_folders(inputs)
    plans: List[AnimatePlan] = []
    early_failures: List[AnimateResult] = []

    for folder in folders:
        try:
            plans.append(
                plan_animation(
                    folder,
                    fps=fps,
                    duration_seconds=duration_seconds,
                    size=size,
                )
            )
        except ValueError as exc:
            early_failures.append(
                AnimateResult(
                    game_dir=folder,
                    output_path=folder.parent / f"{folder.name}.mp4",
                    success=False,
                    skipped=False,
                    message=str(exc),
                )
            )

    worker_count = max(1, int(workers))
    results_by_dir: Dict[Path, AnimateResult] = {
        item.game_dir: item for item in early_failures
    }

    def _run(plan: AnimatePlan) -> AnimateResult:
        return encode_game_video(
            plan,
            ffmpeg_bin=ffmpeg_bin,
            force=force,
            dry_run=dry_run,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures: Dict[Future[AnimateResult], AnimatePlan] = {
            pool.submit(_run, plan): plan for plan in plans
        }
        for future in as_completed(futures):
            plan = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - surface encode crashes
                result = AnimateResult(
                    game_dir=plan.game_dir,
                    output_path=plan.output_path,
                    success=False,
                    skipped=False,
                    message=f"unexpected error: {exc}",
                )
            results_by_dir[result.game_dir] = result
            if on_progress is not None:
                on_progress(result.game_dir.name, result.message)

    ordered: List[AnimateResult] = []
    for folder in folders:
        if folder in results_by_dir:
            ordered.append(results_by_dir[folder])
    return ordered
