"""
Action-panorama detection.

Finds stitchable pans in a burst of sports photos. A known panorama is the
run of frames before a pair of nearly uniform frames (lens covered, a shot
of sky, or a zoom into the ground). A guessed panorama has no marker:
neighboring frames share control points, and frames a few shots apart do
not. Hugin's ``cpfind`` supplies the control points.

The command writes symlink folders and a ``.pto`` project. Every project
is optimized for yaw, pitch, roll, and field of view only. Stitching
and a black-canvas crop are on by default. ``hugin_executor`` stitches
each project in process, with no batch window. ``--crop`` then writes
``<pano_name>_cropped.jpg`` beside each stitched panorama, with the
black canvas removed.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from loguru import logger
from PIL import Image, ImageChops, ImageStat, UnidentifiedImageError
from tqdm import tqdm

from .game import (
    collect_photo_paths,
    expand_input_directories,
    format_game_id,
    timestamp_from_exif_tags,
)

DEFAULT_MARKER_VAR = 100.0
DEFAULT_MARKER_SAMPLE = 64
DEFAULT_MARKER_COUNT = 2
DEFAULT_DISCONTINUITY_SECONDS = 10.0
DEFAULT_SPLIT_POINTS = 0
DEFAULT_STRIDE = 3
DEFAULT_OVERLAP_POINTS = 10
DEFAULT_FAR_POINTS = 0
DEFAULT_MIN_FRAMES = 5
POSITION_VIEW_VARIABLES = "y,p,r,v"
DEFAULT_CP_EDGE = 1600
DEFAULT_CROP_THRESHOLD = 8
PANOS_SUFFIX = "-panos"
_STITCH_SUFFIXES = (".jpg", ".jpeg", ".tif", ".tiff")
_PANO_STEM_RE = re.compile(r"^(?:known_pano|guessed_pano)\d+_.+")

_IMAGE_LINE_RE = re.compile(r'^i\b.*\bn"([^"]*)"')
_CONTROL_POINT_RE = re.compile(
    r"^c\s+n(\d+)\s+N(\d+)\s+x([-\d.]+)\s+y([-\d.]+)\s+" r"X([-\d.]+)\s+Y([-\d.]+)"
)


@dataclass(frozen=True)
class PanoConfig:
    """
    Thresholds for uniform-frame markers and stitch geometry.

    Parameters
    ----------
    marker_var : float
        Maximum variance of the grayscale thumbnail around its mean. At or
        below this, a frame is a marker. Covered-lens, sky, and zoomed
        ground frames are all low.
    marker_sample : int
        Edge length of the grayscale thumbnail used for that variance.
    marker_count : int
        Consecutive uniform frames that mark a known panorama.
    discontinuity : float
        Seconds. A larger gap is two different shots.
    split_points : int
        A consecutive pair with this many control points or fewer is a
        candidate cut while walking back from a uniform marker.
    boundary : str
        ``largest-gap`` or ``first``.
    guess : bool
        Search for unmarked panoramas after known ones are claimed.
    stride : int
        Frame distance that must not overlap in a guessed panorama.
    overlap_points : int
        Minimum control points between neighbors to count as overlap.
    far_points : int
        Maximum control points allowed ``stride`` frames apart.
    min_frames : int
        Minimum frames written as a panorama. Guessed runs also need
        ``stride + 1`` frames so the stride test has a pair.
    cp_edge : int
        Long-edge size of the JPEG previews passed to ``cpfind``.
    copy_files : bool
        Copy photos into the output folder instead of symlinking.
    write_pto : bool
        Write a Hugin project next to the photos.
    stitch : bool
        After every project is optimized, stitch it with
        ``hugin_executor``. No batch window is opened.
    crop : bool
        After the panoramas exist, write ``<pano_name>_cropped.jpg`` for
        each stitched image, dropping the black canvas.
    dry_run : bool
        Score and match, but do not create folders.
    progress : bool
        Print a tqdm bar and a line for each frame, survey, and decision.
    """

    marker_var: float = DEFAULT_MARKER_VAR
    marker_sample: int = DEFAULT_MARKER_SAMPLE
    marker_count: int = DEFAULT_MARKER_COUNT
    discontinuity: float = DEFAULT_DISCONTINUITY_SECONDS
    split_points: int = DEFAULT_SPLIT_POINTS
    boundary: str = "largest-gap"
    guess: bool = True
    stride: int = DEFAULT_STRIDE
    overlap_points: int = DEFAULT_OVERLAP_POINTS
    far_points: int = DEFAULT_FAR_POINTS
    min_frames: int = DEFAULT_MIN_FRAMES
    cp_edge: int = DEFAULT_CP_EDGE
    copy_files: bool = False
    write_pto: bool = True
    stitch: bool = True
    crop: bool = True
    dry_run: bool = False
    progress: bool = True


@dataclass(frozen=True)
class Frame:
    """
    One decoded photo in capture order.

    Parameters
    ----------
    path : Path
        Image file.
    timestamp : datetime
        EXIF capture time.
    variation : float
        Variance of the grayscale thumbnail around its mean.
    width : int
        Full-resolution width in pixels.
    height : int
        Full-resolution height in pixels.
    mean : float
        Mean gray level of that thumbnail, 0 to 255.
    """

    path: Path
    timestamp: datetime
    variation: float
    width: int
    height: int
    mean: float = 0.0


@dataclass(frozen=True)
class ControlPoint:
    """
    One Hugin control point in full-resolution pixel coordinates.

    ``x`` and ``y`` lie on ``left_index``. ``x_right`` and ``y_right`` lie
    on ``right_index``. ``left_index`` is less than ``right_index``.

    Parameters
    ----------
    left_index : int
        Earlier image in the surveyed sequence.
    right_index : int
        Later image in the surveyed sequence.
    x : float
        Horizontal position on the earlier image.
    y : float
        Vertical position on the earlier image.
    x_right : float
        Horizontal position on the later image.
    y_right : float
        Vertical position on the later image.
    """

    left_index: int
    right_index: int
    x: float
    y: float
    x_right: float
    y_right: float


@dataclass
class PairSurvey:
    """
    Control points for one ordered sequence of frames.

    Parameters
    ----------
    counts : dict
        Control-point totals keyed by ``(earlier_index, later_index)``.
    points : list of ControlPoint
        Individual points in full-resolution coordinates.
    """

    counts: Dict[Tuple[int, int], int]
    points: List[ControlPoint]


@dataclass(frozen=True)
class LinkReport:
    """
    Overlap between two frames in a candidate run.

    Parameters
    ----------
    older : str
        Basename of the earlier frame.
    newer : str
        Basename of the later frame.
    control_points : int
        How many control points ``cpfind`` reported.
    gap_seconds : float
        Capture-time gap in seconds.
    """

    older: str
    newer: str
    control_points: int
    gap_seconds: float


@dataclass
class PanoGroup:
    """
    A known panorama, a guessed panorama, or a rejected run.

    Parameters
    ----------
    kind : str
        ``known``, ``guessed``, ``held``, or ``open``.
    accepted : bool
        True when a folder should be written.
    reason : str
        Why the run was accepted or rejected.
    frames : list of Frame
        Photos that belong in the stitch, excluding uniform markers.
    marker : list of Frame
        Black frames that ended a known panorama. Empty otherwise.
    neighbor_links : list of LinkReport
        Consecutive pairs inside ``frames``.
    stride_links : list of LinkReport
        Pairs ``stride`` frames apart inside ``frames``.
    cut_link : LinkReport, optional
        The weak pair that started a known panorama, when the walk-back
        dropped older frames.
    points : list of ControlPoint
        Full-resolution points indexed into ``frames``.
    number : int, optional
        1-based index among accepted panoramas, in capture order.
    folder : Path, optional
        Folder written for an accepted panorama.
    """

    kind: str
    accepted: bool
    reason: str
    frames: List[Frame]
    marker: List[Frame] = field(default_factory=list)
    neighbor_links: List[LinkReport] = field(default_factory=list)
    stride_links: List[LinkReport] = field(default_factory=list)
    cut_link: Optional[LinkReport] = None
    points: List[ControlPoint] = field(default_factory=list)
    number: Optional[int] = None
    folder: Optional[Path] = None


@dataclass
class PanoResult:
    """
    Outcome of one ``sb pano`` pass.

    Parameters
    ----------
    frames : list of Frame
        Decoded photos in capture order.
    groups : list of PanoGroup
        Accepted and rejected runs.
    marker_runs : list of tuple of int
        Inclusive index ranges of uniform markers inside ``frames``.
    skipped_no_exif : int
        Files with no usable capture time.
    skipped_undecodable : int
        Files Pillow could not decode.
    output_dirs : list of Path
        Sibling directories that receive each input's panoramas. Listed
        even on a dry run, when nothing is created.
    cropped : list of Path
        ``<pano_name>_cropped.jpg`` files written from stitched panoramas.
    """

    frames: List[Frame]
    groups: List[PanoGroup]
    marker_runs: List[Tuple[int, int]]
    skipped_no_exif: int
    skipped_undecodable: int
    output_dirs: List[Path] = field(default_factory=list)
    cropped: List[Path] = field(default_factory=list)


SurveyFn = Callable[[Sequence[Frame], int], PairSurvey]


def validate_config(config: PanoConfig) -> None:
    """
    Reject thresholds that cannot define a panorama.

    Parameters
    ----------
    config : PanoConfig
        Thresholds from the CLI.

    Raises
    ------
    ValueError
        If a threshold is out of range or ``boundary`` is unknown.
    """
    if config.marker_var < 0.0:
        raise ValueError("--marker-var must be zero or positive")
    if config.marker_sample < 1:
        raise ValueError("--marker-sample must be at least 1")
    if config.marker_count < 1:
        raise ValueError("--marker-count must be at least 1")
    if config.discontinuity < 0.0:
        raise ValueError("--discontinuity must be zero or positive")
    if config.split_points < 0:
        raise ValueError("--split-points must be zero or positive")
    if config.boundary not in ("largest-gap", "first"):
        raise ValueError("--boundary must be largest-gap or first")
    if config.stride < 1:
        raise ValueError("--stride must be at least 1")
    if config.overlap_points < 0:
        raise ValueError("--overlap-points must be zero or positive")
    if config.far_points < 0:
        raise ValueError("--far-points must be zero or positive")
    if config.min_frames < 1:
        raise ValueError("--min-frames must be at least 1")
    if config.cp_edge < 1:
        raise ValueError("--cp-edge must be at least 1")
    if config.stitch and not config.write_pto:
        raise ValueError("--stitch requires the Hugin project (--pto)")


def measure_marker_variation(path: Path, sample: int) -> Tuple[float, float, int, int]:
    """
    Score how uniform a frame is, at any brightness.

    The thumbnail is ``sample`` by ``sample`` grayscale pixels. The mean
    is the average gray level. The variation is the mean squared deviation
    from that mean. A covered lens, a frame of sky, and a zoomed frame of
    ground all score near 0. A detailed frame scores high.

    Parameters
    ----------
    path : Path
        Image Pillow can decode.
    sample : int
        Thumbnail edge in pixels. At least 1.

    Returns
    -------
    tuple of float, float, int, int
        Variation, mean gray level, full width, and full height.

    Raises
    ------
    ValueError
        If ``sample`` is less than 1.
    OSError
        If the file cannot be read.
    UnidentifiedImageError
        If Pillow does not recognize the file.
    """
    if sample < 1:
        raise ValueError("marker sample edge must be at least 1")
    with Image.open(path) as image:
        width, height = image.size
        image.draft("L", (sample, sample))
        gray = image.convert("L")
        if gray.size != (sample, sample):
            gray = gray.resize((sample, sample), Image.Resampling.BOX)
        stat = ImageStat.Stat(gray)
        mean = float(stat.mean[0])
        variation = float(stat.var[0])
    return variation, mean, int(width), int(height)


def find_marker_runs(
    frames: Sequence[Frame],
    marker_var: float,
    marker_count: int,
    discontinuity: float,
) -> List[Tuple[int, int]]:
    """
    Find runs of consecutive uniform frames.

    A run breaks when a frame's variation is above ``marker_var`` or the
    gap since the previous frame is greater than ``discontinuity`` seconds.
    Runs shorter than ``marker_count`` are ignored.

    Parameters
    ----------
    frames : sequence of Frame
        Photos in capture order.
    marker_var : float
        Maximum grayscale variance, inclusive.
    marker_count : int
        Minimum run length.
    discontinuity : float
        Maximum seconds between frames that still belong to one run.

    Returns
    -------
    list of tuple of int
        Inclusive ``(start, end)`` indexes into ``frames``.
    """
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, frame in enumerate(frames):
        is_marker = frame.variation <= marker_var
        if start is not None and is_marker:
            gap = _gap_seconds(frames[index - 1], frame)
            if gap > discontinuity:
                _close_marker_run(runs, start, index - 1, marker_count)
                start = index
            continue
        if is_marker:
            start = index
            continue
        if start is not None:
            _close_marker_run(runs, start, index - 1, marker_count)
            start = None
    if start is not None:
        _close_marker_run(runs, start, len(frames) - 1, marker_count)
    return runs


def known_window_indices(
    frames: Sequence[Frame],
    marker_start: int,
    discontinuity: float,
    marker_var: float,
    claimed: Mapping[int, bool],
) -> List[int]:
    """
    Collect the frames immediately before a uniform-frame marker.

    Walking stops at a gap greater than ``discontinuity`` seconds, at a
    uniform frame, or at an index already claimed by a later panorama.

    Parameters
    ----------
    frames : sequence of Frame
        Photos in capture order.
    marker_start : int
        Index of the first uniform frame in the marker.
    discontinuity : float
        Maximum seconds that still count as the same shot.
    marker_var : float
        Variance threshold used to recognize marker frames.
    claimed : mapping
        Indexes that a later marker already took. Presence means claimed.

    Returns
    -------
    list of int
        Inclusive window indexes from oldest to newest. Empty when the
        marker is at the start of the sequence.
    """
    end = marker_start - 1
    if end < 0:
        return []
    if end in claimed or frames[end].variation <= marker_var:
        return []
    start = end
    index = end
    while index > 0:
        previous = index - 1
        gap = _gap_seconds(frames[previous], frames[index])
        if gap > discontinuity:
            break
        if previous in claimed or frames[previous].variation <= marker_var:
            break
        start = previous
        index = previous
    return list(range(start, end + 1))


def cut_known_start(
    control_points: Sequence[int],
    gap_seconds: Sequence[float],
    split_points: int,
    boundary: str,
) -> int:
    """
    Choose where a known panorama starts inside its temporal window.

    ``control_points[i]`` and ``gap_seconds[i]`` describe the pair between
    local frames ``i`` and ``i + 1``. A pair at or below ``split_points``
    is a weak link.

    ``largest-gap`` cuts at the weak link with the largest time gap. Equal
    gaps keep the link closer to the marker frames (the higher index), so
    the folder is the pan just finished. ``first`` cuts at the weak link
    closest to the marker frames. No weak link keeps the whole window.

    Parameters
    ----------
    control_points : sequence of int
        Control points on each consecutive pair.
    gap_seconds : sequence of float
        Time gap of each consecutive pair, same length.
    split_points : int
        Weak-link threshold, inclusive.
    boundary : str
        ``largest-gap`` or ``first``.

    Returns
    -------
    int
        Local index of the first frame to keep.

    Raises
    ------
    ValueError
        If the two sequences differ in length or ``boundary`` is unknown.
    """
    if len(control_points) != len(gap_seconds):
        raise ValueError("control points and gaps must describe the same pairs")
    if boundary not in ("largest-gap", "first"):
        raise ValueError("boundary must be largest-gap or first")
    weak = [
        index for index, points in enumerate(control_points) if points <= split_points
    ]
    if not weak:
        return 0
    if boundary == "first":
        return max(weak) + 1
    best = weak[0]
    best_key = (gap_seconds[best], best)
    for index in weak[1:]:
        key = (gap_seconds[index], index)
        if key > best_key:
            best_key = key
            best = index
    return best + 1


def classify_burst_ranges(
    count: int,
    counts: Mapping[Tuple[int, int], int],
    stride: int,
    overlap_points: int,
    far_points: int,
    min_frames: int,
) -> List[Tuple[str, int, int]]:
    """
    Label guessed panoramas and rejected runs inside one time burst.

    A guessed panorama is at least ``max(min_frames, stride + 1)`` frames
    where every neighbor has ``overlap_points`` or more and every pair
    ``stride`` apart has ``far_points`` or fewer. A hold is a run of that
    length whose neighbors overlap but the stride pairs still overlap.
    Anything else of that length, when the burst produced no panorama and
    no hold, is ``open`` so the control-point counts can be tuned.

    Parameters
    ----------
    count : int
        Number of frames in the burst.
    counts : mapping
        Control points keyed by ``(earlier, later)`` local indexes.
    stride : int
        Distance that must not overlap.
    overlap_points : int
        Minimum neighbor control points.
    far_points : int
        Maximum control points at ``stride``.
    min_frames : int
        Minimum panorama length before the stride requirement.

    Returns
    -------
    list of tuple
        ``(kind, start, end)`` inclusive. ``kind`` is ``guessed``,
        ``held``, or ``open``.
    """
    min_len = max(min_frames, stride + 1)
    if count < min_len:
        return []

    def neighbor_ok(index: int) -> bool:
        return counts.get((index, index + 1), 0) >= overlap_points

    def stride_ok(index: int) -> bool:
        return counts.get((index, index + stride), 0) <= far_points

    def pano_valid(start: int, end: int) -> bool:
        if end - start + 1 < min_len:
            return False
        for index in range(start, end):
            if not neighbor_ok(index):
                return False
        for index in range(start, end - stride + 1):
            if not stride_ok(index):
                return False
        return True

    def longest_pano(start: int) -> Optional[int]:
        end = start
        best: Optional[int] = None
        while True:
            if pano_valid(start, end):
                best = end
            if end + 1 >= count or not neighbor_ok(end):
                break
            next_end = end + 1
            stride_index = next_end - stride
            if stride_index >= start and not stride_ok(stride_index):
                break
            end = next_end
        return best

    panos: List[Tuple[int, int]] = []
    start = 0
    while start < count:
        end = longest_pano(start)
        if end is None:
            start += 1
            continue
        panos.append((start, end))
        start = end + 1

    covered = {index for begin, end in panos for index in range(begin, end + 1)}
    holds: List[Tuple[int, int]] = []
    index = 0
    while index < count:
        if index + 1 < count and neighbor_ok(index):
            end = index
            while end + 1 < count and neighbor_ok(end):
                end += 1
            cursor = index
            while cursor <= end:
                if cursor in covered:
                    cursor += 1
                    continue
                segment_end = cursor
                while segment_end + 1 <= end and (segment_end + 1) not in covered:
                    segment_end += 1
                if segment_end - cursor + 1 >= min_len and not pano_valid(
                    cursor, segment_end
                ):
                    holds.append((cursor, segment_end))
                cursor = segment_end + 1
            index = end + 1
            continue
        index += 1

    labeled: List[Tuple[str, int, int]] = [
        ("guessed", begin, end) for begin, end in panos
    ]
    labeled.extend(("held", begin, end) for begin, end in holds)
    if not labeled:
        labeled.append(("open", 0, count - 1))
    return labeled


def parse_pto_control_points(text: str) -> Tuple[List[str], List[ControlPoint]]:
    """
    Read image paths and control points from a Hugin ``.pto`` file.

    Point coordinates are returned as written. Callers scale them when
    ``cpfind`` ran on a preview.

    Parameters
    ----------
    text : str
        Full ``.pto`` document.

    Returns
    -------
    tuple
        Image paths in project order, and control points. Each point's
        indexes refer to that image list. The earlier image is
        ``left_index``.
    """
    images: List[str] = []
    points: List[ControlPoint] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        image_match = _IMAGE_LINE_RE.match(line)
        if image_match is not None:
            images.append(image_match.group(1))
            continue
        point_match = _CONTROL_POINT_RE.match(line)
        if point_match is None:
            continue
        left = int(point_match.group(1))
        right = int(point_match.group(2))
        x_left = float(point_match.group(3))
        y_left = float(point_match.group(4))
        x_right = float(point_match.group(5))
        y_right = float(point_match.group(6))
        if left == right:
            continue
        if left > right:
            left, right = right, left
            x_left, x_right = x_right, x_left
            y_left, y_right = y_right, y_left
        points.append(
            ControlPoint(
                left_index=left,
                right_index=right,
                x=x_left,
                y=y_left,
                x_right=x_right,
                y_right=y_right,
            )
        )
    return images, points


def render_pto(
    names: Sequence[str],
    frames: Sequence[Frame],
    points: Sequence[ControlPoint],
) -> str:
    """
    Build a Hugin project that points at photos in a panorama folder.

    Parameters
    ----------
    names : sequence of str
        Basenames written into the folder, same order as ``frames``.
    frames : sequence of Frame
        Full-resolution photos.
    points : sequence of ControlPoint
        Control points indexed into ``frames``.

    Returns
    -------
    str
        ``.pto`` document text.

    Raises
    ------
    ValueError
        If ``names`` and ``frames`` differ in length.
    """
    if len(names) != len(frames):
        raise ValueError("pto image names must match the frame list")
    canvas_w = max((frame.width for frame in frames), default=0)
    canvas_h = max((frame.height for frame in frames), default=0)
    lines = [
        "# hugin project generated by sportball",
        "# control points were found on previews and scaled to full resolution",
        f'p f0 w{canvas_w} h{canvas_h} v50 E0 R0 n"JPEG"',
        "m g1 i0 m2 p0.00784314",
        "",
    ]
    for name, frame in zip(names, frames):
        quoted = name.replace('"', "_")
        lines.append(
            f"i w{frame.width} h{frame.height} f0 v50 "
            "Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 "
            f'TrX0 TrY0 TrZ0 j0 a0 b0 c0 d0 e0 g0 t0 n"{quoted}"'
        )
    lines.append("")
    for point in points:
        lines.append(
            f"c n{point.left_index} N{point.right_index} "
            f"x{point.x:.4f} y{point.y:.4f} "
            f"X{point.x_right:.4f} Y{point.y_right:.4f} t0"
        )
    lines.append("")
    lines.append("v")
    lines.append("")
    return "\n".join(lines)


def anchor_index(count: int) -> int:
    """
    Return the median image used as the position and exposure anchor.

    Five photos use the 3rd (index 2). Four photos use the 2nd (index 1).
    That is the earlier of the two middle photos when the count is even.

    Parameters
    ----------
    count : int
        Number of photos in the panorama. At least 1.

    Returns
    -------
    int
        Zero-based index of the anchor image.

    Raises
    ------
    ValueError
        If ``count`` is less than 1.
    """
    if count < 1:
        raise ValueError("a panorama anchor needs at least one photo")
    return (count - 1) // 2


def resolve_hugin_tool(name: str) -> str:
    """
    Locate a Hugin executable on ``PATH``.

    Parameters
    ----------
    name : str
        Executable name, such as ``cpfind`` or ``nona``.

    Returns
    -------
    str
        Path returned by ``shutil.which``.

    Raises
    ------
    RuntimeError
        If the tool is not on ``PATH``.
    """
    found = shutil.which(name)
    if found is None:
        package = "hugin" if name == "hugin_executor" else "hugin-tools"
        raise RuntimeError(
            f"{name} was not found on PATH. Install the {package} package."
        )
    return found


def survey_control_points(
    frames: Sequence[Frame],
    linearmatchlen: int,
    cp_edge: int,
    progress: bool = True,
) -> PairSurvey:
    """
    Run ``pto_gen`` and ``cpfind`` on downscaled previews.

    Control points are scaled back to the full-resolution frames. Matching
    is linear: each image is compared with the next ``linearmatchlen``
    images only.

    Parameters
    ----------
    frames : sequence of Frame
        Photos to match, in stitch order.
    linearmatchlen : int
        How many following images ``cpfind --linearmatch`` may use.
    cp_edge : int
        Long-edge size of each preview JPEG.
    progress : bool
        Show a tqdm bar for previews, ``pto_gen``, and ``cpfind``.

    Returns
    -------
    PairSurvey
        Counts and full-resolution points. Empty when fewer than two
        frames were given.

    Raises
    ------
    RuntimeError
        If either tool fails or a preview cannot be decoded.
    """
    if len(frames) < 2:
        return PairSurvey(counts={}, points=[])
    if linearmatchlen < 1:
        raise RuntimeError("cpfind linearmatch length must be at least 1")
    pto_gen_bin = resolve_hugin_tool("pto_gen")
    cpfind_bin = resolve_hugin_tool("cpfind")

    _say(
        f"cpfind {len(frames)} frames {_span_names(frames)} "
        f"linearmatchlen={linearmatchlen} edge={cp_edge}",
        progress,
    )
    with tempfile.TemporaryDirectory(prefix="sportball-pano-") as temp_name:
        temp_dir = Path(temp_name)
        previews: List[Path] = []
        scales: List[Tuple[float, float]] = []
        with _bar(len(frames) + 2, "cpfind", progress, unit="step") as bar:
            for index, frame in enumerate(frames):
                bar.set_postfix_str(frame.path.name)
                preview = temp_dir / f"{index:04d}.jpg"
                orig_w, orig_h, preview_w, preview_h = _write_preview(
                    frame.path, preview, cp_edge
                )
                previews.append(preview)
                scales.append((orig_w / float(preview_w), orig_h / float(preview_h)))
                _say(
                    f"  preview {frame.path.name} "
                    f"{orig_w}x{orig_h} -> {preview_w}x{preview_h}",
                    progress,
                )
                bar.update(1)

            skeleton = temp_dir / "survey.pto"
            matched = temp_dir / "survey_cp.pto"
            bar.set_postfix_str("pto_gen")
            _say(f"  pto_gen {len(previews)} previews", progress)
            _run_hugin(
                [pto_gen_bin, "-o", str(skeleton), *[str(path) for path in previews]]
            )
            bar.update(1)
            bar.set_postfix_str("matching")
            _say(
                f"  cpfind --linearmatch --linearmatchlen {linearmatchlen}",
                progress,
            )
            _run_hugin(
                [
                    cpfind_bin,
                    "--linearmatch",
                    "--linearmatchlen",
                    str(linearmatchlen),
                    "-o",
                    str(matched),
                    str(skeleton),
                ]
            )
            bar.update(1)
        document = matched.read_text(encoding="utf-8", errors="replace")
        image_paths, raw_points = parse_pto_control_points(document)
        ordered = _align_preview_points(raw_points, image_paths, previews)
        scaled = _scale_points(ordered, scales, len(frames))
    surveyed = PairSurvey(counts=_count_points(scaled), points=scaled)
    _say(
        f"  {len(surveyed.points)} control points across {len(surveyed.counts)} pairs",
        progress,
    )
    return surveyed


def detect_panos(
    frames: Sequence[Frame],
    config: PanoConfig,
    survey: SurveyFn,
) -> Tuple[List[PanoGroup], List[Tuple[int, int]]]:
    """
    Find known and guessed panoramas in an already scored frame list.

    Parameters
    ----------
    frames : sequence of Frame
        Decoded photos in capture order.
    config : PanoConfig
        Thresholds.
    survey : callable
        ``(frames, linearmatchlen) -> PairSurvey`` for one contiguous run.

    Returns
    -------
    tuple
        Groups (accepted and rejected) and inclusive marker-run indexes.

    Raises
    ------
    ValueError
        If ``config`` is inconsistent.
    """
    validate_config(config)
    show = config.progress
    runs = find_marker_runs(
        frames, config.marker_var, config.marker_count, config.discontinuity
    )
    _say(
        f"{len(runs)} uniform marker(s) in {len(frames)} frames "
        f"(var<={config.marker_var:g}, count>={config.marker_count})",
        show,
    )
    for start, end in runs:
        names = ", ".join(frames[index].path.name for index in range(start, end + 1))
        _say(f"  marker {names}", show)
    claimed: Dict[int, bool] = {}
    for start, end in runs:
        for index in range(start, end + 1):
            claimed[index] = True

    groups: List[PanoGroup] = []
    known_jobs = list(reversed(runs))
    with _bar(len(known_jobs), "Known panoramas", show, unit="marker") as bar:
        for start, end in known_jobs:
            marker_frames = [frames[index] for index in range(start, end + 1)]
            bar.set_postfix_str(marker_frames[0].path.name)
            window = known_window_indices(
                frames, start, config.discontinuity, config.marker_var, claimed
            )
            if not window:
                _say(f"No frames before {_span_names(marker_frames)}", show)
                bar.update(1)
                continue
            window_frames = [frames[index] for index in window]
            _say(
                f"Known window {_span_names(window_frames)} "
                f"({len(window_frames)} frames) before {_span_names(marker_frames)}",
                show,
            )
            linearmatchlen = config.stride if config.guess else 1
            if len(window) >= 2:
                surveyed = survey(window_frames, max(linearmatchlen, 1))
            else:
                surveyed = PairSurvey(counts={}, points=[])
            known = _known_group_from_window(
                frames,
                window,
                marker_frames,
                surveyed,
                config,
            )
            groups.append(known)
            _announce_group(known, show)
            local_cut = _local_cut_index(frames, window, known)
            if config.guess and local_cut > 0:
                prefix_groups = _groups_from_ranges(
                    frames,
                    window[:local_cut],
                    _slice_survey(surveyed, 0, local_cut),
                    config,
                )
                for group in prefix_groups:
                    _announce_group(group, show)
                groups.extend(prefix_groups)
            for index in window:
                claimed[index] = True
            bar.update(1)

    if config.guess:
        min_len = max(config.min_frames, config.stride + 1)
        bursts = [
            burst
            for burst in _remaining_bursts(frames, claimed, config)
            if len(burst) >= min_len
        ]
        with _bar(len(bursts), "Guessed panoramas", show, unit="burst") as bar:
            for burst in bursts:
                burst_frames = [frames[index] for index in burst]
                bar.set_postfix_str(burst_frames[0].path.name)
                _say(
                    f"Unmarked burst {_span_names(burst_frames)} "
                    f"({len(burst_frames)} frames), linearmatchlen={config.stride}",
                    show,
                )
                surveyed = survey(burst_frames, max(config.stride, 1))
                burst_groups = _groups_from_ranges(frames, burst, surveyed, config)
                for group in burst_groups:
                    _announce_group(group, show)
                groups.extend(burst_groups)
                bar.update(1)

    _assign_numbers(groups)
    groups.sort(
        key=lambda group: (
            group.frames[0].timestamp,
            group.frames[0].path.name,
            group.kind,
        )
    )
    return groups, runs


def sibling_pano_dir(directory: Path) -> Path:
    """
    Return the panorama directory that sits beside an input directory.

    Parameters
    ----------
    directory : Path
        One input folder, typically a game folder.

    Returns
    -------
    Path
        The input's parent, the input name, and ``-panos``.
        ``Game03_19Sep2026_120915-132501`` becomes
        ``Game03_19Sep2026_120915-132501-panos``.
    """
    return directory.parent / f"{directory.name}{PANOS_SUFFIX}"


def _drop_generated_pano_dirs(directories: Sequence[Path]) -> List[Path]:
    """
    Skip panorama siblings that belong to another selected input.

    Parameters
    ----------
    directories : sequence of Path
        Expanded input directories.

    Returns
    -------
    list of Path
        Inputs that are not another input's ``-panos`` sibling.

    Raises
    ------
    ValueError
        If every directory was a generated sibling.
    """
    destinations = {sibling_pano_dir(directory).resolve() for directory in directories}
    kept = [
        directory
        for directory in directories
        if directory.resolve() not in destinations
    ]
    if not kept:
        raise ValueError("No input directories found")
    return kept


def content_box(
    image: Image.Image, threshold: int = DEFAULT_CROP_THRESHOLD
) -> Optional[Tuple[int, int, int, int]]:
    """
    Bound the pixels that are not black canvas.

    A pixel counts as canvas when every channel is at or below
    ``threshold``. The box is Pillow's crop box: left and top are
    inclusive, right and bottom are exclusive. Black wedges in the
    corners stay when the picture already touches every edge.

    Parameters
    ----------
    image : Image.Image
        Stitched panorama.
    threshold : int
        Maximum channel value that still counts as black.

    Returns
    -------
    tuple of int or None
        ``(left, top, right, bottom)``. None when every pixel is black.
    """
    rgb = image.convert("RGB")
    red, green, blue = rgb.split()
    brightest = ImageChops.lighter(red, ImageChops.lighter(green, blue))
    mask = brightest.point(lambda value: 255 if int(value) > threshold else 0)
    return mask.getbbox()


def crop_black_canvas(
    source: Path,
    destination: Path,
    threshold: int = DEFAULT_CROP_THRESHOLD,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Write a JPEG of ``source`` with the black canvas removed.

    Parameters
    ----------
    source : Path
        Stitched panorama.
    destination : Path
        ``<pano_name>_cropped.jpg``.
    threshold : int
        Maximum channel value that still counts as black.

    Returns
    -------
    tuple of int or None
        The crop box. None when the frame is entirely black, in which
        case nothing is written.

    Raises
    ------
    OSError
        If the file cannot be read or written.
    UnidentifiedImageError
        If Pillow does not recognize ``source``.
    """
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        box = content_box(rgb, threshold)
        if box is None:
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb.crop(box).save(destination, "JPEG", quality=95)
    return box


def _stitch_rank(panos_dir: Path, path: Path) -> Tuple[int, int]:
    """
    Prefer a JPEG in the panorama directory over a TIFF inside a folder.

    Parameters
    ----------
    panos_dir : Path
        Directory that holds the panorama folders.
    path : Path
        One candidate stitch.

    Returns
    -------
    tuple of int
        Sort key. Lower is the file to crop.
    """
    suffix_rank = {".jpg": 0, ".jpeg": 1, ".tif": 2, ".tiff": 3}
    depth = 0 if path.parent == panos_dir else 1
    return (suffix_rank.get(path.suffix.lower(), 9), depth)


def stitched_panorama_images(panos_dir: Path) -> List[Path]:
    """
    Find one stitched image per panorama name.

    A loose ``guessed_pano*.jpg`` or ``known_pano*.jpg`` in ``panos_dir``
    wins over ``<name>/<name>.tif`` inside the folder. Files already
    named ``_cropped`` are not sources.

    Parameters
    ----------
    panos_dir : Path
        Sibling directory written by ``sb pano``.

    Returns
    -------
    list of Path
        Chosen stitch files, sorted by name.
    """
    if not panos_dir.is_dir():
        return []
    chosen: Dict[str, Path] = {}
    candidates: List[Path] = []
    for path in panos_dir.iterdir():
        if path.is_file() and _is_stitch_image(path):
            candidates.append(path)
            continue
        if not path.is_dir() or _PANO_STEM_RE.fullmatch(path.name) is None:
            continue
        for suffix in _STITCH_SUFFIXES:
            inner = path / f"{path.name}{suffix}"
            if inner.is_file():
                candidates.append(inner)
    for path in candidates:
        current = chosen.get(path.stem)
        if current is None or _stitch_rank(panos_dir, path) < _stitch_rank(
            panos_dir, current
        ):
            chosen[path.stem] = path
    return [chosen[stem] for stem in sorted(chosen)]


def _is_stitch_image(path: Path) -> bool:
    """
    Return whether ``path`` is a full stitched panorama, not a crop.

    Parameters
    ----------
    path : Path
        File in a panorama directory.

    Returns
    -------
    bool
        True for ``known_pano`` and ``guessed_pano`` images.
    """
    if path.suffix.lower() not in _STITCH_SUFFIXES:
        return False
    if path.stem.endswith("_cropped"):
        return False
    return _PANO_STEM_RE.fullmatch(path.stem) is not None


def crop_panorama_directory(
    panos_dir: Path,
    threshold: int = DEFAULT_CROP_THRESHOLD,
    progress: bool = True,
) -> List[Path]:
    """
    Write ``<pano_name>_cropped.jpg`` for each stitched panorama.

    The full image is left in place. The cropped file sits in
    ``panos_dir`` and drops the black canvas.

    Parameters
    ----------
    panos_dir : Path
        Sibling directory written by ``sb pano``.
    threshold : int
        Maximum channel value that still counts as black.
    progress : bool
        Print one line per cropped file.

    Returns
    -------
    list of Path
        Cropped JPEGs that were written.
    """
    written: List[Path] = []
    sources = stitched_panorama_images(panos_dir)
    if not sources:
        _say(f"No stitched panoramas to crop in {panos_dir}", progress)
        return written
    for source in sources:
        destination = panos_dir / f"{source.stem}_cropped.jpg"
        try:
            box = crop_black_canvas(source, destination, threshold)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            logger.warning(f"Could not crop {source}: {exc}")
            _say(f"  skip crop {source.name}: {exc}", progress)
            continue
        if box is None:
            _say(f"  skip crop {source.name}: frame is black", progress)
            continue
        width = box[2] - box[0]
        height = box[3] - box[1]
        _say(
            f"  cropped {source.name} -> {destination.name} ({width}x{height})",
            progress,
        )
        written.append(destination)
    return written


def find_action_panos(
    inputs: Sequence[str],
    config: PanoConfig,
    pattern: str = "*",
    survey: Optional[SurveyFn] = None,
) -> PanoResult:
    """
    Score each input on its own and write a sibling ``-panos`` folder.

    Five input directories produce five siblings. Panoramas from one
    input stay in that input's sibling. Numbering restarts in each
    sibling. A dry run writes nothing.

    Parameters
    ----------
    inputs : sequence of str
        Directories or globs, same shape as ``sb split``.
    config : PanoConfig
        Thresholds and output switches.
    pattern : str
        Glob passed to the photo collector.
    survey : callable, optional
        Replacement for Hugin. Tests pass this. Production leaves it empty
        and shells out to ``pto_gen`` and ``cpfind``.

    Returns
    -------
    PanoResult
        Frames, groups, and the sibling directories for each input.

    Raises
    ------
    ValueError
        If no input directories exist or the config is invalid.
    RuntimeError
        If EXIF cannot be read, or Hugin is required and missing.
    """
    validate_config(config)
    show = config.progress
    directories = _drop_generated_pano_dirs(expand_input_directories(inputs))
    _say("Scanning " + ", ".join(str(path) for path in directories), show)
    if survey is None:
        matcher: SurveyFn = _hugin_survey(config)
    else:
        matcher = survey
    if config.stitch and not config.dry_run:
        resolve_hugin_tool("hugin_executor")

    all_frames: List[Frame] = []
    all_groups: List[PanoGroup] = []
    skipped_exif = 0
    skipped_decode = 0
    output_dirs: List[Path] = []
    projects: List[Path] = []
    for directory in directories:
        destination = sibling_pano_dir(directory)
        output_dirs.append(destination)
        photos = collect_photo_paths(
            [directory], pattern=pattern, output_dir=destination
        )
        _say(
            f"Collected {len(photos)} photos from {directory.name} "
            f"(pattern {pattern!r})",
            show,
        )
        frames, skipped_here, undecodable_here = load_frames(
            photos,
            config.marker_sample,
            progress=show,
            marker_var=config.marker_var,
        )
        skipped_exif += skipped_here
        skipped_decode += undecodable_here
        groups, _runs = detect_panos(frames, config, matcher)
        all_frames.extend(frames)
        all_groups.extend(groups)
        if config.dry_run:
            continue
        projects.extend(
            write_pano_folders(
                groups,
                destination,
                copy_files=config.copy_files,
                write_pto=config.write_pto,
                progress=show,
                stitch=False,
            )
        )
    all_frames.sort(key=lambda frame: (frame.timestamp, frame.path.name))
    all_groups.sort(
        key=lambda group: (
            group.frames[0].timestamp if group.frames else datetime.min,
            group.frames[0].path.name if group.frames else "",
            group.kind,
        )
    )
    marker_runs = find_marker_runs(
        all_frames, config.marker_var, config.marker_count, config.discontinuity
    )
    cropped: List[Path] = []
    if config.dry_run:
        _say("Dry run: panorama folders will not be written", show)
    else:
        if config.stitch:
            stitch_projects(projects, progress=show)
        if config.crop:
            for directory in output_dirs:
                cropped.extend(crop_panorama_directory(directory, progress=show))
    return PanoResult(
        frames=all_frames,
        groups=all_groups,
        marker_runs=marker_runs,
        skipped_no_exif=skipped_exif,
        skipped_undecodable=skipped_decode,
        output_dirs=output_dirs,
        cropped=cropped,
    )


def load_frames(
    photos: Sequence[Path],
    sample: int,
    progress: bool = True,
    marker_var: Optional[float] = None,
) -> Tuple[List[Frame], int, int]:
    """
    Read capture times and grayscale variation for each photo.

    Parameters
    ----------
    photos : sequence of Path
        Candidate images.
    sample : int
        Thumbnail edge used for the mean and the variance.
    progress : bool
        Show a tqdm bar and one line per photo.
    marker_var : float, optional
        When set, lines at or below this variance are marked MARKER.

    Returns
    -------
    tuple
        Frames sorted by capture time, count skipped for missing EXIF,
        and count skipped because Pillow could not decode them.

    Raises
    ------
    RuntimeError
        If fast-exif-rs-py is not installed.
    """
    tag_maps = _read_exif_tag_maps(photos, progress=progress)
    frames: List[Frame] = []
    skipped_exif = 0
    skipped_decode = 0
    _say(f"Scoring grayscale variation on a {sample}px sample", progress)
    with _bar(len(photos), "Variation", progress) as bar:
        for photo, tags in zip(photos, tag_maps):
            bar.set_postfix_str(photo.name)
            timestamp = timestamp_from_exif_tags(tags)
            if timestamp is None:
                skipped_exif += 1
                _say(f"  skip {photo.name}: no EXIF capture time", progress)
                bar.update(1)
                continue
            try:
                variation, mean, width, height = measure_marker_variation(
                    photo, sample
                )
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                logger.warning(f"Skipping undecodable image {photo}: {exc}")
                skipped_decode += 1
                _say(f"  skip {photo.name}: cannot decode ({exc})", progress)
                bar.update(1)
                continue
            mark = ""
            if marker_var is not None and variation <= marker_var:
                mark = "  MARKER"
            _say(
                f"  {timestamp:%H:%M:%S}  {photo.name}  "
                f"mean={mean:.1f}  var={variation:.1f}  {width}x{height}{mark}",
                progress,
            )
            frames.append(
                Frame(
                    path=photo,
                    timestamp=timestamp,
                    variation=variation,
                    width=width,
                    height=height,
                    mean=mean,
                )
            )
            bar.update(1)
    frames.sort(key=lambda frame: (frame.timestamp, frame.path.name))
    _say(
        f"Scored {len(frames)} frames, skipped {skipped_exif} without EXIF, "
        f"{skipped_decode} undecodable",
        progress,
    )
    return frames, skipped_exif, skipped_decode


def write_pano_folders(
    groups: Sequence[PanoGroup],
    output_dir: Path,
    copy_files: bool,
    write_pto: bool,
    progress: bool = True,
    stitch: bool = False,
    optimize: bool = True,
) -> List[Path]:
    """
    Create one folder per accepted panorama.

    When a project is written and ``optimize`` is true, yaw, pitch, roll,
    and field of view are optimized with the median image as the position
    and exposure anchor. ``stitch`` runs ``hugin_executor`` on those
    projects after every folder in this call has been written. No
    batch window is opened.

    Parameters
    ----------
    groups : sequence of PanoGroup
        Detection output. Rejected groups are ignored. Accepted groups
        must already have ``number`` set.
    output_dir : Path
        Parent directory. Created if needed.
    copy_files : bool
        Copy when True, symlink when False.
    write_pto : bool
        Write ``{folder_name}.pto`` inside each folder.
    progress : bool
        Show a tqdm bar and one line per folder.
    stitch : bool
        Stitch the projects after the folders are written.
    optimize : bool
        Run the positions-and-view correction. Tests that inspect the
        unoptimized project pass False.

    Returns
    -------
    list of Path
        Hugin projects written into the panorama folders.

    Raises
    ------
    RuntimeError
        If a required Hugin tool is missing or fails.
    """
    if stitch and not write_pto:
        raise ValueError("--stitch requires the Hugin project (--pto)")
    accepted = [group for group in groups if group.accepted and group.number]
    if not accepted:
        output_dir.mkdir(parents=True, exist_ok=True)
        _say(f"No panorama folders to write under {output_dir}", progress)
        return []
    if write_pto and optimize:
        resolve_hugin_tool("pto_var")
        resolve_hugin_tool("autooptimiser")
    if stitch:
        resolve_hugin_tool("hugin_executor")
    total = len(accepted)
    output_dir.mkdir(parents=True, exist_ok=True)
    projects: List[Path] = []
    kind = "copies" if copy_files else "symlinks"
    _say(f"Writing {total} panorama folders to {output_dir} ({kind})", progress)
    with _bar(total, "Folders", progress, unit="pano") as bar:
        for group in accepted:
            assert group.number is not None
            folder_name = format_pano_folder(
                group.kind,
                group.number,
                total,
                group.frames[0].timestamp,
                group.frames[-1].timestamp,
            )
            bar.set_postfix_str(folder_name)
            folder = output_dir / folder_name
            folder.mkdir(exist_ok=True)
            names: List[str] = []
            used: Dict[str, bool] = {}
            for frame in group.frames:
                name = _unique_leaf_name(frame.path.name, used)
                destination = folder / name
                _place_photo(frame.path, destination, copy_files)
                names.append(name)
                verb = "copy" if copy_files else "symlink"
                _say(f"    {verb} {name}", progress)
            group.folder = folder
            if write_pto:
                document = render_pto(names, group.frames, group.points)
                pto_path = folder / f"{folder_name}.pto"
                pto_path.write_text(document, encoding="utf-8")
                anchor = anchor_index(len(names))
                _say(
                    f"  anchor {names[anchor]} "
                    f"({anchor + 1} of {len(names)}, position and exposure)",
                    progress,
                )
                if optimize:
                    correct_positions_and_view(
                        pto_path,
                        anchor,
                        progress=progress,
                    )
                projects.append(pto_path)
                _say(
                    f"  {folder_name}  {len(names)} {kind}, "
                    f"{len(group.points)} control points -> {pto_path.name}",
                    progress,
                )
            else:
                _say(f"  {folder_name}  {len(names)} {kind}", progress)
            bar.update(1)
    if stitch:
        stitch_projects(projects, progress=progress)
    return projects


def format_pano_folder(
    kind: str,
    number: int,
    total: int,
    start: datetime,
    end: datetime,
) -> str:
    """
    Build a panorama folder name.

    Parameters
    ----------
    kind : str
        ``known`` or ``guessed``.
    number : int
        1-based index among accepted panoramas.
    total : int
        How many accepted panoramas share this numbering.
    start : datetime
        Capture time of the first frame.
    end : datetime
        Capture time of the last frame.

    Returns
    -------
    str
        ``known_pano01_20Sep2025_090012-090018`` style name.
    """
    prefix = "known_pano" if kind == "known" else "guessed_pano"
    label = format_game_id(number, total)
    date_str = start.strftime("%d%b%Y")
    start_clock = start.strftime("%H%M%S")
    end_clock = end.strftime("%H%M%S")
    return f"{prefix}{label}_{date_str}_{start_clock}-{end_clock}"


def _close_marker_run(
    runs: List[Tuple[int, int]], start: int, end: int, marker_count: int
) -> None:
    """
    Append a uniform run when it is long enough.

    Parameters
    ----------
    runs : list
        Accumulator of inclusive ranges.
    start : int
        First marker index.
    end : int
        Last marker index.
    marker_count : int
        Minimum length.
    """
    if end - start + 1 >= marker_count:
        runs.append((start, end))


def _gap_seconds(older: Frame, newer: Frame) -> float:
    """
    Return seconds from ``older`` to ``newer``.

    Parameters
    ----------
    older : Frame
        Earlier photo.
    newer : Frame
        Later photo.

    Returns
    -------
    float
        Non-negative capture gap.
    """
    return (newer.timestamp - older.timestamp).total_seconds()


def _count_points(points: Sequence[ControlPoint]) -> Dict[Tuple[int, int], int]:
    """
    Tally control points per image pair.

    Parameters
    ----------
    points : sequence of ControlPoint
        Points for one survey.

    Returns
    -------
    dict
        Counts keyed by ``(left_index, right_index)``.
    """
    counts: Dict[Tuple[int, int], int] = {}
    for point in points:
        key = (point.left_index, point.right_index)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _align_preview_points(
    points: Sequence[ControlPoint],
    image_paths: Sequence[str],
    previews: Sequence[Path],
) -> List[ControlPoint]:
    """
    Renumber control points onto the preview order we passed to ``pto_gen``.

    Parameters
    ----------
    points : sequence of ControlPoint
        Points indexed into the ``.pto`` image list.
    image_paths : sequence of str
        Paths from that image list.
    previews : sequence of Path
        Previews in frame order.

    Returns
    -------
    list of ControlPoint
        Points indexed into ``previews``. The original points are returned
        when the project image list does not match those previews.
    """
    if len(image_paths) != len(previews):
        return list(points)
    resolved = {preview.resolve(): index for index, preview in enumerate(previews)}
    as_text = {str(preview): index for index, preview in enumerate(previews)}
    pto_to_frame: List[int] = []
    for raw in image_paths:
        path = Path(raw)
        if path.resolve() in resolved:
            pto_to_frame.append(resolved[path.resolve()])
        elif raw in as_text:
            pto_to_frame.append(as_text[raw])
        else:
            return list(points)
    remapped: List[ControlPoint] = []
    for point in points:
        if point.left_index >= len(pto_to_frame) or point.right_index >= len(
            pto_to_frame
        ):
            continue
        left = pto_to_frame[point.left_index]
        right = pto_to_frame[point.right_index]
        x_left, y_left = point.x, point.y
        x_right, y_right = point.x_right, point.y_right
        if left == right:
            continue
        if left > right:
            left, right = right, left
            x_left, x_right = x_right, x_left
            y_left, y_right = y_right, y_left
        remapped.append(
            ControlPoint(
                left_index=left,
                right_index=right,
                x=x_left,
                y=y_left,
                x_right=x_right,
                y_right=y_right,
            )
        )
    return remapped


def _scale_points(
    points: Sequence[ControlPoint],
    scales: Sequence[Tuple[float, float]],
    frame_count: int,
) -> List[ControlPoint]:
    """
    Multiply preview coordinates onto full-resolution frames.

    Parameters
    ----------
    points : sequence of ControlPoint
        Points in preview pixels.
    scales : sequence of tuple
        ``(scale_x, scale_y)`` per frame. Multiply a preview coordinate
        by the matching scale.
    frame_count : int
        Expected number of frames. Points that refer outside it are dropped.

    Returns
    -------
    list of ControlPoint
        Scaled points.
    """
    scaled: List[ControlPoint] = []
    for point in points:
        if not (
            0 <= point.left_index < frame_count and 0 <= point.right_index < frame_count
        ):
            continue
        left_x, left_y = scales[point.left_index]
        right_x, right_y = scales[point.right_index]
        scaled.append(
            ControlPoint(
                left_index=point.left_index,
                right_index=point.right_index,
                x=point.x * left_x,
                y=point.y * left_y,
                x_right=point.x_right * right_x,
                y_right=point.y_right * right_y,
            )
        )
    return scaled


def _write_preview(source: Path, dest: Path, edge: int) -> Tuple[int, int, int, int]:
    """
    Write a JPEG preview and report both sizes.

    Parameters
    ----------
    source : Path
        Original image.
    dest : Path
        Preview destination.
    edge : int
        Maximum long-edge length. Smaller originals are copied at full size.

    Returns
    -------
    tuple of int
        Original width, original height, preview width, preview height.

    Raises
    ------
    RuntimeError
        If Pillow cannot decode ``source``.
    """
    try:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            orig_w, orig_h = rgb.size
            long_edge = max(orig_w, orig_h)
            if long_edge > edge:
                scale = edge / float(long_edge)
                preview_w = max(1, int(round(orig_w * scale)))
                preview_h = max(1, int(round(orig_h * scale)))
                rgb = rgb.resize((preview_w, preview_h), Image.Resampling.LANCZOS)
            else:
                preview_w, preview_h = orig_w, orig_h
            rgb.save(dest, "JPEG", quality=90)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise RuntimeError(f"Could not preview {source}: {exc}") from exc
    return orig_w, orig_h, preview_w, preview_h


def _run_hugin(command: Sequence[str], cwd: Optional[Path] = None) -> str:
    """
    Run a Hugin tool and raise on failure.

    Parameters
    ----------
    command : sequence of str
        Argv, including the executable.
    cwd : Path, optional
        Working directory. Image names in a project are basenames, so
        optimization and stitching run inside the panorama folder.

    Returns
    -------
    str
        Combined standard output and standard error.

    Raises
    ------
    RuntimeError
        If the process cannot start or returns non-zero.
    """
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            cwd=None if cwd is None else str(cwd),
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to run {command[0]}: {exc}") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        detail = output.strip()
        raise RuntimeError(
            f"{Path(command[0]).name} failed ({completed.returncode}): {detail}"
        )
    return output


def _echo_tool(output: str, progress: bool) -> None:
    """
    Repeat a Hugin tool's log under the progress bar.

    Parameters
    ----------
    output : str
        Combined stdout and stderr.
    progress : bool
        When False, the log is dropped.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            _say(f"    {stripped}", progress)


def correct_positions_and_view(
    pto_path: Path,
    anchor: int,
    progress: bool = True,
) -> None:
    """
    Optimize yaw, pitch, roll, and field of view, and nothing else.

    The anchor image keeps yaw, pitch, and roll at zero. Its field of
    view is still optimized. The same image is the exposure anchor, so
    Hugin marks it AC. Exposure, white balance, vignetting, and
    distortion are not variables in this pass. ``pto_var`` and
    ``autooptimiser`` are taken from ``PATH``.

    Parameters
    ----------
    pto_path : Path
        Project inside the panorama folder. Replaced with the optimized
        project.
    anchor : int
        Zero-based median image.
    progress : bool
        Print each tool's log.

    Raises
    ------
    RuntimeError
        If either tool is missing from ``PATH`` or fails.
    """
    pto_var_bin = resolve_hugin_tool("pto_var")
    autooptimiser_bin = resolve_hugin_tool("autooptimiser")
    pto_path = pto_path.resolve()
    folder = pto_path.parent
    marked = pto_path.with_name(f"{pto_path.stem}.marked.pto")
    _say(
        f"  pto_var --anchor={anchor} --color-anchor={anchor} "
        f"--opt={POSITION_VIEW_VARIABLES}",
        progress,
    )
    try:
        marked_log = _run_hugin(
            [
                pto_var_bin,
                f"--anchor={anchor}",
                f"--color-anchor={anchor}",
                f"--opt={POSITION_VIEW_VARIABLES}",
                "-o",
                str(marked),
                str(pto_path),
            ],
            cwd=folder,
        )
        _echo_tool(marked_log, progress)
        _say("  autooptimiser -n (positions and view: y, p, r, v)", progress)
        optimized_log = _run_hugin(
            [autooptimiser_bin, "-n", "-o", str(pto_path), str(marked)],
            cwd=folder,
        )
        _echo_tool(optimized_log, progress)
    finally:
        marked.unlink(missing_ok=True)


def stitch_projects(
    projects: Sequence[Path],
    progress: bool = True,
) -> List[Path]:
    """
    Stitch each project in process with ``hugin_executor``.

    Projects run one after another. The output prefix is the project
    path without ``.pto``, in that panorama folder. No batch window is
    opened, so the stitched image exists before cropping starts.

    Parameters
    ----------
    projects : sequence of Path
        Optimized ``.pto`` files, in stitch order.
    progress : bool
        Print each stitch.

    Returns
    -------
    list of Path
        Stitched images, one per project.

    Raises
    ------
    RuntimeError
        If ``hugin_executor`` is not on ``PATH``, fails, or writes no image.
    """
    written: List[Path] = []
    if not projects:
        _say("No projects to stitch", progress)
        return written
    _say(f"Stitching {len(projects)} project(s) with hugin_executor", progress)
    with _bar(len(projects), "Stitching", progress, unit="pano") as bar:
        for project in projects:
            image = _stitch_one_project(project, progress)
            written.append(image)
            bar.update(1)
    return written


def _stitch_one_project(project: Path, progress: bool) -> Path:
    """
    Run ``hugin_executor --stitching`` on one project.

    Parameters
    ----------
    project : Path
        Optimized ``.pto`` file.
    progress : bool
        Print the command and the tool log.

    Returns
    -------
    Path
        The stitched image beside the project.

    Raises
    ------
    RuntimeError
        If the tool is missing, fails, or writes no image.
    """
    project = project.resolve()
    prefix = project.with_suffix("")
    folder = project.parent
    _say(f"  hugin_executor --stitching {project.name}", progress)
    log = _run_hugin(
        [
            resolve_hugin_tool("hugin_executor"),
            "--stitching",
            f"--prefix={prefix}",
            str(project),
        ],
        cwd=folder,
    )
    _echo_tool(log, progress)
    return _stitched_image(prefix)


def _stitched_image(prefix: Path) -> Path:
    """
    Return the image ``hugin_executor`` wrote for ``prefix``.

    Parameters
    ----------
    prefix : Path
        Project path without the ``.pto`` suffix.

    Returns
    -------
    Path
        ``prefix`` plus a stitch suffix.

    Raises
    ------
    RuntimeError
        If none of the expected images exist.
    """
    for suffix in _STITCH_SUFFIXES:
        image = prefix.with_suffix(suffix)
        if image.is_file():
            return image
    raise RuntimeError(
        "hugin_executor finished but did not write "
        + " or ".join(f"{prefix.name}{suffix}" for suffix in _STITCH_SUFFIXES)
    )


def _say(message: str, enabled: bool) -> None:
    """
    Print one progress line without breaking an active tqdm bar.

    Parameters
    ----------
    message : str
        Text to show.
    enabled : bool
        When False, the line is dropped.
    """
    if enabled:
        tqdm.write(message, file=sys.stderr)


def _bar(total: int, desc: str, enabled: bool, unit: str = "photo") -> Any:
    """
    Open a tqdm bar on stderr.

    Parameters
    ----------
    total : int
        Number of steps.
    desc : str
        Bar label.
    enabled : bool
        When False, the bar is disabled.
    unit : str
        Unit shown after the count.

    Returns
    -------
    tqdm
        Progress bar. ``update`` is a no-op when disabled.
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        disable=not enabled,
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )


def _span_names(frames: Sequence[Frame]) -> str:
    """
    Return a short ``first .. last`` label for a run of frames.

    Parameters
    ----------
    frames : sequence of Frame
        Photos in order.

    Returns
    -------
    str
        One name, ``first .. last``, or ``(none)``.
    """
    if not frames:
        return "(none)"
    if len(frames) == 1:
        return frames[0].path.name
    return f"{frames[0].path.name} .. {frames[-1].path.name}"


def _announce_group(group: PanoGroup, enabled: bool) -> None:
    """
    Print a panorama decision and its control-point links.

    Parameters
    ----------
    group : PanoGroup
        Accepted or rejected run.
    enabled : bool
        When False, nothing is printed.
    """
    status = "keep" if group.accepted else "skip"
    _say(f"  {status} {group.kind}: {_span_names(group.frames)}", enabled)
    _say(f"    {group.reason}", enabled)
    if group.cut_link is not None:
        link = group.cut_link
        _say(
            f"    cut {link.older} --{link.control_points}cp "
            f"{link.gap_seconds:.1f}s-- {link.newer}",
            enabled,
        )
    for link in group.neighbor_links:
        _say(
            f"    {link.older} --{link.control_points}cp "
            f"{link.gap_seconds:.1f}s-- {link.newer}",
            enabled,
        )
    for link in group.stride_links:
        _say(
            f"    stride {link.older} --{link.control_points}cp "
            f"{link.gap_seconds:.1f}s-- {link.newer}",
            enabled,
        )


def _hugin_survey(config: PanoConfig) -> SurveyFn:
    """
    Build a survey callable that runs ``pto_gen`` and ``cpfind`` from ``PATH``.

    Parameters
    ----------
    config : PanoConfig
        Supplies ``cp_edge`` and whether to print progress.

    Returns
    -------
    SurveyFn
        Closure over that config.
    """

    def survey(frames: Sequence[Frame], linearmatchlen: int) -> PairSurvey:
        return survey_control_points(
            frames,
            linearmatchlen,
            config.cp_edge,
            progress=config.progress,
        )

    return survey


def _read_exif_tag_maps(
    photo_paths: Sequence[Path], progress: bool = True
) -> List[Dict[str, str]]:
    """
    Read EXIF tag maps with fast-exif-rs-py.

    Parallel read is preferred. If that call fails as a batch, each file
    is read individually so one corrupt image does not drop the rest.

    Parameters
    ----------
    photo_paths : sequence of Path
        Image files.
    progress : bool
        Print when the parallel read falls back to one file at a time.

    Returns
    -------
    list of dict
        One string-to-string tag map per input path, same order.

    Raises
    ------
    RuntimeError
        If fast-exif-rs-py is not installed or the result length mismatches.
    """
    try:
        import fast_exif_rs_py
    except ImportError as exc:
        raise RuntimeError(
            "fast-exif-rs-py is required to read EXIF capture times. "
            "Install project dependencies (Rust is required to build the "
            "extension)."
        ) from exc

    path_strs = [str(photo_path) for photo_path in photo_paths]
    _say(f"Reading EXIF for {len(path_strs)} photos", progress)
    try:
        tag_maps = cast(
            List[Dict[str, str]],
            fast_exif_rs_py.read_exif_files_parallel(path_strs),
        )
    except RuntimeError as exc:
        logger.warning(f"Parallel EXIF read failed ({exc}); reading files individually")
        _say(f"Parallel EXIF read failed ({exc}); reading one file at a time", progress)
        reader = fast_exif_rs_py.PyFastExifReader()
        tag_maps = []
        with _bar(len(path_strs), "EXIF", progress) as bar:
            for path_str in path_strs:
                bar.set_postfix_str(Path(path_str).name)
                try:
                    tag_maps.append(cast(Dict[str, str], reader.read_file(path_str)))
                except RuntimeError as file_exc:
                    logger.debug(f"EXIF read failed for {path_str}: {file_exc}")
                    _say(f"  EXIF failed {Path(path_str).name}: {file_exc}", progress)
                    tag_maps.append({})
                bar.update(1)

    if len(tag_maps) != len(photo_paths):
        raise RuntimeError(
            "fast-exif-rs-py returned a different number of EXIF maps than input files"
        )
    return tag_maps


def _known_group_from_window(
    frames: Sequence[Frame],
    window: Sequence[int],
    marker: Sequence[Frame],
    surveyed: PairSurvey,
    config: PanoConfig,
) -> PanoGroup:
    """
    Cut one known panorama out of the window before a uniform marker.

    Parameters
    ----------
    frames : sequence of Frame
        Full capture sequence.
    window : sequence of int
        Indexes before the marker.
    marker : sequence of Frame
        The uniform frames that ended this panorama.
    surveyed : PairSurvey
        Control points for ``window``, already computed.
    config : PanoConfig
        Thresholds.

    Returns
    -------
    PanoGroup
        Accepted when the cut still has ``min_frames`` photos.
    """
    pair_points = [
        surveyed.counts.get((index, index + 1), 0) for index in range(len(window) - 1)
    ]
    pair_gaps = [
        _gap_seconds(frames[window[index]], frames[window[index + 1]])
        for index in range(len(window) - 1)
    ]
    local_start = cut_known_start(
        pair_points, pair_gaps, config.split_points, config.boundary
    )
    kept_frames = [frames[index] for index in window[local_start:]]
    kept_survey = _slice_survey(surveyed, local_start, len(window))
    cut_link: Optional[LinkReport] = None
    if local_start > 0:
        older = frames[window[local_start - 1]]
        newer = frames[window[local_start]]
        cut_link = LinkReport(
            older=older.path.name,
            newer=newer.path.name,
            control_points=surveyed.counts.get((local_start - 1, local_start), 0),
            gap_seconds=_gap_seconds(older, newer),
        )
    accepted = len(kept_frames) >= config.min_frames
    if cut_link is None:
        reason = "uniform marker; control points stay above the split threshold"
    elif config.boundary == "first":
        reason = (
            "uniform marker; cut at the first weak link "
            f"({cut_link.gap_seconds:.1f}s, {cut_link.control_points} control points)"
        )
    else:
        reason = (
            "uniform marker; cut at the largest weak gap "
            f"({cut_link.gap_seconds:.1f}s, {cut_link.control_points} control points)"
        )
    if not accepted:
        reason = (
            f"only {len(kept_frames)} frame(s) before the uniform marker; "
            f"need {config.min_frames}. {reason}"
        )
    return PanoGroup(
        kind="known",
        accepted=accepted,
        reason=reason,
        frames=kept_frames,
        marker=list(marker),
        neighbor_links=_neighbor_links(kept_frames, kept_survey.counts),
        stride_links=_stride_links(kept_frames, kept_survey.counts, config.stride),
        cut_link=cut_link,
        points=_reindex_points(kept_survey.points, list(range(len(kept_frames)))),
    )


def _local_cut_index(
    frames: Sequence[Frame], window: Sequence[int], group: PanoGroup
) -> int:
    """
    Return how many window frames were dropped before a known panorama.

    Parameters
    ----------
    frames : sequence of Frame
        Full sequence.
    window : sequence of int
        Window indexes.
    group : PanoGroup
        Known group cut from that window.

    Returns
    -------
    int
        Local start. Zero keeps the whole window.
    """
    if not group.frames:
        return len(window)
    first = group.frames[0].path
    for local, index in enumerate(window):
        if frames[index].path == first:
            return local
    return 0


def _slice_survey(surveyed: PairSurvey, start: int, end: int) -> PairSurvey:
    """
    Restrict a survey to a half-open local range and renumber indexes.

    Parameters
    ----------
    surveyed : PairSurvey
        Survey of a larger window.
    start : int
        First local index to keep.
    end : int
        One past the last local index to keep.

    Returns
    -------
    PairSurvey
        Counts and points whose indexes start at 0.
    """
    mapping = {old: new for new, old in enumerate(range(start, end))}
    points: List[ControlPoint] = []
    for point in surveyed.points:
        if point.left_index in mapping and point.right_index in mapping:
            points.append(
                ControlPoint(
                    left_index=mapping[point.left_index],
                    right_index=mapping[point.right_index],
                    x=point.x,
                    y=point.y,
                    x_right=point.x_right,
                    y_right=point.y_right,
                )
            )
    counts: Dict[Tuple[int, int], int] = {}
    for (left, right), total in surveyed.counts.items():
        if left in mapping and right in mapping:
            counts[(mapping[left], mapping[right])] = total
    if not counts and points:
        counts = _count_points(points)
    return PairSurvey(counts=counts, points=points)


def _groups_from_ranges(
    frames: Sequence[Frame],
    indexes: Sequence[int],
    surveyed: PairSurvey,
    config: PanoConfig,
) -> List[PanoGroup]:
    """
    Turn classified local ranges into groups.

    Parameters
    ----------
    frames : sequence of Frame
        Full sequence.
    indexes : sequence of int
        Burst or prefix indexes, in order.
    surveyed : PairSurvey
        Control points indexed into ``indexes``.
    config : PanoConfig
        Thresholds.

    Returns
    -------
    list of PanoGroup
        Guessed, held, and open runs.
    """
    ranges = classify_burst_ranges(
        len(indexes),
        surveyed.counts,
        config.stride,
        config.overlap_points,
        config.far_points,
        config.min_frames,
    )
    groups: List[PanoGroup] = []
    for kind, start, end in ranges:
        local = list(range(start, end + 1))
        chosen = [frames[indexes[offset]] for offset in local]
        piece = _slice_survey(surveyed, start, end + 1)
        if kind == "guessed":
            reason = (
                f"neighbors have at least {config.overlap_points} control points "
                f"and frames {config.stride} apart have at most {config.far_points}"
            )
            accepted = True
        elif kind == "held":
            reason = (
                f"neighbors overlap, and frames {config.stride} apart still overlap"
            )
            accepted = False
        else:
            reason = "not enough neighbor overlap to stitch"
            accepted = False
        groups.append(
            PanoGroup(
                kind=kind,
                accepted=accepted,
                reason=reason,
                frames=chosen,
                neighbor_links=_neighbor_links(chosen, piece.counts),
                stride_links=_stride_links(chosen, piece.counts, config.stride),
                points=(
                    _reindex_points(piece.points, list(range(len(chosen))))
                    if accepted
                    else []
                ),
            )
        )
    return groups


def _reindex_points(
    points: Sequence[ControlPoint], kept: Sequence[int]
) -> List[ControlPoint]:
    """
    Renumber control points onto a compacted frame list.

    Parameters
    ----------
    points : sequence of ControlPoint
        Points already in the destination index space.
    kept : sequence of int
        Identity map ``0 .. n-1`` when ``points`` is already local.

    Returns
    -------
    list of ControlPoint
        Points whose indexes fall inside ``kept``.
    """
    allowed = set(kept)
    return [
        point
        for point in points
        if point.left_index in allowed and point.right_index in allowed
    ]


def _neighbor_links(
    frames: Sequence[Frame], counts: Mapping[Tuple[int, int], int]
) -> List[LinkReport]:
    """
    Describe every consecutive pair.

    Parameters
    ----------
    frames : sequence of Frame
        Run in order.
    counts : mapping
        Control-point totals.

    Returns
    -------
    list of LinkReport
        One report per consecutive pair.
    """
    links: List[LinkReport] = []
    for index in range(len(frames) - 1):
        links.append(
            LinkReport(
                older=frames[index].path.name,
                newer=frames[index + 1].path.name,
                control_points=counts.get((index, index + 1), 0),
                gap_seconds=_gap_seconds(frames[index], frames[index + 1]),
            )
        )
    return links


def _stride_links(
    frames: Sequence[Frame],
    counts: Mapping[Tuple[int, int], int],
    stride: int,
) -> List[LinkReport]:
    """
    Describe every pair ``stride`` frames apart.

    Parameters
    ----------
    frames : sequence of Frame
        Run in order.
    counts : mapping
        Control-point totals.
    stride : int
        Frame distance.

    Returns
    -------
    list of LinkReport
        One report per stride pair. Empty when the run is shorter than
        ``stride + 1``.
    """
    links: List[LinkReport] = []
    if stride < 1:
        return links
    for index in range(len(frames) - stride):
        links.append(
            LinkReport(
                older=frames[index].path.name,
                newer=frames[index + stride].path.name,
                control_points=counts.get((index, index + stride), 0),
                gap_seconds=_gap_seconds(frames[index], frames[index + stride]),
            )
        )
    return links


def _remaining_bursts(
    frames: Sequence[Frame],
    claimed: Mapping[int, bool],
    config: PanoConfig,
) -> List[List[int]]:
    """
    Group unclaimed, non-marker frames into time bursts.

    Parameters
    ----------
    frames : sequence of Frame
        Full sequence.
    claimed : mapping
        Indexes already taken by a marker or a known window.
    config : PanoConfig
        Supplies the variance threshold and the discontinuity gap.

    Returns
    -------
    list of list of int
        Bursts whose consecutive gaps are within ``discontinuity``.
    """
    bursts: List[List[int]] = []
    current: List[int] = []
    for index, frame in enumerate(frames):
        if index in claimed or frame.variation <= config.marker_var:
            if current:
                bursts.append(current)
                current = []
            continue
        if current:
            gap = _gap_seconds(frames[current[-1]], frame)
            if gap > config.discontinuity:
                bursts.append(current)
                current = [index]
                continue
        current.append(index)
    if current:
        bursts.append(current)
    return bursts


def _assign_numbers(groups: List[PanoGroup]) -> None:
    """
    Number accepted panoramas in capture order.

    Parameters
    ----------
    groups : list of PanoGroup
        Mutated in place. Rejected groups keep ``number`` unset.
    """
    accepted = [group for group in groups if group.accepted and group.frames]
    accepted.sort(
        key=lambda group: (group.frames[0].timestamp, group.frames[0].path.name)
    )
    for number, group in enumerate(accepted, start=1):
        group.number = number


def _unique_leaf_name(name: str, used: Dict[str, bool]) -> str:
    """
    Pick a basename that does not collide inside one folder.

    Parameters
    ----------
    name : str
        Preferred filename.
    used : dict
        Names already chosen. Updated with the returned name.

    Returns
    -------
    str
        Unique filename.
    """
    if name not in used:
        used[name] = True
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    serial = 2
    while True:
        candidate = f"{stem}_{serial}{suffix}"
        if candidate not in used:
            used[candidate] = True
            return candidate
        serial += 1


def _place_photo(source: Path, destination: Path, copy_files: bool) -> None:
    """
    Symlink or copy one photo into a panorama folder.

    Parameters
    ----------
    source : Path
        Original image.
    destination : Path
        Link or copy path. Replaced when it already exists.
    copy_files : bool
        Copy when True, symlink when False.
    """
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
        return
    destination.symlink_to(source.resolve())
