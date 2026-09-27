"""
Game Detection Module

Automated game boundary detection based on image EXIF capture times with
support for manual splits and comprehensive game session management.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, Any
from loguru import logger

_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".hif",
    ".cr2",
    ".nef",
    ".arw",
    ".dng",
    ".raf",
    ".orf",
    ".rw2",
}

# Camera capture time first; filesystem mtime is never used.
_EXIF_DATETIME_TAGS = (
    "DateTimeOriginal",
    "CreateDate",
    "DateTimeDigitized",
    "DateTime",
    "ModifyDate",
)

_EXIF_DATETIME_FORMATS = (
    "%Y:%m:%d %H:%M:%S",
    "%Y:%m:%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
)

_GLOB_METACHARACTERS = "*?["

DEFAULT_MIN_PHOTOS = 10
DEFAULT_MIN_PHOTOS_PER_HOUR = 100


def _has_glob_metacharacters(value: str) -> bool:
    """
    Return whether ``value`` looks like a glob pattern.

    Parameters
    ----------
    value : str
        Path or glob supplied on the command line.

    Returns
    -------
    bool
        True when ``*``, ``?``, or ``[`` is present.
    """
    return any(char in value for char in _GLOB_METACHARACTERS)


def expand_input_directories(
    inputs: Sequence[str],
    output_dir: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> List[Path]:
    """
    Expand user-supplied roots (including globs like ``*``) into directories.

    Files are ignored. The output directory is skipped so a default
    ``Games`` folder is not ingested when expanding ``*``.

    Parameters
    ----------
    inputs : sequence of str
        Directory paths or globs, e.g. ``("04_Apr", "05_May")`` or ``("*",)``.
    output_dir : Path, optional
        Destination for game folders; excluded from the result when it
        matches an expanded path.
    cwd : Path, optional
        Base directory for relative paths and globs. Defaults to the
        process working directory.

    Returns
    -------
    list of Path
        Existing directories, resolved, in a stable order.

    Raises
    ------
    ValueError
        If a non-glob path is missing, or nothing usable remains.
    """
    base = cwd if cwd is not None else Path.cwd()
    exclude: Optional[Path] = None
    if output_dir is not None:
        exclude = output_dir if output_dir.is_absolute() else (base / output_dir)
        exclude = exclude.resolve()

    discovered: List[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        text = str(raw)
        if _has_glob_metacharacters(text):
            matches = sorted(base.glob(text))
        else:
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = base / candidate
            matches = [candidate]

        if not matches and not _has_glob_metacharacters(text):
            raise ValueError(f"Input path does not exist: {text}")

        for match in matches:
            if not match.is_dir():
                continue
            resolved = match.resolve()
            if exclude is not None and resolved == exclude:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)

    if not discovered:
        raise ValueError("No input directories found")
    return discovered


def collect_photo_paths(
    directories: Sequence[Path],
    pattern: str = "*",
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Gather image files under one or more directories.

    Paths are de-duplicated by resolved location. Anything inside
    ``output_dir`` is omitted so a previous ``Games`` split is not
    ingested again.

    Parameters
    ----------
    directories : sequence of Path
        Roots to search.
    pattern : str
        Glob passed to ``Path.rglob`` (default ``*``).
    output_dir : Path, optional
        Game-folder destination to exclude.

    Returns
    -------
    list of Path
        Image files, unsorted.
    """
    exclude: Optional[Path] = None
    if output_dir is not None:
        exclude = output_dir.resolve()

    photos: List[Path] = []
    seen: set[Path] = set()

    for directory in directories:
        for photo_path in directory.rglob(pattern):
            if not photo_path.is_file():
                continue
            if photo_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            resolved = photo_path.resolve()
            if exclude is not None and (
                resolved == exclude or exclude in resolved.parents
            ):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            photos.append(photo_path)

    return photos


def _normalize_photo_directories(
    photo_directory: Union[Path, Sequence[Path]],
) -> List[Path]:
    """
    Coerce a single path or a sequence of paths into a directory list.

    Parameters
    ----------
    photo_directory : Path or sequence of Path
        One dump folder or several (e.g. month folders).

    Returns
    -------
    list of Path
        Directories as ``Path`` objects.

    Raises
    ------
    ValueError
        If the input is empty.
    """
    if isinstance(photo_directory, (str, Path)):
        directories = [Path(photo_directory)]
    else:
        directories = [Path(item) for item in photo_directory]
    if not directories:
        raise ValueError("No photo directories given")
    return directories


def _exif_tag_value(tags: Mapping[str, str], name: str) -> Optional[str]:
    """
    Return a tag value by ExifTool-style name.

    Parameters
    ----------
    tags : mapping
        EXIF tag map from fast-exif-rs-py.
    name : str
        Bare tag name such as ``DateTimeOriginal``.

    Returns
    -------
    str or None
        Tag value, or None if the tag is missing or empty.
    """
    direct = tags.get(name)
    if direct:
        return direct

    suffix = f":{name}"
    for key, value in tags.items():
        if key.endswith(suffix) and value:
            return value
    return None


def parse_exif_datetime(value: str) -> Optional[datetime]:
    """
    Parse a camera EXIF datetime string into a naive datetime.

    Parameters
    ----------
    value : str
        EXIF datetime, commonly ``YYYY:MM:DD HH:MM:SS``.

    Returns
    -------
    datetime or None
        Naive capture time, or None if the string cannot be parsed.
    """
    text = value.strip()
    if not text:
        return None

    # Drop a trailing timezone so all photos sort in camera-local time.
    if len(text) >= 6 and text[-6] in "+-" and text[-3] == ":":
        text = text[:-6].rstrip()
    elif len(text) >= 5 and text[-5] in "+-":
        text = text[:-5].rstrip()
    if text.endswith("Z"):
        text = text[:-1]

    for fmt in _EXIF_DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def _subseconds_to_microseconds(value: str) -> int:
    """
    Convert an EXIF SubSecTime value to microseconds.

    Parameters
    ----------
    value : str
        Subsecond digits, e.g. ``96`` or ``960``.

    Returns
    -------
    int
        Microseconds in ``[0, 999999]``.
    """
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return 0
    padded = (digits + "000000")[:6]
    return int(padded)


def timestamp_from_exif_tags(tags: Mapping[str, str]) -> Optional[datetime]:
    """
    Pick a capture time from an EXIF tag map.

    Prefers ``DateTimeOriginal``, then digitized/create times. Subsecond
    tags are applied when present.

    Parameters
    ----------
    tags : mapping
        EXIF tag map from fast-exif-rs-py.

    Returns
    -------
    datetime or None
        Naive capture time, or None if no usable datetime tag exists.
    """
    raw_dt: Optional[str] = None
    source_name: Optional[str] = None
    for name in _EXIF_DATETIME_TAGS:
        candidate = _exif_tag_value(tags, name)
        if candidate:
            raw_dt = candidate
            source_name = name
            break

    if raw_dt is None or source_name is None:
        return None

    parsed = parse_exif_datetime(raw_dt)
    if parsed is None:
        return None

    subsec_name = {
        "DateTimeOriginal": "SubSecTimeOriginal",
        "CreateDate": "SubSecTimeDigitized",
        "DateTimeDigitized": "SubSecTimeDigitized",
        "DateTime": "SubSecTime",
        "ModifyDate": "SubSecTime",
    }.get(source_name, "SubSecTime")
    subsec = _exif_tag_value(tags, subsec_name)
    if subsec and parsed.microsecond == 0:
        parsed = parsed.replace(microsecond=_subseconds_to_microseconds(subsec))
    return parsed


def photos_per_hour(photo_count: int, duration_minutes: float) -> float:
    """
    Convert a session's photo count into an hourly shooting rate.

    A zero-length span (every photo at the same instant) is treated as
    one second so the rate stays finite.

    Parameters
    ----------
    photo_count : int
        Number of photos in the session.
    duration_minutes : float
        Elapsed minutes from first to last capture.

    Returns
    -------
    float
        Photos per hour. Zero when ``photo_count`` is not positive.
    """
    if photo_count < 1:
        return 0.0
    minutes = duration_minutes if duration_minutes > 0.0 else (1.0 / 60.0)
    return float(photo_count) * 60.0 / minutes


def game_id_width(total_games: int) -> int:
    """
    Digit width for game numbers so lexical order matches numeric order.

    At least two digits (``01``) so ``Game10`` does not sort before
    ``Game2``. Wider when there are 100 or more games.

    Parameters
    ----------
    total_games : int
        How many games are in this split.

    Returns
    -------
    int
        Pad width, at least 2.
    """
    if total_games < 1:
        return 2
    return max(2, len(str(total_games)))


def format_game_id(game_id: int, total_games: int) -> str:
    """
    Format a game number with zero padding.

    Parameters
    ----------
    game_id : int
        1-based game index.
    total_games : int
        How many games are in this split.

    Returns
    -------
    str
        Zero-padded id, e.g. ``01`` or ``001``.
    """
    return f"{int(game_id):0{game_id_width(total_games)}d}"


_HISTOGRAM_BIN_MINUTES = (
    1,
    2,
    5,
    10,
    15,
    30,
    60,
    120,
    180,
    360,
    720,
    1440,
)


def histogram_bin_minutes(span_minutes: float, target_bins: int = 40) -> int:
    """
    Pick a histogram bin width for a capture span.

    Parameters
    ----------
    span_minutes : float
        Minutes from first to last photo.
    target_bins : int
        Approximate number of bins if every slot were filled.

    Returns
    -------
    int
        Bin width in minutes. At least 1, at most one day.
    """
    if span_minutes <= 0:
        return 1
    raw = span_minutes / float(max(target_bins, 1))
    for size in _HISTOGRAM_BIN_MINUTES:
        if raw <= size:
            return size
    return 1440


def floor_to_bin(timestamp: datetime, bin_minutes: int) -> datetime:
    """
    Floor a capture time to the start of its histogram bin.

    Parameters
    ----------
    timestamp : datetime
        Naive capture time.
    bin_minutes : int
        Bin width in minutes.

    Returns
    -------
    datetime
        Bin start, seconds and microseconds cleared.
    """
    width = max(int(bin_minutes), 1)
    if width >= 1440:
        return datetime(timestamp.year, timestamp.month, timestamp.day)
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    slot = (minutes_since_midnight // width) * width
    return datetime(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        slot // 60,
        slot % 60,
    )


def leftover_index_ranges(
    photo_count: int,
    boundaries: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Inclusive index ranges that are not part of any detected game.

    Parameters
    ----------
    photo_count : int
        Number of photos in capture-time order.
    boundaries : sequence of tuple of int
        Inclusive ``(start, end)`` game ranges.

    Returns
    -------
    list of tuple of int
        Inclusive leftover ranges, in order.
    """
    if photo_count < 1:
        return []
    covered = [False] * photo_count
    for start_idx, end_idx in boundaries:
        low = max(0, int(start_idx))
        high = min(photo_count - 1, int(end_idx))
        for index in range(low, high + 1):
            covered[index] = True

    ranges: List[Tuple[int, int]] = []
    index = 0
    while index < photo_count:
        if covered[index]:
            index += 1
            continue
        end_idx = index
        while end_idx + 1 < photo_count and not covered[end_idx + 1]:
            end_idx += 1
        ranges.append((index, end_idx))
        index = end_idx + 1
    return ranges


def leftover_reason(
    photo_count: int,
    duration_minutes: float,
    config: GameDetectionConfig,
) -> str:
    """
    Explain why a cluster was not kept as a game album.

    Parameters
    ----------
    photo_count : int
        Photos in the cluster.
    duration_minutes : float
        Elapsed minutes from first to last capture.
    config : GameDetectionConfig
        Thresholds used for the split.

    Returns
    -------
    str
        Short reason for the CLI.
    """
    if photo_count < 2:
        return "fewer than 2 photos"
    if (
        config.min_photos is not None
        and photo_count < int(config.min_photos)
    ):
        return f"{photo_count} photos is under {config.min_photos} photos"
    if duration_minutes < float(config.min_game_duration_minutes):
        return (
            f"{duration_minutes:.1f} min is under "
            f"{config.min_game_duration_minutes} min duration"
        )
    if config.min_photos_per_hour is not None:
        rate = photos_per_hour(photo_count, duration_minutes)
        if rate < float(config.min_photos_per_hour):
            return f"{rate:.0f}/h is under {config.min_photos_per_hour}/h"
    return "not grouped into a game"


def histogram_bar(count: int, max_count: int, width: int = 32) -> str:
    """
    Build a text bar for one histogram bin.

    Parameters
    ----------
    count : int
        Photos in the bin.
    max_count : int
        Largest bin count in the chart.
    width : int
        Maximum bar width in characters.

    Returns
    -------
    str
        A string of block characters, empty when ``count`` is 0.
    """
    if count <= 0 or max_count <= 0 or width <= 0:
        return ""
    filled = int(round(float(count) / float(max_count) * float(width)))
    if filled < 1:
        filled = 1
    if filled > width:
        filled = width
    return "█" * filled


_GAP_BIN_EDGES_SECONDS: Tuple[float, ...] = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    15.0,
    30.0,
    45.0,
    60.0,
    90.0,
    120.0,
    180.0,
    300.0,
    450.0,
    600.0,
    900.0,
    1200.0,
    1800.0,
    3600.0,
    7200.0,
    14400.0,
    float("inf"),
)


def intershot_gaps_seconds(timestamps: Sequence[datetime]) -> List[float]:
    """
    Seconds between consecutive photos in capture-time order.

    Parameters
    ----------
    timestamps : sequence of datetime
        Sorted capture times.

    Returns
    -------
    list of float
        One gap per adjacent pair. Empty when fewer than two photos.
    """
    gaps: List[float] = []
    for index in range(1, len(timestamps)):
        gaps.append((timestamps[index] - timestamps[index - 1]).total_seconds())
    return gaps


def format_fractional_minutes(seconds: float) -> str:
    """
    Format a duration as minutes for ``--min-gap``.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Minutes with up to four decimal places, trailing zeros stripped.
    """
    if seconds <= 0.0:
        return "0"
    text = f"{seconds / 60.0:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _format_gap_bin_seconds(seconds: float) -> str:
    """
    Short label for a histogram bin edge.

    Parameters
    ----------
    seconds : float
        Edge in seconds, or infinity.

    Returns
    -------
    str
        Compact duration label.
    """
    if seconds == float("inf"):
        return "∞"
    if seconds < 60.0:
        if seconds == int(seconds):
            return f"{int(seconds)}s"
        return f"{seconds:g}s"
    minutes = seconds / 60.0
    if minutes == int(minutes) and minutes < 60.0:
        return f"{int(minutes)}m"
    if minutes < 60.0:
        return f"{minutes:g}m"
    hours = minutes / 60.0
    if hours == int(hours):
        return f"{int(hours)}h"
    return f"{hours:g}h"


def build_gap_histogram(
    timestamps: Sequence[datetime],
    min_gap_minutes: float,
) -> List[Dict[str, Any]]:
    """
    Count consecutive-shot gaps by duration so ``--min-gap`` can be chosen.

    Each bin is labeled in seconds and as fractional minutes (the
    ``--min-gap`` value that would start splitting at that bin).

    Parameters
    ----------
    timestamps : sequence of datetime
        Sorted capture times.
    min_gap_minutes : float
        Current split threshold, used to mark keep vs split bins.

    Returns
    -------
    list of dict
        Histogram rows from the first occupied bin through the last.
    """
    gaps = intershot_gaps_seconds(timestamps)
    edges = _GAP_BIN_EDGES_SECONDS
    counts = [0] * (len(edges) - 1)
    for gap in gaps:
        placed = False
        for index in range(len(edges) - 1):
            low = edges[index]
            high = edges[index + 1]
            if gap >= low and gap < high:
                counts[index] += 1
                placed = True
                break
        if not placed and gaps:
            counts[-1] += 1

    first_used = next((i for i, count in enumerate(counts) if count > 0), None)
    last_used = next(
        (i for i, count in reversed(list(enumerate(counts))) if count > 0),
        None,
    )
    if first_used is None or last_used is None:
        return []

    threshold_seconds = float(min_gap_minutes) * 60.0
    max_count = max(counts[first_used : last_used + 1])
    rows: List[Dict[str, Any]] = []
    for index in range(first_used, last_used + 1):
        low = edges[index]
        high = edges[index + 1]
        count = counts[index]
        if high != float("inf") and high <= threshold_seconds:
            effect = "keep"
        elif low >= threshold_seconds:
            effect = "split"
        else:
            effect = "straddle"
        rows.append(
            {
                "low_seconds": low,
                "high_seconds": None if high == float("inf") else high,
                "label": (
                    f"{_format_gap_bin_seconds(low)}–"
                    f"{_format_gap_bin_seconds(high)}"
                ),
                "min_gap_minutes": format_fractional_minutes(low),
                "count": count,
                "bar": histogram_bar(count, max_count),
                "effect": effect,
            }
        )
    return rows


def build_activity_histogram(
    timestamps: Sequence[datetime],
    assignments: Sequence[Optional[int]],
    total_games: int,
    bin_minutes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Bin photos by capture time and label each bin with a game or unsorted.

    Empty bins are omitted so a season of Saturday shoots stays readable.

    Parameters
    ----------
    timestamps : sequence of datetime
        Capture times in the same order as ``assignments``.
    assignments : sequence of int or None
        Game id for each photo, or None if unsorted.
    total_games : int
        Used to zero-pad game labels.
    bin_minutes : int, optional
        Bin width. Chosen from the span when omitted.

    Returns
    -------
    list of dict
        Each item has ``start``, ``label``, ``count``, ``bar_count``,
        ``assignment``, and ``game_id``.
    """
    if not timestamps:
        return []
    if len(timestamps) != len(assignments):
        raise ValueError("timestamps and assignments must be the same length")

    first = timestamps[0]
    last = timestamps[-1]
    span_minutes = max((last - first).total_seconds() / 60.0, 0.0)
    width = (
        int(bin_minutes)
        if bin_minutes is not None and bin_minutes > 0
        else histogram_bin_minutes(span_minutes)
    )

    bins: Dict[datetime, Dict[str, Any]] = {}
    for timestamp, game_id in zip(timestamps, assignments):
        start = floor_to_bin(timestamp, width)
        bucket = bins.get(start)
        if bucket is None:
            game_counts: Dict[Optional[int], int] = {}
            bucket = {"start": start, "count": 0, "game_counts": game_counts}
            bins[start] = bucket
        bucket["count"] = int(bucket["count"]) + 1
        counts = bucket["game_counts"]
        counts[game_id] = int(counts.get(game_id, 0)) + 1

    rows: List[Dict[str, Any]] = []
    for start in sorted(bins):
        bucket = bins[start]
        game_counts = bucket["game_counts"]
        majority_id = max(game_counts, key=lambda key: int(game_counts[key]))
        mixed = len(game_counts) > 1
        if mixed:
            assignment = "mixed"
        elif majority_id is None:
            assignment = "unsorted"
        else:
            assignment = f"Game{format_game_id(int(majority_id), max(total_games, 1))}"
        if width >= 1440:
            label = start.strftime("%d %b %Y")
        else:
            label = start.strftime("%d %b %H:%M")
        rows.append(
            {
                "start": start,
                "label": label,
                "count": int(bucket["count"]),
                "game_id": majority_id,
                "assignment": assignment,
                "mixed": mixed,
                "bin_minutes": width,
            }
        )
    return rows


@dataclass
class GameDetectionConfig:
    """Configuration for game detection."""

    min_game_duration_minutes: int = 0
    min_gap_minutes: float = 10.0
    min_photos: Optional[int] = DEFAULT_MIN_PHOTOS
    min_photos_per_hour: Optional[int] = DEFAULT_MIN_PHOTOS_PER_HOUR


def resolve_session_floors(
    min_photos: int,
    min_rate: int,
    *,
    photos_explicit: bool,
    rate_explicit: bool,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Resolve mutually exclusive photo-count and photos-per-hour floors.

    When neither flag was passed, both defaults apply (10 photos and
    100/hour). Passing one flag disables the other. Passing both is an
    error so wrestling (absolute count) and field sports (hourly rate)
    stay distinct.

    Parameters
    ----------
    min_photos : int
        Absolute photo floor (CLI default 10).
    min_rate : int
        Photos-per-hour floor (CLI default 100).
    photos_explicit : bool
        True when the user passed ``--min-photos``.
    rate_explicit : bool
        True when the user passed ``--min-rate``.

    Returns
    -------
    tuple of (int or None, int or None)
        ``(min_photos, min_rate)``. ``None`` disables that floor.

    Raises
    ------
    ValueError
        If both flags were passed.
    """
    if photos_explicit and rate_explicit:
        raise ValueError("Use either --min-photos or --min-rate, not both")
    if photos_explicit:
        return min_photos, None
    if rate_explicit:
        return None, min_rate
    return min_photos, min_rate


def _optional_floor(value: object) -> Optional[int]:
    """
    Convert a count or rate floor, treating None and non-positive as off.

    Parameters
    ----------
    value : object
        Integer floor, or ``None`` to disable.

    Returns
    -------
    int or None
        Positive floor, or ``None`` when the check should be skipped.
    """
    if value is None:
        return None
    number = int(value)
    if number <= 0:
        return None
    return number


def session_meets_thresholds(
    photo_count: int,
    duration_minutes: float,
    config: GameDetectionConfig,
) -> bool:
    """
    Return whether a candidate session should be kept as a game.

    Defaults require at least 10 photos and 100 photos per hour. Either
    floor can be disabled (``None``) so a short wrestling match can use
    an absolute count while a soccer game uses rate only. At least two
    photos are always required so a span can be measured.

    Parameters
    ----------
    photo_count : int
        Photos in the candidate session.
    duration_minutes : float
        Elapsed minutes from first to last capture.
    config : GameDetectionConfig
        Duration, count, and rate thresholds.

    Returns
    -------
    bool
        True when the session should become a game album.
    """
    if photo_count < 2:
        return False
    if duration_minutes < float(config.min_game_duration_minutes):
        return False
    if config.min_photos is not None and photo_count < int(config.min_photos):
        return False
    if config.min_photos_per_hour is not None:
        rate = photos_per_hour(photo_count, duration_minutes)
        if rate < float(config.min_photos_per_hour):
            return False
    return True


@dataclass
class GameSession:
    """Represents a detected game session."""

    game_id: int
    start_time: datetime
    end_time: datetime
    photo_count: int
    photo_files: List[Path]
    gap_before: Optional[int] = None  # seconds
    gap_after: Optional[int] = None  # seconds


class GameDetector:
    """
    Game boundary detection based on EXIF capture times.

    This class reads DateTimeOriginal (and related capture tags) from each
    image via fast-exif-rs-py, then looks for gaps that mark game boundaries.
    """

    def __init__(
        self, config: Optional[GameDetectionConfig] = None, cache_enabled: bool = True
    ):
        """
        Initialize the GameDetector.

        Args:
            config: Optional configuration for game detection
            cache_enabled: Whether to enable result caching
        """
        self.config = config or GameDetectionConfig()
        self.cache_enabled = cache_enabled
        self.games: List[GameSession] = []
        self._photo_timestamps: Dict[str, datetime] = {}
        self.logger = logger.bind(component="game_detector")
        self.logger.info("Initialized GameDetector")

    def detect_games(
        self,
        photo_directory: Union[Path, Sequence[Path]],
        pattern: str = "*",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Detect game boundaries across one or more directories of photos.

        All matching images are pooled, sorted by EXIF capture time, and
        numbered as a single season of games.

        Args:
            photo_directory: Directory or sequence of directories
            pattern: File pattern to match
            **kwargs: Additional arguments for game detection. ``output_dir``
                excludes an existing Games folder from the scan.

        Returns:
            Dictionary containing game detection results
        """
        try:
            prepared = self._prepare_detection(photo_directory, pattern, kwargs)
            if not prepared["success"]:
                return {
                    "success": False,
                    "error": prepared.get("error", "Game detection failed"),
                }
            if not prepared["games"]:
                return {"error": "No games detected", "success": False}

            results = {
                "success": True,
                "games": self._format_games_for_output(prepared["games"]),
                "summary": {
                    "total_games": len(prepared["games"]),
                    "total_photos": sum(
                        game.photo_count for game in prepared["games"]
                    ),
                    "detected_dates": self._get_detected_dates(
                        prepared["photo_metadata"]
                    ),
                },
            }
            return results

        except Exception as e:
            self.logger.error(f"Game detection failed: {e}")
            return {"error": str(e), "success": False}

    def analyze_games(
        self,
        photo_directory: Union[Path, Sequence[Path]],
        pattern: str = "*",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Explain how photos would be split, including gaps and leftovers.

        Unlike ``detect_games``, an empty game list is still a successful
        analysis: every cluster is reported as unsorted.

        Parameters
        ----------
        photo_directory : Path or sequence of Path
            Directories to scan.
        pattern : str
            File glob passed to ``rglob``.
        **kwargs
            Same detection knobs as ``detect_games``, plus optional
            ``bin_minutes`` for the activity histogram.

        Returns
        -------
        dict
            Timeline, histogram bins, games, and unsorted clusters.
        """
        try:
            prepared = self._prepare_detection(photo_directory, pattern, kwargs)
            if not prepared["success"]:
                return {
                    "success": False,
                    "error": prepared.get("error", "Game analysis failed"),
                }

            photo_metadata: List[Dict[str, Any]] = prepared["photo_metadata"]
            games: List[GameSession] = prepared["games"]
            boundaries: List[Tuple[int, int]] = prepared["boundaries"]
            bin_minutes = kwargs.get("bin_minutes")
            parsed_bin: Optional[int] = (
                int(bin_minutes) if bin_minutes is not None else None
            )
            return self._build_analysis(
                photo_metadata, games, boundaries, bin_minutes=parsed_bin
            )
        except Exception as e:
            self.logger.error(f"Game analysis failed: {e}")
            return {"error": str(e), "success": False}

    def _prepare_detection(
        self,
        photo_directory: Union[Path, Sequence[Path]],
        pattern: str,
        kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Collect photos, read EXIF, and cluster into game boundaries.

        Parameters
        ----------
        photo_directory : Path or sequence of Path
            Directories to scan.
        pattern : str
            File glob.
        kwargs : mapping
            Detection knobs, including optional ``output_dir``.

        Returns
        -------
        dict
            ``success``, and on success ``photo_metadata``, ``boundaries``,
            and ``games`` (games may be empty).
        """
        directories = _normalize_photo_directories(photo_directory)
        self.logger.info(
            f"Detecting games in {len(directories)} director"
            f"{'y' if len(directories) == 1 else 'ies'} with pattern {pattern}"
        )
        self._apply_detect_kwargs(kwargs)

        output_dir: Optional[Path] = None
        if "output_dir" in kwargs and kwargs["output_dir"] is not None:
            output_dir = Path(kwargs["output_dir"])

        photo_paths = collect_photo_paths(
            directories, pattern=pattern, output_dir=output_dir
        )
        if not photo_paths:
            return {"success": False, "error": "No photos found"}

        self.logger.info(f"Found {len(photo_paths)} photos")
        photo_metadata = self._analyze_timestamps(photo_paths)
        if not photo_metadata:
            return {"success": False, "error": "No valid EXIF capture times found"}

        boundaries = self._detect_game_boundaries(photo_metadata)
        games = (
            self._create_game_sessions(photo_metadata, boundaries) if boundaries else []
        )
        self.games = games
        return {
            "success": True,
            "photo_metadata": photo_metadata,
            "boundaries": boundaries,
            "games": games,
        }

    def _build_analysis(
        self,
        photo_metadata: List[Dict[str, Any]],
        games: List[GameSession],
        boundaries: List[Tuple[int, int]],
        bin_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build histogram and leftover clusters from a finished detection.

        Parameters
        ----------
        photo_metadata : list of dict
            Sorted capture records.
        games : list of GameSession
            Detected albums.
        boundaries : list of tuple of int
            Inclusive index ranges for those albums.
        bin_minutes : int, optional
            Histogram bin width.

        Returns
        -------
        dict
            Analysis payload for the CLI.
        """
        total_games = max(len(games), 1)
        photo_count = len(photo_metadata)
        assigned: List[Optional[int]] = [None] * photo_count
        for game, (start_idx, end_idx) in zip(games, boundaries):
            for index in range(start_idx, end_idx + 1):
                assigned[index] = game.game_id

        leftover_clusters: List[Dict[str, Any]] = []
        leftover_photo_count = 0
        for start_idx, end_idx in leftover_index_ranges(photo_count, boundaries):
            cluster = photo_metadata[start_idx : end_idx + 1]
            start_time = cluster[0]["timestamp"]
            end_time = cluster[-1]["timestamp"]
            duration_minutes = (end_time - start_time).total_seconds() / 60.0
            count = len(cluster)
            leftover_photo_count += count
            leftover_clusters.append(
                {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "start_label": start_time.strftime("%d %b %Y %H:%M:%S"),
                    "end_label": end_time.strftime("%H:%M:%S"),
                    "duration_minutes": round(duration_minutes, 1),
                    "photo_count": count,
                    "photos_per_hour": round(
                        photos_per_hour(count, duration_minutes), 1
                    ),
                    "reason": leftover_reason(
                        count, duration_minutes, self.config
                    ),
                }
            )

        timestamps = [item["timestamp"] for item in photo_metadata]
        histogram = build_activity_histogram(
            timestamps, assigned, len(games), bin_minutes=bin_minutes
        )
        max_bin = max((int(row["count"]) for row in histogram), default=0)
        for row in histogram:
            row["bar"] = histogram_bar(int(row["count"]), max_bin)
        gap_histogram = build_gap_histogram(
            timestamps, float(self.config.min_gap_minutes)
        )

        timeline = self._build_timeline(games, leftover_clusters)
        first = photo_metadata[0]["timestamp"]
        last = photo_metadata[-1]["timestamp"]
        span_minutes = (last - first).total_seconds() / 60.0

        return {
            "success": True,
            "games": self._format_games_for_output(games),
            "unsorted": leftover_clusters,
            "histogram": histogram,
            "gap_histogram": gap_histogram,
            "timeline": timeline,
            "summary": {
                "total_games": len(games),
                "game_photos": sum(game.photo_count for game in games),
                "unsorted_photos": leftover_photo_count,
                "total_photos": photo_count,
                "span_minutes": round(span_minutes, 1),
                "detected_dates": self._get_detected_dates(photo_metadata),
                "bin_minutes": histogram[0]["bin_minutes"] if histogram else 1,
                "min_photos": self.config.min_photos,
                "min_photos_per_hour": self.config.min_photos_per_hour,
                "min_gap_minutes": self.config.min_gap_minutes,
                "min_game_duration_minutes": self.config.min_game_duration_minutes,
            },
        }

    def _build_timeline(
        self,
        games: List[GameSession],
        leftover_clusters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Interleave games, unsorted clusters, and the gaps between them.

        Parameters
        ----------
        games : list of GameSession
            Detected albums.
        leftover_clusters : list of dict
            Clusters that did not become games.

        Returns
        -------
        list of dict
            Ordered ``game``, ``unsorted``, and ``break`` events.
        """
        items: List[Dict[str, Any]] = []
        total_games = max(len(games), 1)
        for game in games:
            items.append(
                {
                    "kind": "game",
                    "game_id": game.game_id,
                    "label": f"Game{format_game_id(game.game_id, total_games)}",
                    "start": game.start_time,
                    "end": game.end_time,
                    "photo_count": game.photo_count,
                    "duration_minutes": round(
                        (game.end_time - game.start_time).total_seconds() / 60.0, 1
                    ),
                    "photos_per_hour": round(
                        photos_per_hour(
                            game.photo_count,
                            (game.end_time - game.start_time).total_seconds() / 60.0,
                        ),
                        1,
                    ),
                }
            )
        for cluster in leftover_clusters:
            start_time = datetime.fromisoformat(str(cluster["start_time"]))
            end_time = datetime.fromisoformat(str(cluster["end_time"]))
            items.append(
                {
                    "kind": "unsorted",
                    "label": "unsorted",
                    "start": start_time,
                    "end": end_time,
                    "photo_count": cluster["photo_count"],
                    "duration_minutes": cluster["duration_minutes"],
                    "photos_per_hour": cluster["photos_per_hour"],
                    "reason": cluster["reason"],
                }
            )
        items.sort(key=lambda item: item["start"])

        timeline: List[Dict[str, Any]] = []
        for index, item in enumerate(items):
            if index > 0:
                previous = items[index - 1]
                gap_minutes = (
                    item["start"] - previous["end"]
                ).total_seconds() / 60.0
                timeline.append(
                    {
                        "kind": "break",
                        "gap_minutes": round(gap_minutes, 1),
                        "after": previous["label"],
                        "before": item["label"],
                    }
                )
            event = dict(item)
            event["start_label"] = item["start"].strftime("%d %b %Y %H:%M:%S")
            event["end_label"] = item["end"].strftime("%H:%M:%S")
            timeline.append(event)
        return timeline

    def _apply_detect_kwargs(self, kwargs: Mapping[str, Any]) -> None:
        """
        Overlay CLI/core keyword arguments onto the detector config.

        Parameters
        ----------
        kwargs : mapping
            May include ``min_duration``, ``min_gap``, ``min_photos``
            (absolute count), and ``min_rate`` / ``min_photos_per_hour``,
            or the config field names. ``None`` or ``0`` disables a floor.
        """
        if "min_duration" in kwargs:
            self.config.min_game_duration_minutes = int(kwargs["min_duration"])
        if "min_game_duration_minutes" in kwargs:
            self.config.min_game_duration_minutes = int(
                kwargs["min_game_duration_minutes"]
            )
        if "min_gap" in kwargs:
            self.config.min_gap_minutes = float(kwargs["min_gap"])
        if "min_gap_minutes" in kwargs:
            self.config.min_gap_minutes = float(kwargs["min_gap_minutes"])
        if "min_photos" in kwargs:
            self.config.min_photos = _optional_floor(kwargs["min_photos"])
        if "min_rate" in kwargs:
            self.config.min_photos_per_hour = _optional_floor(
                kwargs["min_rate"]
            )
        if "min_photos_per_hour" in kwargs:
            self.config.min_photos_per_hour = _optional_floor(
                kwargs["min_photos_per_hour"]
            )

    def _analyze_timestamps(self, photo_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Read capture times from each image's EXIF data.

        Parameters
        ----------
        photo_paths : list of Path
            Image files to read.

        Returns
        -------
        list of dict
            Metadata dicts with ``path``, ``filename``, ``timestamp``, and
            ``time_str``, sorted by capture time. Files without a usable
            EXIF datetime are omitted.
        """
        tag_maps = self._read_exif_tag_maps(photo_paths)
        photo_metadata: List[Dict[str, Any]] = []
        self._photo_timestamps = {}
        skipped = 0

        for photo_path, tags in zip(photo_paths, tag_maps):
            timestamp = timestamp_from_exif_tags(tags)
            if timestamp is None:
                skipped += 1
                continue

            resolved = str(photo_path.resolve())
            self._photo_timestamps[resolved] = timestamp
            photo_metadata.append(
                {
                    "path": photo_path,
                    "filename": photo_path.name,
                    "timestamp": timestamp,
                    "time_str": timestamp.strftime("%H%M%S"),
                }
            )

        if skipped:
            self.logger.warning(
                f"Skipped {skipped} photos with no EXIF capture time"
            )

        photo_metadata.sort(key=lambda item: item["timestamp"])
        return photo_metadata

    def _read_exif_tag_maps(
        self, photo_paths: List[Path]
    ) -> List[Dict[str, str]]:
        """
        Read EXIF tag maps for ``photo_paths`` with fast-exif-rs-py.

        Parallel read is preferred. If that call fails as a batch, each
        file is read individually so one corrupt image does not drop the
        rest.

        Parameters
        ----------
        photo_paths : list of Path
            Image files to read.

        Returns
        -------
        list of dict
            One string-to-string tag map per input path, same order.

        Raises
        ------
        RuntimeError
            If fast-exif-rs-py is not installed.
        """
        try:
            import fast_exif_rs_py
        except ImportError as exc:
            raise RuntimeError(
                "fast-exif-rs-py is required to read EXIF capture times for "
                "game splitting. Install project dependencies (Rust is "
                "required to build the extension)."
            ) from exc

        path_strs = [str(photo_path) for photo_path in photo_paths]
        try:
            tag_maps = fast_exif_rs_py.read_exif_files_parallel(path_strs)
        except RuntimeError as exc:
            self.logger.warning(
                f"Parallel EXIF read failed ({exc}); reading files individually"
            )
            reader = fast_exif_rs_py.PyFastExifReader()
            tag_maps = []
            for path_str in path_strs:
                try:
                    tag_maps.append(reader.read_file(path_str))
                except RuntimeError as file_exc:
                    self.logger.debug(
                        f"EXIF read failed for {path_str}: {file_exc}"
                    )
                    tag_maps.append({})

        if len(tag_maps) != len(photo_paths):
            raise RuntimeError(
                "fast-exif-rs-py returned a different number of EXIF maps "
                "than input files"
            )

        return tag_maps

    def extract_timestamp_from_exif(self, photo_path: Path) -> Optional[datetime]:
        """
        Read the capture time from one image's EXIF data.

        Parameters
        ----------
        photo_path : Path
            Image file to read.

        Returns
        -------
        datetime or None
            Naive capture time, or None if no usable EXIF datetime exists.
        """
        resolved = str(photo_path.resolve())
        cached = self._photo_timestamps.get(resolved)
        if cached is not None:
            return cached

        tag_maps = self._read_exif_tag_maps([photo_path])
        timestamp = timestamp_from_exif_tags(tag_maps[0])
        if timestamp is not None:
            self._photo_timestamps[resolved] = timestamp
        return timestamp

    def extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Extract timestamp from filename.

        Kept for callers that still pass camera-style names. Game splitting
        uses EXIF via :meth:`extract_timestamp_from_exif`.

        Parameters
        ----------
        filename : str
            The filename to parse.

        Returns
        -------
        datetime or None
            Parsed datetime or None if parsing fails.
        """
        try:
            name = Path(filename).stem
            patterns = [
                "%Y%m%d_%H%M%S",
                "%Y%m%d_%H%M%S_%f",
                "IMG_%Y%m%d_%H%M%S",
            ]

            for pattern in patterns:
                try:
                    return datetime.strptime(name, pattern)
                except ValueError:
                    continue

            if "_" in name:
                parts = name.split("_")
                if len(parts) >= 2:
                    date_part = parts[0]
                    time_part = parts[1]

                    if len(date_part) == 8 and len(time_part) >= 6:
                        try:
                            date_str = f"{date_part}_{time_part[:6]}"
                            return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        except ValueError:
                            pass

            return None

        except Exception as e:
            self.logger.debug(f"Failed to parse timestamp from {filename}: {e}")
            return None

    def _detect_game_boundaries(
        self, photo_metadata: List[Dict]
    ) -> List[Tuple[int, int]]:
        """Detect game boundaries based on timestamp gaps."""
        if len(photo_metadata) < 2:
            return []

        boundaries = []
        current_start = 0

        for i in range(1, len(photo_metadata)):
            prev_time = photo_metadata[i - 1]["timestamp"]
            curr_time = photo_metadata[i]["timestamp"]

            # Calculate gap in minutes
            gap_minutes = (curr_time - prev_time).total_seconds() / 60

            # Use adaptive gap threshold based on context
            adaptive_gap_threshold = self._calculate_adaptive_gap_threshold(
                photo_metadata, current_start, i, gap_minutes
            )

            # If gap is large enough, end current game and start new one
            if gap_minutes >= adaptive_gap_threshold:
                game_duration_minutes = (
                    prev_time - photo_metadata[current_start]["timestamp"]
                ).total_seconds() / 60
                photo_count = i - current_start

                if session_meets_thresholds(
                    photo_count, game_duration_minutes, self.config
                ):
                    boundaries.append((current_start, i - 1))

                current_start = i

        # Add the last game if it meets criteria
        if current_start < len(photo_metadata):
            last_time = photo_metadata[-1]["timestamp"]
            first_time = photo_metadata[current_start]["timestamp"]
            game_duration_minutes = (last_time - first_time).total_seconds() / 60
            photo_count = len(photo_metadata) - current_start

            if session_meets_thresholds(
                photo_count, game_duration_minutes, self.config
            ):
                boundaries.append((current_start, len(photo_metadata) - 1))

        return boundaries

    def _calculate_adaptive_gap_threshold(
        self, photo_metadata: List[Dict], current_start: int, current_index: int, gap_minutes: float
    ) -> float:
        """
        Calculate an adaptive gap threshold based on context.
        
        This helps distinguish between:
        - Small breaks within a game (halftime, timeouts) - should NOT split
        - Actual game boundaries - should split
        """
        # Base threshold from config
        base_threshold = self.config.min_gap_minutes
        
        # Calculate current segment stats
        current_duration = (
            photo_metadata[current_index - 1]["timestamp"] - 
            photo_metadata[current_start]["timestamp"]
        ).total_seconds() / 60
        photo_count = current_index - current_start
        
        # If this would create an under-rate segment, be more lenient
        if not session_meets_thresholds(photo_count, current_duration, self.config):
            # Increase threshold to avoid splitting thin segments
            return max(base_threshold * 2, 30)  # At least 30 minutes
        
        # If the current potential game is already long enough, be more strict
        if current_duration >= max(float(self.config.min_game_duration_minutes), 30.0):
            # Game is already long enough, use normal threshold
            return base_threshold
        
        # If we're in the middle of what could be a game, be more lenient
        # Look ahead to see if there's a much larger gap coming
        look_ahead_distance = min(100, len(photo_metadata) - current_index)  # Increased look-ahead
        max_future_gap = 0
        
        for j in range(current_index + 1, min(current_index + look_ahead_distance, len(photo_metadata))):
            if j < len(photo_metadata):
                future_gap = (
                    photo_metadata[j]["timestamp"] - photo_metadata[j-1]["timestamp"]
                ).total_seconds() / 60
                max_future_gap = max(max_future_gap, future_gap)
        
        # If there's a much larger gap coming, this might be a small break within a game
        if max_future_gap > gap_minutes * 2 and gap_minutes < 30:  # Reduced multiplier
            self.logger.debug(f"Adaptive threshold: Found larger gap {max_future_gap:.1f} min ahead, increasing threshold from {base_threshold} to {max(base_threshold * 1.5, 25)}")
            return max(base_threshold * 1.5, 25)  # Be more lenient
        
        # Special case: if we're early in the sequence and there's a big gap coming,
        # be more lenient with small gaps
        if current_start < 200 and max_future_gap > 50:  # More lenient conditions
            self.logger.debug(f"Adaptive threshold: Early sequence with big gap ahead {max_future_gap:.1f} min, increasing threshold from {base_threshold} to {max(base_threshold * 2, 25)}")
            return max(base_threshold * 2, 25)  # Be more lenient
        
        return base_threshold

    def _merge_split_games(
        self, photo_metadata: List[Dict], boundaries: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Merge games that were split by small gaps but should be considered one game.
        
        This handles cases where there are small breaks within a game (like halftime)
        that shouldn't split the game into separate sessions.
        """
        if len(boundaries) <= 1:
            return boundaries

        merged_boundaries = []
        i = 0
        
        while i < len(boundaries):
            current_start, current_end = boundaries[i]
            
            # Check if we should merge with the next game
            if i + 1 < len(boundaries):
                next_start, next_end = boundaries[i + 1]
                
                # Calculate the gap between current and next game
                current_end_time = photo_metadata[current_end]["timestamp"]
                next_start_time = photo_metadata[next_start]["timestamp"]
                gap_minutes = (next_start_time - current_end_time).total_seconds() / 60
                
                # Calculate total duration if merged
                total_start_time = photo_metadata[current_start]["timestamp"]
                total_end_time = photo_metadata[next_end]["timestamp"]
                total_duration_minutes = (total_end_time - total_start_time).total_seconds() / 60
                total_photos = next_end - current_start + 1
                
                # Merge if:
                # 1. Gap is relatively small (less than 30 minutes)
                # 2. Total duration would be reasonable (less than 4 hours)
                # 3. Both games individually meet minimum requirements
                current_duration = (current_end_time - total_start_time).total_seconds() / 60
                next_duration = (total_end_time - next_start_time).total_seconds() / 60
                
                should_merge = (
                    gap_minutes < 30 and  # Small gap
                    total_duration_minutes < 240 and  # Less than 4 hours total
                    session_meets_thresholds(
                        current_end - current_start + 1, current_duration, self.config
                    )
                    and session_meets_thresholds(
                        next_end - next_start + 1, next_duration, self.config
                    )
                    and session_meets_thresholds(
                        total_photos, total_duration_minutes, self.config
                    )
                )
                
                if should_merge:
                    # Merge the games
                    merged_boundaries.append((current_start, next_end))
                    i += 2  # Skip the next game since we merged it
                    continue
            
            # Don't merge, keep current game
            merged_boundaries.append((current_start, current_end))
            i += 1
        
        return merged_boundaries

    def _create_game_sessions(
        self, photo_metadata: List[Dict], boundaries: List[Tuple[int, int]]
    ) -> List[GameSession]:
        """Create GameSession objects from boundaries."""
        games = []

        for i, (start_idx, end_idx) in enumerate(boundaries):
            game_photos = [
                meta["path"] for meta in photo_metadata[start_idx : end_idx + 1]
            ]
            start_time = photo_metadata[start_idx]["timestamp"]
            end_time = photo_metadata[end_idx]["timestamp"]

            game = GameSession(
                game_id=i + 1,
                start_time=start_time,
                end_time=end_time,
                photo_count=len(game_photos),
                photo_files=game_photos,
            )
            games.append(game)

        # Calculate gaps between games
        for i, game in enumerate(games):
            if i > 0:
                prev_game = games[i - 1]
                game.gap_before = int(
                    (game.start_time - prev_game.end_time).total_seconds()
                )

            if i < len(games) - 1:
                next_game = games[i + 1]
                game.gap_after = int(
                    (next_game.start_time - game.end_time).total_seconds()
                )

        return games

    def _format_games_for_output(self, games: List[GameSession]) -> List[Dict]:
        """Format games for output."""
        formatted_games = []

        for game in games:
            duration_minutes = (game.end_time - game.start_time).total_seconds() / 60

            formatted_game = {
                "game_id": game.game_id,
                "start_time": game.start_time.isoformat(),
                "end_time": game.end_time.isoformat(),
                "start_time_formatted": game.start_time.strftime("%H:%M:%S"),
                "end_time_formatted": game.end_time.strftime("%H:%M:%S"),
                "duration_minutes": round(duration_minutes, 1),
                "photo_count": game.photo_count,
                "gap_before_minutes": round(game.gap_before / 60, 1)
                if game.gap_before
                else None,
                "gap_after_minutes": round(game.gap_after / 60, 1)
                if game.gap_after
                else None,
                "photo_files": [str(photo_path) for photo_path in game.photo_files],
            }

            formatted_games.append(formatted_game)

        return formatted_games

    def _get_detected_dates(self, photo_metadata: List[Dict]) -> List[str]:
        """Get list of unique dates detected in the photos."""
        if not photo_metadata:
            return []

        dates = set()
        for meta in photo_metadata:
            dates.add(meta["timestamp"].strftime("%Y-%m-%d"))

        return sorted(list(dates))

    def load_split_file(self, split_file_path: Path) -> List[datetime]:
        """
        Load manual splits from a text file.

        Args:
            split_file_path: Path to the split file

        Returns:
            List of parsed timestamps
        """
        manual_splits = []

        try:
            with open(split_file_path, "r") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                try:
                    # Parse timestamp - handle both formats
                    if " " in line:
                        # Format: "YYYY-MM-DD HH:MM:SS"
                        timestamp = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
                    elif ":" in line:
                        # Format: "HH:MM:SS" - assume current detected date
                        time_part = datetime.strptime(line, "%H:%M:%S")
                        # Use the first detected date from the games
                        if self.games:
                            first_game_date = self.games[0].start_time.date()
                            timestamp = datetime.combine(
                                first_game_date, time_part.time()
                            )
                        else:
                            # Fallback to September 20th, 2025
                            timestamp = datetime(
                                2025,
                                9,
                                20,
                                time_part.hour,
                                time_part.minute,
                                time_part.second,
                            )
                    else:
                        # Format: "HHMMSS" - assume current detected date
                        time_part = datetime.strptime(line, "%H%M%S")
                        if self.games:
                            first_game_date = self.games[0].start_time.date()
                            timestamp = datetime.combine(
                                first_game_date, time_part.time()
                            )
                        else:
                            # Fallback to September 20th, 2025
                            timestamp = datetime(
                                2025,
                                9,
                                20,
                                time_part.hour,
                                time_part.minute,
                                time_part.second,
                            )

                    manual_splits.append(timestamp)

                except ValueError as e:
                    self.logger.warning(
                        f"Invalid timestamp format on line {line_num}: '{line}' - {e}"
                    )
                    continue

            # Sort splits
            manual_splits.sort()

            self.logger.info(
                f"Loaded {len(manual_splits)} manual splits from {split_file_path}"
            )
            return manual_splits

        except FileNotFoundError:
            self.logger.error(f"Split file not found: {split_file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading split file: {e}")
            return []

    def apply_manual_splits(self, manual_splits: List[datetime]) -> List[GameSession]:
        """
        Apply manual splits to the detected games.

        Args:
            manual_splits: List of manual split timestamps

        Returns:
            List of GameSession objects with manual splits applied
        """
        if not manual_splits or not self.games:
            return self.games

        self.logger.info(f"Applying {len(manual_splits)} manual splits")

        final_games = []

        for game in self.games:
            # Find manual splits that fall within this game's time range
            game_splits = [
                split
                for split in manual_splits
                if game.start_time <= split <= game.end_time
            ]

            if not game_splits:
                # No splits within this game, keep it as is
                final_games.append(game)
                continue

            # Sort splits and add start/end times
            game_splits.sort()
            split_points = [game.start_time] + game_splits + [game.end_time]

            # Create sub-games for each segment
            for i in range(len(split_points) - 1):
                segment_start = split_points[i]
                segment_end = split_points[i + 1]

                # Find photos in this time segment using EXIF capture times
                segment_photos = []
                for photo in game.photo_files:
                    photo_time = self.extract_timestamp_from_exif(photo)
                    if photo_time is None:
                        continue
                    if segment_start <= photo_time <= segment_end:
                        segment_photos.append(photo)

                segment_minutes = (segment_end - segment_start).total_seconds() / 60
                if session_meets_thresholds(
                    len(segment_photos), segment_minutes, self.config
                ):
                    # Create new game session for this segment
                    segment_game = GameSession(
                        game_id=len(final_games) + 1,
                        start_time=segment_start,
                        end_time=segment_end,
                        photo_count=len(segment_photos),
                        photo_files=segment_photos,
                    )
                    final_games.append(segment_game)

        # Reassign game IDs and calculate gaps
        for i, game in enumerate(final_games, 1):
            game.game_id = i

        # Calculate gaps between games
        for i, game in enumerate(final_games):
            if i > 0:
                prev_game = final_games[i - 1]
                game.gap_before = int(
                    (game.start_time - prev_game.end_time).total_seconds()
                )

            if i < len(final_games) - 1:
                next_game = final_games[i + 1]
                game.gap_after = int(
                    (next_game.start_time - game.end_time).total_seconds()
                )

        self.logger.info(f"Applied manual splits. Final games: {len(final_games)}")
        return final_games
