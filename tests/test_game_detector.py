"""
Tests for EXIF-based game timestamp extraction.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from sportball.detectors.game import (
    GameDetectionConfig,
    GameDetector,
    build_activity_histogram,
    build_gap_histogram,
    collect_photo_paths,
    expand_input_directories,
    format_fractional_minutes,
    format_game_id,
    histogram_bar,
    histogram_bin_minutes,
    leftover_index_ranges,
    leftover_reason,
    parse_exif_datetime,
    photos_per_hour,
    resolve_session_floors,
    session_meets_thresholds,
    timestamp_from_exif_tags,
)


def test_parse_exif_datetime_standard() -> None:
    """Parse the usual EXIF DateTimeOriginal format."""
    parsed = parse_exif_datetime("2025:09:20 09:01:22")
    assert parsed == datetime(2025, 9, 20, 9, 1, 22)


def test_parse_exif_datetime_strips_timezone() -> None:
    """Ignore a trailing offset so photos sort in camera-local time."""
    parsed = parse_exif_datetime("2025:09:20 09:01:22-04:00")
    assert parsed == datetime(2025, 9, 20, 9, 1, 22)


def test_timestamp_prefers_datetime_original() -> None:
    """DateTimeOriginal wins over later modify times."""
    tags = {
        "ModifyDate": "2025:09:21 00:00:00",
        "DateTimeOriginal": "2025:09:20 09:01:22",
        "SubSecTimeOriginal": "96",
    }
    parsed = timestamp_from_exif_tags(tags)
    assert parsed == datetime(2025, 9, 20, 9, 1, 22, 960000)


def test_timestamp_accepts_exiftool_group_prefix() -> None:
    """Accept EXIF:DateTimeOriginal as well as the bare tag name."""
    tags = {"EXIF:DateTimeOriginal": "2025:09:20 11:30:05"}
    parsed = timestamp_from_exif_tags(tags)
    assert parsed == datetime(2025, 9, 20, 11, 30, 5)


def test_timestamp_missing_returns_none() -> None:
    """No capture tag means the photo is not used for splitting."""
    assert timestamp_from_exif_tags({"Make": "Canon"}) is None


def test_analyze_timestamps_uses_fast_exif_rs_py(tmp_path: Path) -> None:
    """Game detection reads EXIF through fast-exif-rs-py, not filenames."""
    photo_a = tmp_path / "random_name_a.jpg"
    photo_b = tmp_path / "random_name_b.jpg"
    photo_a.write_bytes(b"fake")
    photo_b.write_bytes(b"fake")

    tag_maps: List[Dict[str, str]] = [
        {"DateTimeOriginal": "2025:09:20 11:30:05"},
        {"DateTimeOriginal": "2025:09:20 09:01:22"},
    ]
    fake_module = MagicMock()
    fake_module.read_exif_files_parallel.return_value = tag_maps

    detector = GameDetector()
    with patch.dict("sys.modules", {"fast_exif_rs_py": fake_module}):
        metadata = detector._analyze_timestamps([photo_a, photo_b])

    fake_module.read_exif_files_parallel.assert_called_once()
    assert [item["path"] for item in metadata] == [photo_b, photo_a]
    assert metadata[0]["timestamp"] == datetime(2025, 9, 20, 9, 1, 22)
    assert metadata[1]["timestamp"] == datetime(2025, 9, 20, 11, 30, 5)


def test_photos_per_hour_matches_short_sessions() -> None:
    """14 shots in 12 min is 70/hour; 37 in 9 min is 37*60/9."""
    assert photos_per_hour(14, 12.0) == 70.0
    assert photos_per_hour(37, 9.0) == 37.0 * 60.0 / 9.0


def test_short_dense_sessions_qualify_at_one_hundred_per_hour() -> None:
    """Default floors are 10 photos and 100/hour."""
    config = GameDetectionConfig()
    assert session_meets_thresholds(20, 12.0, config)
    assert session_meets_thresholds(12, 7.0, config)
    assert not session_meets_thresholds(10, 12.0, config)


def test_sparse_session_below_hourly_rate_is_rejected() -> None:
    """Same 14 photos over an hour are only 14/hour."""
    config = GameDetectionConfig()
    assert not session_meets_thresholds(14, 60.0, config)
    assert not session_meets_thresholds(9, 12.0, config)
    assert not session_meets_thresholds(1, 12.0, config)


def test_wrestling_uses_absolute_photo_floor() -> None:
    """A 7-minute match with 10 frames counts when rate is off."""
    config = GameDetectionConfig(min_photos=10, min_photos_per_hour=None)
    assert session_meets_thresholds(10, 7.0, config)
    assert not session_meets_thresholds(9, 7.0, config)


def test_rate_only_ignores_absolute_count() -> None:
    """Field sports can keep a short burst that is still dense."""
    config = GameDetectionConfig(min_photos=None, min_photos_per_hour=100)
    assert session_meets_thresholds(5, 2.0, config)
    assert not session_meets_thresholds(5, 12.0, config)


def test_resolve_session_floors_are_mutually_exclusive() -> None:
    """One flag disables the other; passing both is an error."""
    assert resolve_session_floors(
        10, 100, photos_explicit=False, rate_explicit=False
    ) == (10, 100)
    assert resolve_session_floors(
        15, 100, photos_explicit=True, rate_explicit=False
    ) == (15, None)
    assert resolve_session_floors(
        10, 80, photos_explicit=False, rate_explicit=True
    ) == (None, 80)
    with pytest.raises(ValueError, match="either --min-photos or --min-rate"):
        resolve_session_floors(10, 100, photos_explicit=True, rate_explicit=True)


def test_apply_detect_kwargs_maps_count_not_rate() -> None:
    """min_photos is an absolute count; min_rate is photos per hour."""
    detector = GameDetector()
    detector._apply_detect_kwargs({"min_photos": 12, "min_rate": None})
    assert detector.config.min_photos == 12
    assert detector.config.min_photos_per_hour is None


def test_detect_boundaries_keeps_twelve_minute_burst() -> None:
    """A 20-photo 12-minute burst separated from a later game is kept."""
    start = datetime(2026, 9, 12, 11, 7, 1)
    early = [
        {
            "path": Path(f"early_{index}.jpg"),
            "filename": f"early_{index}.jpg",
            "timestamp": start + timedelta(seconds=index * 36),
            "time_str": "",
        }
        for index in range(20)
    ]
    later_start = datetime(2026, 9, 19, 12, 9, 15)
    later = [
        {
            "path": Path(f"later_{index}.jpg"),
            "filename": f"later_{index}.jpg",
            "timestamp": later_start.replace(
                minute=9 + (index // 30), second=index % 60
            ),
            "time_str": "",
        }
        for index in range(80)
    ]
    detector = GameDetector()
    boundaries = detector._detect_game_boundaries(early + later)
    assert boundaries[0] == (0, 19)
    assert len(boundaries) >= 2


def test_expand_input_directories_keeps_month_folders(tmp_path: Path) -> None:
    """Named month folders are kept; Games is excluded from a star glob."""
    april = tmp_path / "04_Apr"
    may = tmp_path / "05_May"
    games = tmp_path / "Games"
    april.mkdir()
    may.mkdir()
    games.mkdir()
    (tmp_path / "notes.txt").write_text("skip files")

    selected = expand_input_directories(
        ("04_Apr", "05_May"), output_dir=Path("Games"), cwd=tmp_path
    )
    assert selected == [april.resolve(), may.resolve()]

    starred = expand_input_directories(
        ("*",), output_dir=Path("SpringGames"), cwd=tmp_path
    )
    assert games.resolve() in starred
    assert april.resolve() in starred

    starred_default = expand_input_directories(
        ("*",), output_dir=Path("Games"), cwd=tmp_path
    )
    assert games.resolve() not in starred_default
    assert set(starred_default) == {april.resolve(), may.resolve()}


def test_collect_photo_paths_pools_directories_and_skips_output(
    tmp_path: Path,
) -> None:
    """Photos from every input dir are kept; previous game albums are not."""
    april = tmp_path / "04_Apr"
    may = tmp_path / "05_May"
    games = tmp_path / "Games" / "Game1_old"
    april.mkdir()
    may.mkdir()
    games.mkdir(parents=True)
    (april / "a.jpg").write_bytes(b"a")
    (may / "b.jpg").write_bytes(b"b")
    (games / "already.jpg").write_bytes(b"old")

    photos = collect_photo_paths(
        [april, may, tmp_path], output_dir=tmp_path / "Games"
    )
    names = sorted(path.name for path in photos)
    assert names == ["a.jpg", "b.jpg"]


def test_format_game_id_zero_pads_for_sort_order() -> None:
    """Single-digit games use 01; 100+ games use three digits."""
    assert format_game_id(1, 9) == "01"
    assert format_game_id(1, 12) == "01"
    assert format_game_id(12, 12) == "12"
    assert format_game_id(1, 100) == "001"
    assert format_game_id(100, 100) == "100"


def test_leftover_index_ranges_skips_detected_games() -> None:
    """Photos between game index ranges stay unsorted."""
    assert leftover_index_ranges(10, [(2, 5), (8, 9)]) == [(0, 1), (6, 7)]
    assert leftover_index_ranges(4, [(0, 3)]) == []


def test_activity_histogram_labels_games_and_unsorted() -> None:
    """Bins show Game01 or unsorted; empty slots are dropped."""
    stamps = [datetime(2026, 9, 12, 11, 7, index) for index in range(10)]
    stamps.extend(datetime(2026, 9, 19, 12, 0, index) for index in range(10))
    assignments: List[Optional[int]] = [None] * 10 + [1] * 10
    rows = build_activity_histogram(stamps, assignments, 1, bin_minutes=60)
    assert len(rows) == 2
    assert rows[0]["assignment"] == "unsorted"
    assert rows[0]["count"] == 10
    assert rows[1]["assignment"] == "Game01"
    assert "█" in histogram_bar(10, 10)
    assert histogram_bin_minutes(12.0) == 1


def test_gap_histogram_counts_intershot_seconds() -> None:
    """Burst gaps and a 10-minute hole show up as separate --min-gap bins."""
    base = datetime(2026, 9, 19, 12, 0, 0)
    stamps = [base + timedelta(seconds=0.3 * index) for index in range(5)]
    stamps.append(stamps[-1] + timedelta(seconds=600))
    rows = build_gap_histogram(stamps, min_gap_minutes=10.0)
    by_label = {str(row["label"]): row for row in rows}
    assert by_label["0.25s–0.5s"]["count"] == 4
    ten_min = by_label["10m–15m"]
    assert ten_min["count"] == 1
    assert ten_min["min_gap_minutes"] == "10"
    assert ten_min["effect"] == "split"
    assert format_fractional_minutes(30.0) == "0.5"
    assert format_fractional_minutes(5.0) == "0.0833"


def test_leftover_reason_explains_hourly_floor() -> None:
    """14 photos over an hour miss the 100/h floor."""
    config = GameDetectionConfig()
    assert "under" in leftover_reason(14, 60.0, config)
    assert leftover_reason(1, 0.0, config) == "fewer than 2 photos"
    assert leftover_reason(5, 1.0, config) == "5 photos is under 10 photos"


def test_analyze_timeline_includes_breaks() -> None:
    """Games and leftover clusters are listed in time order with gaps."""
    from sportball.detectors.game import GameSession

    detector = GameDetector()
    early = [
        {
            "path": Path(f"e{index}.jpg"),
            "filename": f"e{index}.jpg",
            "timestamp": datetime(2026, 9, 12, 11, 7, index),
            "time_str": "",
        }
        for index in range(3)
    ]
    later = [
        {
            "path": Path(f"l{index}.jpg"),
            "filename": f"l{index}.jpg",
            "timestamp": datetime(2026, 9, 19, 12, 0, index),
            "time_str": "",
        }
        for index in range(5)
    ]
    metadata = early + later
    games = [
        GameSession(
            game_id=1,
            start_time=later[0]["timestamp"],
            end_time=later[-1]["timestamp"],
            photo_count=5,
            photo_files=[item["path"] for item in later],
        )
    ]
    analysis = detector._build_analysis(metadata, games, [(3, 7)])
    kinds = [event["kind"] for event in analysis["timeline"]]
    assert kinds == ["unsorted", "break", "game"]
    assert analysis["unsorted"][0]["photo_count"] == 3
    assert analysis["histogram"][0]["assignment"] == "unsorted"
    assert analysis["histogram"][-1]["assignment"] == "Game01"
    assert analysis["gap_histogram"]
    assert analysis["summary"]["min_photos"] == 10
    assert analysis["summary"]["min_photos_per_hour"] == 100


def test_split_rejects_min_photos_and_min_rate_together(
    tmp_path: Path,
) -> None:
    """CLI refuses both --min-photos and --min-rate together."""
    from click.testing import CliRunner

    from sportball.cli.main import cli

    dump = tmp_path / "10-Oct"
    dump.mkdir()
    result = CliRunner().invoke(
        cli,
        [
            "--quiet",
            "split",
            str(dump),
            "--min-photos",
            "10",
            "--min-rate",
            "100",
        ],
    )
    assert result.exit_code != 0
    assert "either --min-photos or --min-rate" in result.output
