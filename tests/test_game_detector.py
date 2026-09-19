"""
Tests for EXIF-based game timestamp extraction.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

from sportball.detectors.game import (
    GameDetector,
    parse_exif_datetime,
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
