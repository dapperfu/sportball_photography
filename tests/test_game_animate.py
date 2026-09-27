"""
Tests for game-folder animation helpers.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

from sportball.detectors.animate import (
    build_ffmpeg_command,
    discover_game_folders,
    encode_game_video,
    is_game_folder,
    list_game_photos,
    parse_video_size,
    plan_animation,
    resolve_fps,
)


def test_is_game_folder_matches_split_names(tmp_path: Path) -> None:
    """Only Game##_<date>_<HHMMSS>-<HHMMSS> directories qualify."""
    good = tmp_path / "Game44_12Oct2025_144301-144303"
    good.mkdir()
    bad = tmp_path / "SpringGames"
    bad.mkdir()
    assert is_game_folder(good)
    assert not is_game_folder(bad)
    assert not is_game_folder(tmp_path / "missing")


def test_discover_game_folders_from_parent_and_album(tmp_path: Path) -> None:
    """Parent Games/ yields child albums; a single album is accepted."""
    parent = tmp_path / "Games"
    parent.mkdir()
    first = parent / "Game01_12Oct2025_140118-140119"
    second = parent / "Game44_12Oct2025_144301-144303"
    first.mkdir()
    second.mkdir()
    (parent / "notes.txt").write_text("ignore")

    found = discover_game_folders([parent])
    assert [item.name for item in found] == [
        "Game01_12Oct2025_140118-140119",
        "Game44_12Oct2025_144301-144303",
    ]
    assert discover_game_folders([second]) == [second.resolve()]


def test_resolve_fps_from_duration_and_explicit() -> None:
    """Duration mode derives rate; --fps passes through fractions."""
    assert resolve_fps(120, duration_seconds=60.0) == 2.0
    assert resolve_fps(30, fps=12.5) == 12.5


def test_plan_animation_writes_matching_mp4_name(tmp_path: Path) -> None:
    """Output is Game##_date_times.mp4 beside the album folder."""
    album = tmp_path / "Game44_12Oct2025_144301-144303"
    album.mkdir()
    for index in range(3):
        (album / f"20251012_14430{index}.jpg").write_bytes(b"fake")

    plan = plan_animation(album, duration_seconds=60.0)
    assert plan.output_path == tmp_path / "Game44_12Oct2025_144301-144303.mp4"
    assert plan.fps == 3.0 / 60.0
    assert len(plan.photos) == 3


def test_list_game_photos_sorts_by_name(tmp_path: Path) -> None:
    """Frames are sorted by basename so time-stamped names play in order."""
    album = tmp_path / "Game01_12Oct2025_140118-140119"
    album.mkdir()
    later = album / "20251012_140119.jpg"
    earlier = album / "20251012_140118.jpg"
    later.write_bytes(b"b")
    earlier.write_bytes(b"a")
    (album / "readme.txt").write_text("skip")
    names = [path.name for path in list_game_photos(album)]
    assert names == ["20251012_140118.jpg", "20251012_140119.jpg"]


def test_encode_game_video_dry_run_and_skip(tmp_path: Path) -> None:
    """Dry-run reports the plan; existing mp4 is skipped without --force."""
    album = tmp_path / "Game02_12Oct2025_140154-140155"
    album.mkdir()
    (album / "a.jpg").write_bytes(b"a")
    plan = plan_animation(album, fps=1.0)
    dry = encode_game_video(plan, dry_run=True)
    assert dry.success and not dry.skipped
    assert "would write" in dry.message

    plan.output_path.write_bytes(b"old")
    skipped = encode_game_video(plan, dry_run=False)
    assert skipped.skipped
    assert "exists" in skipped.message


def test_encode_game_video_invokes_ffmpeg(tmp_path: Path) -> None:
    """Successful ffmpeg run writes the planned output path in the command."""
    album = tmp_path / "Game03_12Oct2025_140226-140229"
    album.mkdir()
    (album / "a.jpg").write_bytes(b"a")
    (album / "b.jpg").write_bytes(b"b")
    plan = plan_animation(album, fps=0.5)

    fake = MagicMock()
    fake.returncode = 0
    fake.stderr = ""
    with patch(
        "sportball.detectors.animate.find_ffmpeg", return_value="/usr/bin/ffmpeg"
    ), patch("sportball.detectors.animate.subprocess.run", return_value=fake) as run:
        result = encode_game_video(plan, force=True)

    assert result.success and not result.skipped
    cmd: List[str] = run.call_args.args[0]
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert str(plan.output_path) in cmd
    assert "-f" in cmd and "concat" in cmd
    assert "-vf" not in cmd


def test_parse_video_size_full_and_partial() -> None:
    """5568x3712 is exact; 5568x and x1080 leave one axis free."""
    exact = parse_video_size("5568x3712")
    assert exact.width == 5568
    assert exact.height == 3712
    assert exact.scale_filter() == "scale=5568:3712"
    assert exact.label == "5568x3712"

    width_only = parse_video_size("5568x")
    assert width_only.width == 5568
    assert width_only.height is None
    assert width_only.scale_filter() == "scale=5568:-2"
    assert width_only.label == "5568x"

    height_only = parse_video_size("x3712")
    assert height_only.scale_filter() == "scale=-2:3712"

    hd = parse_video_size("x1080")
    assert hd.height == 1080
    assert hd.scale_filter() == "scale=-2:1080"
    assert hd.label == "x1080"

    assert parse_video_size("5568X3712").scale_filter() == "scale=5568:3712"

    for bad in ("x", "1080", "x0", "0x1080", "5569x", "5568x3712x", ""):
        try:
            parse_video_size(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_encode_command_scales_to_requested_size(tmp_path: Path) -> None:
    """--size x1080 becomes an ffmpeg scale filter; native size omits -vf."""
    album = tmp_path / "Game04_12Oct2025_140230-140231"
    album.mkdir()
    (album / "a.jpg").write_bytes(b"a")
    size = parse_video_size("x1080")
    plan = plan_animation(album, fps=1.0, size=size)
    cmd = build_ffmpeg_command("ffmpeg", tmp_path / "concat.txt", plan)
    assert cmd[cmd.index("-vf") + 1] == "scale=-2:1080"

    native = plan_animation(album, fps=1.0)
    native_cmd = build_ffmpeg_command("ffmpeg", tmp_path / "concat.txt", native)
    assert "-vf" not in native_cmd

    dry = encode_game_video(plan, dry_run=True)
    assert "x1080" in dry.message
