"""
Tests for action-panorama detection.

Control-point search is stubbed. One test drives fake ``pto_gen`` and
``cpfind`` executables so the preview scaling path runs without Hugin.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import os
import stat
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pytest
from click.testing import CliRunner
from PIL import Image, UnidentifiedImageError

from sportball.cli.commands.pano_commands import pano
from sportball.detectors import pano as pano_mod
from sportball.detectors.pano import (
    ControlPoint,
    Frame,
    LinkReport,
    PairSurvey,
    PanoConfig,
    PanoGroup,
    PanoResult,
    SurveyFn,
    anchor_index,
    classify_burst_ranges,
    cut_known_start,
    content_box,
    crop_black_canvas,
    crop_panorama_directory,
    detect_panos,
    find_action_panos,
    find_marker_runs,
    format_pano_folder,
    known_window_indices,
    measure_marker_variation,
    parse_pto_control_points,
    render_pto,
    resolve_hugin_tool,
    survey_control_points,
    write_pano_folders,
)


def _frame(
    directory: Path,
    name: str,
    offset: float,
    mse: float,
    size: Tuple[int, int] = (40, 20),
) -> Frame:
    """Write a tiny JPEG and return a scored frame at ``offset`` seconds."""
    path = directory / name
    level = 0 if mse <= 100.0 else 180
    Image.new("RGB", size, (level, level, level)).save(path, "JPEG")
    base = datetime(2025, 9, 20, 9, 0, 0)
    return Frame(
        path=path,
        timestamp=base + timedelta(seconds=offset),
        variation=mse,
        width=size[0],
        height=size[1],
    )


def _scripted_survey(
    neighbor: int = 20,
    stride_points: int = 0,
    weak_names: Optional[Set[str]] = None,
) -> SurveyFn:
    """
    Build a survey whose neighbor counts are constant except named frames.

    A name in ``weak_names`` is the older frame of a pair with zero points.
    """
    weak = weak_names or set()

    def survey(frames: Sequence[Frame], linearmatchlen: int) -> PairSurvey:
        counts: Dict[Tuple[int, int], int] = {}
        points: List[ControlPoint] = []
        for index in range(len(frames) - 1):
            total = 0 if frames[index].path.name in weak else neighbor
            counts[(index, index + 1)] = total
            if total:
                points.append(ControlPoint(index, index + 1, 5.0, 6.0, 7.0, 8.0))
        if linearmatchlen >= 3:
            for index in range(len(frames) - 3):
                counts[(index, index + 3)] = stride_points
        return PairSurvey(counts=counts, points=points)

    return survey


def test_measure_marker_variation_ignores_brightness(tmp_path: Path) -> None:
    """A flat frame scores near 0 at any brightness. A split frame does not."""
    black = tmp_path / "black.png"
    gray = tmp_path / "gray.png"
    white = tmp_path / "white.png"
    sky = tmp_path / "sky.png"
    ground = tmp_path / "ground.png"
    Image.new("L", (32, 16), 0).save(black)
    Image.new("L", (32, 16), 10).save(gray)
    Image.new("L", (8, 8), 255).save(white)
    Image.new("RGB", (32, 16), (120, 180, 230)).save(sky)
    Image.new("RGB", (32, 16), (70, 110, 40)).save(ground)

    black_var, black_mean, width, height = measure_marker_variation(black, 8)
    gray_var, gray_mean, _, _ = measure_marker_variation(gray, 4)
    white_var, white_mean, _, _ = measure_marker_variation(white, 4)
    sky_var, _, _, _ = measure_marker_variation(sky, 8)
    ground_var, _, _, _ = measure_marker_variation(ground, 8)

    assert (width, height) == (32, 16)
    assert black_var == pytest.approx(0.0)
    assert black_mean == pytest.approx(0.0)
    assert gray_var == pytest.approx(0.0)
    assert gray_mean == pytest.approx(10.0)
    assert white_var == pytest.approx(0.0)
    assert white_mean == pytest.approx(255.0)
    assert sky_var == pytest.approx(0.0, abs=1e-6)
    assert ground_var == pytest.approx(0.0, abs=1e-6)

    split = tmp_path / "split.png"
    image = Image.new("L", (8, 8), 0)
    image.paste(255, (4, 0, 8, 8))
    image.save(split)
    split_var, split_mean, _, _ = measure_marker_variation(split, 8)
    assert split_mean == pytest.approx(127.5)
    assert split_var == pytest.approx(127.5**2)


def test_measure_marker_variation_rejects_garbage(tmp_path: Path) -> None:
    """A non-image is undecodable."""
    path = tmp_path / "notes.jpg"
    path.write_bytes(b"this is not a photo")
    with pytest.raises(UnidentifiedImageError):
        measure_marker_variation(path, 8)


def test_find_marker_runs_needs_two_close_frames(tmp_path: Path) -> None:
    """One flat frame is not a marker. A pair split by 11s is two shorts."""
    frames = [
        _frame(tmp_path, "a.jpg", 0.0, 400.0),
        _frame(tmp_path, "b.jpg", 1.0, 0.0),
        _frame(tmp_path, "c.jpg", 2.0, 400.0),
        _frame(tmp_path, "d.jpg", 3.0, 0.0),
        _frame(tmp_path, "e.jpg", 4.0, 0.0),
        _frame(tmp_path, "f.jpg", 20.0, 0.0),
        _frame(tmp_path, "g.jpg", 21.0, 0.0),
    ]
    runs = find_marker_runs(
        frames, marker_var=100.0, marker_count=2, discontinuity=10.0
    )
    assert runs == [(3, 4), (5, 6)]


def test_known_window_stops_past_ten_seconds(tmp_path: Path) -> None:
    """A gap of 10s stays in the window. A gap of 10.1s does not."""
    close = [
        _frame(tmp_path, "c0.jpg", 0.0, 400.0),
        _frame(tmp_path, "c1.jpg", 10.0, 400.0),
        _frame(tmp_path, "c2.jpg", 10.2, 0.0),
        _frame(tmp_path, "c3.jpg", 10.4, 0.0),
    ]
    wide = [
        _frame(tmp_path, "w0.jpg", 0.0, 400.0),
        _frame(tmp_path, "w1.jpg", 10.1, 400.0),
        _frame(tmp_path, "w2.jpg", 10.3, 0.0),
        _frame(tmp_path, "w3.jpg", 10.5, 0.0),
    ]
    assert known_window_indices(close, 2, 10.0, 100.0, {}) == [0, 1]
    assert known_window_indices(wide, 2, 10.0, 100.0, {}) == [1]


def test_cut_known_start_largest_gap_and_first() -> None:
    """Largest weak gap wins. Equal gaps keep the cut nearer the marker."""
    assert cut_known_start([0, 0], [5.0, 1.0], 0, "largest-gap") == 1
    assert cut_known_start([0, 0], [5.0, 1.0], 0, "first") == 2
    assert cut_known_start([0, 0], [2.0, 2.0], 0, "largest-gap") == 2
    assert cut_known_start([12, 8], [0.2, 4.0], 0, "largest-gap") == 0


def test_classify_burst_guessed_held_and_open() -> None:
    """Neighbors plus a dark stride is a pano. Stride overlap is a hold."""
    guessed = {(index, index + 1): 20 for index in range(4)}
    guessed.update({(index, index + 3): 0 for index in range(2)})
    assert classify_burst_ranges(5, guessed, 3, 10, 0, 4) == [("guessed", 0, 4)]

    held = {(index, index + 1): 20 for index in range(4)}
    held.update({(index, index + 3): 15 for index in range(2)})
    assert classify_burst_ranges(5, held, 3, 10, 0, 4) == [("held", 0, 4)]

    quiet = {(index, index + 1): 2 for index in range(4)}
    assert classify_burst_ranges(5, quiet, 3, 10, 0, 4) == [("open", 0, 4)]


def test_known_and_guessed_share_one_sequence(tmp_path: Path) -> None:
    """The weak link before the markers starts the known pan. The prefix is guessed."""
    specs = [
        ("f00.jpg", 0.0),
        ("f01.jpg", 0.5),
        ("f02.jpg", 1.0),
        ("f03.jpg", 1.5),
        ("f04.jpg", 2.0),
        ("f05.jpg", 5.0),
        ("f06.jpg", 5.5),
        ("f07.jpg", 6.0),
        ("f08.jpg", 6.5),
        ("f09.jpg", 7.0),
    ]
    frames = [_frame(tmp_path, name, offset, 400.0) for name, offset in specs]
    frames.append(_frame(tmp_path, "b0.jpg", 7.2, 0.0))
    frames.append(_frame(tmp_path, "b1.jpg", 7.4, 0.0))

    groups, runs = detect_panos(
        frames,
        PanoConfig(),
        _scripted_survey(weak_names={"f04.jpg"}),
    )
    accepted = [group for group in groups if group.accepted]
    assert runs == [(10, 11)]
    assert [
        (group.kind, [frame.path.name for frame in group.frames]) for group in accepted
    ] == [
        ("guessed", ["f00.jpg", "f01.jpg", "f02.jpg", "f03.jpg", "f04.jpg"]),
        ("known", ["f05.jpg", "f06.jpg", "f07.jpg", "f08.jpg", "f09.jpg"]),
    ]
    known = next(group for group in accepted if group.kind == "known")
    assert known.cut_link is not None
    assert known.cut_link.control_points == 0
    assert known.number == 2
    assert "b0.jpg" not in [frame.path.name for frame in known.frames]
    assert known.marker[0].path.name == "b0.jpg"


def test_discontinuity_splits_a_marked_pan_from_an_earlier_one(tmp_path: Path) -> None:
    """Eleven seconds is a different shot, so the earlier burst is guessed."""
    early = [
        _frame(tmp_path, f"a{index}.jpg", index * 0.5, 400.0) for index in range(5)
    ]
    late = [
        _frame(tmp_path, f"b{index}.jpg", 13.0 + index * 0.5, 400.0)
        for index in range(5)
    ]
    markers = [
        _frame(tmp_path, "k0.jpg", 16.0, 0.0),
        _frame(tmp_path, "k1.jpg", 16.2, 0.0),
    ]
    groups, _runs = detect_panos(
        early + late + markers, PanoConfig(), _scripted_survey()
    )
    accepted = [group for group in groups if group.accepted]
    names = [
        (group.kind, [frame.path.name for frame in group.frames]) for group in accepted
    ]
    assert names == [
        ("guessed", ["a0.jpg", "a1.jpg", "a2.jpg", "a3.jpg", "a4.jpg"]),
        ("known", ["b0.jpg", "b1.jpg", "b2.jpg", "b3.jpg", "b4.jpg"]),
    ]


def test_span_does_not_decide_a_guessed_panorama(tmp_path: Path) -> None:
    """2.5 seconds and 10 seconds are both panoramas when the geometry matches."""
    quick = [
        _frame(tmp_path, f"q{index}.jpg", index * 0.625, 400.0) for index in range(5)
    ]
    slow = [
        _frame(tmp_path, f"s{index}.jpg", 20.0 + index * 2.5, 400.0)
        for index in range(5)
    ]
    groups, runs = detect_panos(quick + slow, PanoConfig(), _scripted_survey())
    assert runs == []
    accepted = [group for group in groups if group.accepted]
    assert [group.kind for group in accepted] == ["guessed", "guessed"]
    assert len(accepted[0].frames) == 5
    assert len(accepted[1].frames) == 5
    quick_span = (
        accepted[0].frames[-1].timestamp - accepted[0].frames[0].timestamp
    ).total_seconds()
    slow_span = (
        accepted[1].frames[-1].timestamp - accepted[1].frames[0].timestamp
    ).total_seconds()
    assert quick_span == pytest.approx(2.5)
    assert slow_span == pytest.approx(10.0)


def test_static_hold_is_not_a_panorama(tmp_path: Path) -> None:
    """Overlap at the stride distance means the camera did not move enough."""
    frames = [
        _frame(tmp_path, f"h{index}.jpg", float(index), 400.0) for index in range(5)
    ]
    groups, _runs = detect_panos(
        frames, PanoConfig(), _scripted_survey(stride_points=15)
    )
    assert len(groups) == 1
    assert groups[0].kind == "held"
    assert groups[0].accepted is False


def test_one_flat_frame_does_not_mark_a_known_panorama(tmp_path: Path) -> None:
    """A single uniform frame is not the marker signal."""
    frames = [
        _frame(tmp_path, f"p{index}.jpg", float(index), 400.0) for index in range(5)
    ]
    frames.append(_frame(tmp_path, "dark.jpg", 5.0, 0.0))
    groups, runs = detect_panos(frames, PanoConfig(), _scripted_survey())
    assert runs == []
    assert [group.kind for group in groups if group.accepted] == ["guessed"]


def test_short_known_run_is_rejected(tmp_path: Path) -> None:
    """Fewer than --min-frames after the marker is reported and not written."""
    frames = [
        _frame(tmp_path, f"s{index}.jpg", float(index), 400.0) for index in range(3)
    ]
    frames.extend(
        [
            _frame(tmp_path, "k0.jpg", 3.2, 0.0),
            _frame(tmp_path, "k1.jpg", 3.4, 0.0),
        ]
    )
    groups, _runs = detect_panos(
        frames, PanoConfig(min_frames=4, guess=False), _scripted_survey()
    )
    assert len(groups) == 1
    assert groups[0].kind == "known"
    assert groups[0].accepted is False
    assert "need 4" in groups[0].reason


def test_first_boundary_keeps_only_the_suffix(tmp_path: Path) -> None:
    """``first`` stops at the weak link nearest the marker frames."""
    frames = [
        _frame(tmp_path, f"f{index}.jpg", float(index), 400.0) for index in range(6)
    ]
    frames.extend(
        [
            _frame(tmp_path, "k0.jpg", 6.2, 0.0),
            _frame(tmp_path, "k1.jpg", 6.4, 0.0),
        ]
    )
    groups, _runs = detect_panos(
        frames,
        PanoConfig(boundary="first", guess=False, min_frames=1),
        _scripted_survey(weak_names={"f1.jpg", "f4.jpg"}),
    )
    known = groups[0]
    assert [frame.path.name for frame in known.frames] == ["f5.jpg"]
    assert known.accepted is True


def test_folder_names_follow_the_game_pattern() -> None:
    """Known and guessed folders share the game date and clock suffix."""
    start = datetime(2025, 9, 20, 9, 0, 12)
    end = datetime(2025, 9, 20, 9, 0, 18)
    date_str = start.strftime("%d%b%Y")
    assert (
        format_pano_folder("known", 1, 2, start, end)
        == f"known_pano01_{date_str}_090012-090018"
    )
    assert format_pano_folder("guessed", 2, 100, start, end).startswith(
        "guessed_pano002_"
    )


def test_render_and_parse_pto_round_trip(tmp_path: Path) -> None:
    """A written project reads back the same images and control points."""
    frames = [
        _frame(tmp_path, "a.jpg", 0.0, 400.0),
        _frame(tmp_path, "b.jpg", 1.0, 400.0),
    ]
    point = ControlPoint(0, 1, 1.5, 2.25, 3.5, 4.75)
    text = render_pto(["a.jpg", "b.jpg"], frames, [point])
    images, points = parse_pto_control_points(text)
    assert images == ["a.jpg", "b.jpg"]
    assert points[0].x == pytest.approx(1.5)
    assert points[0].y_right == pytest.approx(4.75)
    assert 'n"a.jpg"' in text
    assert "black" not in text


def test_write_folders_skip_marker_frames_and_symlink(tmp_path: Path) -> None:
    """Accepted folders link the stitch frames and store a project."""
    frames = [
        _frame(tmp_path, f"f{index:02d}.jpg", index * 0.5, 400.0) for index in range(5)
    ]
    frames.extend(
        _frame(tmp_path, f"g{index}.jpg", 4.0 + index * 0.5, 400.0)
        for index in range(5)
    )
    frames.extend(
        [
            _frame(tmp_path, "b0.jpg", 7.0, 0.0),
            _frame(tmp_path, "b1.jpg", 7.2, 0.0),
        ]
    )
    groups, _runs = detect_panos(
        frames, PanoConfig(), _scripted_survey(weak_names={"f04.jpg"})
    )
    output = tmp_path / "Panos"
    write_pano_folders(groups, output, copy_files=False, write_pto=True, optimize=False)
    folders = sorted(path for path in output.iterdir() if path.is_dir())
    assert len(folders) == 2
    known = next(path for path in folders if path.name.startswith("known_pano"))
    guessed = next(path for path in folders if path.name.startswith("guessed_pano"))
    assert (guessed / "f00.jpg").is_symlink()
    assert (guessed / "f00.jpg").resolve() == frames[0].path.resolve()
    assert not (known / "b0.jpg").exists()
    projects = list(known.glob("*.pto"))
    assert len(projects) == 1
    document = projects[0].read_text(encoding="utf-8")
    assert 'n"g0.jpg"' in document
    assert "b0.jpg" not in document
    assert "c n0 N1 x5.0000 y6.0000 X7.0000 Y8.0000 t0" in document


def _prefer_tools(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    """Put a directory of stand-in Hugin tools first on ``PATH``."""
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")


def test_survey_scales_preview_control_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cpfind points on a half-size preview are doubled onto the original."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pto_gen = bin_dir / "pto_gen"
    cpfind = bin_dir / "cpfind"
    pto_gen.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                "args = sys.argv[1:]",
                "out = args[args.index('-o') + 1]",
                "images = args[args.index('-o') + 2:]",
                "lines = [f'i w100 h50 f0 n\"{image}\"' for image in images]",
                "Path(out).write_text('\\n'.join(lines))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cpfind.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                "args = sys.argv[1:]",
                "out = args[args.index('-o') + 1]",
                "text = Path(args[-1]).read_text()",
                "images = text.count('n\"')",
                "extra = []",
                "for index in range(images - 1):",
                "    for _unused in range(12):",
                "        extra.append(",
                "            f'c n{index} N{index + 1} x10 y10 X20 Y20 t0'",
                "        )",
                "Path(out).write_text(text + '\\n' + '\\n'.join(extra))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for tool in (pto_gen, cpfind):
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    _prefer_tools(monkeypatch, bin_dir)

    frames = [
        _frame(tmp_path, f"full{index}.jpg", float(index), 400.0, size=(200, 100))
        for index in range(3)
    ]
    surveyed = survey_control_points(frames, linearmatchlen=1, cp_edge=100)
    assert surveyed.counts[(0, 1)] == 12
    assert surveyed.counts[(1, 2)] == 12
    assert surveyed.points[0].x == pytest.approx(20.0)
    assert surveyed.points[0].y == pytest.approx(20.0)
    assert surveyed.points[0].x_right == pytest.approx(40.0)
    assert surveyed.points[0].y_right == pytest.approx(40.0)


def test_missing_hugin_tool_names_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool that is not on PATH tells the user to install hugin-tools."""
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="hugin-tools"):
        resolve_hugin_tool("cpfind")


def test_cpfind_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero cpfind exit becomes RuntimeError."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pto_gen = bin_dir / "pto_gen"
    cpfind = bin_dir / "cpfind"
    pto_gen.write_text(
        "#!/bin/sh\nexit 0\n",
        encoding="utf-8",
    )
    cpfind.write_text(
        "#!/bin/sh\necho 'no overlap model' >&2\nexit 2\n",
        encoding="utf-8",
    )
    for tool in (pto_gen, cpfind):
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    _prefer_tools(monkeypatch, bin_dir)
    frames = [
        _frame(tmp_path, "a.jpg", 0.0, 400.0),
        _frame(tmp_path, "b.jpg", 1.0, 400.0),
    ]
    with pytest.raises(RuntimeError, match="cpfind failed"):
        survey_control_points(frames, 1, 1600)


def test_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry run scores the burst and leaves the output directory absent."""
    frames = [
        _frame(tmp_path, f"d{index}.jpg", float(index), 400.0) for index in range(5)
    ]
    monkeypatch.setattr(
        pano_mod, "expand_input_directories", lambda *args, **kwargs: [tmp_path]
    )
    monkeypatch.setattr(
        pano_mod,
        "collect_photo_paths",
        lambda *args, **kwargs: [frame.path for frame in frames],
    )
    monkeypatch.setattr(
        pano_mod, "load_frames", lambda photos, sample, **kwargs: (frames, 1, 2)
    )
    sibling = tmp_path.parent / f"{tmp_path.name}-panos"
    result = find_action_panos(
        [str(tmp_path)],
        PanoConfig(dry_run=True),
        survey=_scripted_survey(),
    )
    assert not sibling.exists()
    assert result.output_dirs == [sibling]
    assert result.skipped_no_exif == 1
    assert result.skipped_undecodable == 2
    assert any(group.accepted and group.kind == "guessed" for group in result.groups)


def test_cli_help_lists_thresholds() -> None:
    """The command advertises the variance and overlap flags."""
    result = CliRunner().invoke(pano, ["--help"])
    assert result.exit_code == 0
    for flag in (
        "--marker-var",
        "--marker-count",
        "--discontinuity",
        "--split-points",
        "--boundary",
        "--stride",
        "--overlap-points",
        "--far-points",
        "--dry-run",
        "--stitch",
        "--no-stitch",
        "--crop",
        "--no-crop",
    ):
        assert flag in result.output
    assert "--output" not in result.output
    assert "-panos" in result.output


def test_cli_rejects_a_zero_marker_count() -> None:
    """--marker-count 0 is not a marker."""
    result = CliRunner().invoke(pano, ["--marker-count", "0", "photos"])
    assert result.exit_code != 0
    assert "marker-count" in result.output


def test_cli_dry_run_report_includes_control_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report shows link counts and does not create a sibling folder."""
    first = _frame(tmp_path, "a.jpg", 0.0, 12.0)
    second = _frame(tmp_path, "b.jpg", 0.4, 900.0)
    group = PanoGroup(
        kind="guessed",
        accepted=True,
        reason="neighbors overlap",
        frames=[first, second],
        number=1,
        neighbor_links=[LinkReport("a.jpg", "b.jpg", 12, 0.4)],
    )
    sibling = tmp_path.parent / f"{tmp_path.name}-panos"

    def fake_find(*args: object, **kwargs: object) -> PanoResult:
        config = args[1]
        assert isinstance(config, PanoConfig)
        assert config.dry_run is True
        return PanoResult(
            frames=[first, second],
            groups=[group],
            marker_runs=[(0, 0)],
            skipped_no_exif=0,
            skipped_undecodable=0,
        )

    monkeypatch.setattr(pano_mod, "find_action_panos", fake_find)
    result = CliRunner().invoke(pano, ["--dry-run", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "a.jpg --12cp 0.4s-- b.jpg" in result.output
    assert "dry run" in result.output
    assert "yes" in result.output
    assert not sibling.exists()


def test_completion_offers_pano() -> None:
    """Bash completion lists pano next to the other commands."""
    from sportball.cli.ultra_minimal_main import cli

    result = CliRunner().invoke(cli, ["completion", "--bash"])
    assert result.exit_code == 0
    assert "pano" in result.output


def test_validate_config_rejects_unknown_boundary() -> None:
    """An unknown cut rule fails before any files are touched."""
    with pytest.raises(ValueError, match="boundary"):
        detect_panos([], PanoConfig(boundary="middle"), _scripted_survey())


def test_copy_writes_real_files(tmp_path: Path) -> None:
    """--copy places file bytes instead of symlinks."""
    frames = [
        _frame(tmp_path, f"c{index}.jpg", float(index), 400.0) for index in range(5)
    ]
    groups, _runs = detect_panos(frames, PanoConfig(), _scripted_survey())
    output = tmp_path / "Panos"
    write_pano_folders(groups, output, copy_files=True, write_pto=False)
    folder = next(output.iterdir())
    placed = folder / "c0.jpg"
    assert placed.is_file()
    assert not placed.is_symlink()
    assert list(folder.glob("*.pto")) == []


def test_progress_names_the_burst_and_can_be_silenced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Progress lines name the burst. ``progress=False`` prints nothing."""
    frames = [
        _frame(tmp_path, f"v{index}.jpg", float(index), 400.0) for index in range(5)
    ]
    detect_panos(frames, PanoConfig(progress=True), _scripted_survey())
    noisy = capsys.readouterr().err
    assert "Unmarked burst v0.jpg .. v4.jpg" in noisy
    assert "keep guessed" in noisy
    assert "Blackness" in noisy or "Guessed panoramas" in noisy

    detect_panos(frames, PanoConfig(progress=False), _scripted_survey())
    quiet = capsys.readouterr().err
    assert quiet == ""


def test_anchor_index_is_the_earlier_middle_photo() -> None:
    """Five photos use the 3rd. Four photos use the 2nd."""
    assert anchor_index(5) == 2
    assert anchor_index(4) == 1
    assert anchor_index(1) == 0


def test_stitch_without_a_project_is_rejected() -> None:
    """Stitching is named after the .pto, so it needs that project."""
    with pytest.raises(ValueError, match="stitch"):
        detect_panos([], PanoConfig(stitch=True, write_pto=False), _scripted_survey())


def _install_fake_hugin(bin_dir: Path) -> None:
    """Write stand-ins that record positions-and-view and the stitch."""
    bin_dir.mkdir()
    scripts = {
        "pto_var": """#!/usr/bin/env python3
import sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("-o") + 1])
src = Path(args[-1])
out.write_text(src.read_text() + "\\n# PTO_VAR " + " ".join(args) + "\\n")
""",
        "autooptimiser": """#!/usr/bin/env python3
import sys
from pathlib import Path
args = sys.argv[1:]
banned = {"-a", "-m", "-l", "-s", "--photometric"}
if "-n" not in args or banned.intersection(args):
    raise SystemExit("unexpected optimiser flags: " + " ".join(args))
out = Path(args[args.index("-o") + 1])
src = Path(args[-1])
out.write_text(src.read_text() + "\\n# AUTO " + " ".join(args) + "\\n")
""",
        "hugin_executor": """#!/usr/bin/env python3
import sys
from pathlib import Path
args = sys.argv[1:]
if "--batch" in args or "-b" in args:
    raise SystemExit("batch was started")
if "--stitching" not in args:
    raise SystemExit("stitching was not requested")
prefix = next(
    (arg.split("=", 1)[1] for arg in args if arg.startswith("--prefix=")),
    None,
)
projects = [arg for arg in args if arg.endswith(".pto")]
if prefix is None or not projects:
    raise SystemExit("missing prefix or project")
project = Path(projects[0])
if "# AUTO" not in project.read_text(encoding="utf-8"):
    raise SystemExit("stitched before optimisation")
Path(prefix).with_suffix(".tif").write_bytes(b"stitched")
log = Path(__file__).resolve().with_name("stitched.txt")
with log.open("a", encoding="utf-8") as handle:
    handle.write("\\n".join(args) + "\\n")
""",
    }
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_correction_uses_the_median_and_stitch_matches_the_pto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anchor is AC, optimisation is y,p,r,v only, and the stitch is last.

    The output directory is relative, matching ``sb pano`` on a game folder.
    ``hugin_executor --stitching`` runs only after optimisation, without
    a batch flag, and the prefix matches the project name.
    """
    bin_dir = tmp_path / "bin"
    _install_fake_hugin(bin_dir)
    _prefer_tools(monkeypatch, bin_dir)
    monkeypatch.chdir(tmp_path)
    cases = ((5, 2), (4, 1))
    for count, anchor in cases:
        frames = [
            _frame(tmp_path, f"n{count}_{index}.jpg", float(index), 400.0)
            for index in range(count)
        ]
        group = PanoGroup(
            kind="guessed",
            accepted=True,
            reason="neighbors overlap",
            frames=frames,
            points=[ControlPoint(0, 1, 1.0, 2.0, 3.0, 4.0)],
            number=1,
        )
        output = Path(f"out{count}")
        write_pano_folders(
            [group],
            output,
            copy_files=False,
            write_pto=True,
            progress=False,
            stitch=True,
            optimize=True,
        )
        folder = next(path for path in output.iterdir() if path.is_dir())
        projects = list(folder.glob("*.pto"))
        assert len(projects) == 1
        project = projects[0]
        text = project.read_text(encoding="utf-8")
        assert f"--anchor={anchor}" in text
        assert f"--color-anchor={anchor}" in text
        assert "--opt=y,p,r,v" in text
        auto = text.split("# AUTO", 1)[1]
        assert " -n " in auto
        assert " -m " not in auto
        assert " -a " not in auto
        stitched = project.with_suffix(".tif")
        assert stitched.is_file()
        log = (bin_dir / "stitched.txt").read_text(encoding="utf-8")
        assert "--stitching" in log
        assert str(project.resolve()) in log
        assert f"--prefix={project.resolve().with_suffix('')}" in log
        assert "--batch" not in log
        assert "-b" not in log.split()
        assert list(folder.glob("*.marked.pto")) == []


def test_cli_stitch_uses_five_frame_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--stitch`` and the new minimum reach the detector config."""
    seen: Dict[str, PanoConfig] = {}

    def fake_find(*args: object, **kwargs: object) -> PanoResult:
        config = args[1]
        assert isinstance(config, PanoConfig)
        seen["config"] = config
        return PanoResult(
            frames=[],
            groups=[],
            marker_runs=[],
            skipped_no_exif=0,
            skipped_undecodable=0,
        )

    monkeypatch.setattr(pano_mod, "find_action_panos", fake_find)
    result = CliRunner().invoke(pano, ["--stitch", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert seen["config"].stitch is True
    assert seen["config"].crop is True
    assert seen["config"].min_frames == 5
    assert not (tmp_path.parent / f"{tmp_path.name}-panos").exists()


def _accepted_group(directory: Path, kind: str, name: str, offset: float) -> PanoGroup:
    """One accepted panorama whose photo lives in ``directory``."""
    return PanoGroup(
        kind=kind,
        accepted=True,
        reason="scripted",
        frames=[_frame(directory, name, offset, 400.0)],
    )


def _stub_detection(
    monkeypatch: pytest.MonkeyPatch,
    groups_by_dir: Dict[Path, List[PanoGroup]],
) -> None:
    """Serve scripted groups for each input without reading EXIF."""
    resolved = {path.resolve(): groups for path, groups in groups_by_dir.items()}

    def load_frames(
        photos: Sequence[Path],
        sample: int,
        progress: bool = True,
        marker_var: Optional[float] = None,
    ) -> Tuple[List[Frame], int, int]:
        if not photos:
            return [], 0, 0
        frames = [
            frame
            for group in resolved[photos[0].resolve().parent]
            for frame in group.frames
        ]
        return frames, 0, 0

    def detect(
        frames: Sequence[Frame],
        config: PanoConfig,
        survey: object,
    ) -> Tuple[List[PanoGroup], List[Tuple[int, int]]]:
        if not frames:
            return [], []
        groups = resolved[frames[0].path.resolve().parent]
        pano_mod._assign_numbers(groups)
        return list(groups), []

    monkeypatch.setattr(pano_mod, "load_frames", load_frames)
    monkeypatch.setattr(pano_mod, "detect_panos", detect)


def test_one_input_writes_a_panos_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A game folder gains a sibling named with ``-panos``."""
    game = tmp_path / "Game03_19Sep2026_120915-132501"
    game.mkdir()
    _stub_detection(
        monkeypatch,
        {game: [_accepted_group(game, "guessed", "a.jpg", 0.0)]},
    )
    result = find_action_panos(
        [str(game)],
        PanoConfig(write_pto=False, stitch=False, progress=False),
    )
    sibling = tmp_path / "Game03_19Sep2026_120915-132501-panos"
    assert [path.resolve() for path in result.output_dirs] == [sibling.resolve()]
    assert sibling.is_dir()
    assert not (tmp_path / "Panos").exists()
    folders = [path for path in sibling.iterdir() if path.is_dir()]
    assert len(folders) == 1
    assert folders[0].name.startswith("guessed_pano01_")
    link = folders[0] / "a.jpg"
    assert link.is_symlink()
    assert link.resolve() == (game / "a.jpg").resolve()


def test_five_inputs_each_get_a_panos_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five source folders stay paired with five ``-panos`` siblings."""
    grouped: Dict[Path, List[PanoGroup]] = {}
    games: List[Path] = []
    for index in range(1, 6):
        game = tmp_path / f"Game0{index}_19Sep2026_120915-132501"
        game.mkdir()
        games.append(game)
        grouped[game] = [
            _accepted_group(game, "known", "k.jpg", 0.0),
            _accepted_group(game, "guessed", "g.jpg", 20.0),
        ]
    _stub_detection(monkeypatch, grouped)
    result = find_action_panos(
        [str(game) for game in games],
        PanoConfig(write_pto=False, stitch=False, progress=False),
    )
    assert len(result.output_dirs) == 5
    assert not (tmp_path / "Panos").exists()
    for game in games:
        sibling = tmp_path / f"{game.name}-panos"
        assert sibling.resolve() in {path.resolve() for path in result.output_dirs}
        names = sorted(path.name for path in sibling.iterdir() if path.is_dir())
        assert len(names) == 2
        assert any(name.startswith("known_pano01_") for name in names)
        assert any(name.startswith("guessed_pano02_") for name in names)
        for folder in sibling.iterdir():
            if not folder.is_dir():
                continue
            for placed in folder.iterdir():
                assert placed.resolve().parent == game.resolve()


def test_dry_run_does_not_create_the_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run names the sibling and leaves it uncreated."""
    game = tmp_path / "Game03_19Sep2026_120915-132501"
    game.mkdir()
    _stub_detection(
        monkeypatch,
        {game: [_accepted_group(game, "guessed", "a.jpg", 0.0)]},
    )
    sibling = tmp_path / f"{game.name}-panos"
    result = find_action_panos(
        [str(game)],
        PanoConfig(dry_run=True, write_pto=False, stitch=False, progress=False),
    )
    assert not sibling.exists()
    assert [path.resolve() for path in result.output_dirs] == [sibling.resolve()]
    assert any(group.accepted for group in result.groups)


def test_glob_skips_an_existing_panos_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``*`` does not scan a ``-panos`` folder that belongs to another input."""
    game = tmp_path / "Game03_19Sep2026_120915-132501"
    game.mkdir()
    sibling = tmp_path / f"{game.name}-panos"
    sibling.mkdir()
    seen: List[Path] = []

    def collect(
        directories: Sequence[Path],
        pattern: str = "*",
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        seen.extend(directories)
        return []

    def load_frames(
        photos: Sequence[Path],
        sample: int,
        progress: bool = True,
        marker_var: Optional[float] = None,
    ) -> Tuple[List[Frame], int, int]:
        return [], 0, 0

    def detect(
        frames: Sequence[Frame],
        config: PanoConfig,
        survey: object,
    ) -> Tuple[List[PanoGroup], List[Tuple[int, int]]]:
        return [], []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pano_mod, "collect_photo_paths", collect)
    monkeypatch.setattr(pano_mod, "load_frames", load_frames)
    monkeypatch.setattr(pano_mod, "detect_panos", detect)
    result = find_action_panos(
        ["*"],
        PanoConfig(dry_run=True, write_pto=False, stitch=False, progress=False),
    )
    assert [path.resolve() for path in seen] == [game.resolve()]
    assert [path.resolve() for path in result.output_dirs] == [sibling.resolve()]


def test_cli_rejects_a_shared_output_flag(tmp_path: Path) -> None:
    """Panoramas are not collected into one ``-o`` directory."""
    result = CliRunner().invoke(pano, ["-o", "Panos", str(tmp_path)])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_stitch_runs_after_every_sibling_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stitching runs once, after each input has its own ``-panos`` folder."""
    bin_dir = tmp_path / "bin"
    _install_fake_hugin(bin_dir)
    _prefer_tools(monkeypatch, bin_dir)
    grouped: Dict[Path, List[PanoGroup]] = {}
    games: List[Path] = []
    for index in range(1, 3):
        game = tmp_path / f"Game0{index}_19Sep2026_120915-132501"
        game.mkdir()
        games.append(game)
        grouped[game] = [_accepted_group(game, "guessed", "a.jpg", float(index))]
    _stub_detection(monkeypatch, grouped)
    queued: List[List[Path]] = []

    def capture_queue(
        projects: Sequence[Path],
        progress: bool = True,
    ) -> None:
        for game in games:
            sibling = tmp_path / f"{game.name}-panos"
            assert sibling.is_dir()
            assert list(sibling.rglob("*.pto"))
        queued.append(list(projects))

    monkeypatch.setattr(pano_mod, "stitch_projects", capture_queue)
    find_action_panos(
        [str(game) for game in games],
        PanoConfig(stitch=True, write_pto=True, progress=False),
    )
    assert len(queued) == 1
    assert len(queued[0]) == 2
    assert {path.parent.parent.name for path in queued[0]} == {
        f"{game.name}-panos" for game in games
    }


def test_crop_black_canvas_drops_the_border(tmp_path: Path) -> None:
    """The cropped JPEG is the bounding box of the non-black pixels."""
    source = tmp_path / "guessed_pano01_20Sep2025_090012-090018.png"
    image = Image.new("RGB", (40, 30), (0, 0, 0))
    image.paste(Image.new("RGB", (10, 8), (20, 140, 60)), (5, 7))
    image.save(source)
    destination = tmp_path / "guessed_pano01_20Sep2025_090012-090018_cropped.jpg"

    with Image.open(source) as opened:
        assert content_box(opened) == (5, 7, 15, 15)
    box = crop_black_canvas(source, destination)

    assert box == (5, 7, 15, 15)
    with Image.open(destination) as cropped:
        assert cropped.size == (10, 8)
        pixel = cropped.getpixel((5, 4))
        assert isinstance(pixel, tuple)
        expected = (20, 140, 60)
        for channel, target in zip(pixel, expected):
            assert abs(int(channel) - target) <= 2


def test_crop_directory_prefers_jpeg_and_ignores_cropped_files(tmp_path: Path) -> None:
    """A loose JPEG wins over the TIFF in the folder. The crop is not recropped."""
    name = "guessed_pano02_20Sep2025_090012-090018"
    folder = tmp_path / name
    folder.mkdir()
    loose = tmp_path / f"{name}.jpg"
    inner = folder / f"{name}.tif"
    _canvas(loose, (80, 40), (10, 6, 30, 22), (200, 40, 40))
    _canvas(inner, (200, 100), (0, 0, 200, 100), (10, 10, 200))

    written = crop_panorama_directory(tmp_path, progress=False)
    again = crop_panorama_directory(tmp_path, progress=False)

    cropped = tmp_path / f"{name}_cropped.jpg"
    assert written == [cropped]
    assert again == [cropped]
    assert not (tmp_path / f"{name}_cropped_cropped.jpg").exists()
    with Image.open(cropped) as image:
        assert image.size[0] < 80
        assert abs(image.size[0] - 20) <= 2
        assert abs(image.size[1] - 16) <= 2


def test_all_black_stitch_is_not_cropped(tmp_path: Path) -> None:
    """A frame with no picture does not produce a cropped file."""
    source = tmp_path / "known_pano03_20Sep2025_090012-090018.jpg"
    Image.new("RGB", (12, 8), (0, 0, 0)).save(source, "JPEG")
    assert crop_black_canvas(source, tmp_path / "out.jpg") is None
    assert crop_panorama_directory(tmp_path, progress=False) == []


def test_cli_can_turn_stitch_and_crop_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-stitch`` and ``--no-crop`` are the opt-out flags."""
    seen: Dict[str, PanoConfig] = {}

    def fake_find(*args: object, **kwargs: object) -> PanoResult:
        config = args[1]
        assert isinstance(config, PanoConfig)
        seen["config"] = config
        return PanoResult(
            frames=[],
            groups=[],
            marker_runs=[],
            skipped_no_exif=0,
            skipped_undecodable=0,
        )

    monkeypatch.setattr(pano_mod, "find_action_panos", fake_find)
    result = CliRunner().invoke(
        pano, ["--no-stitch", "--no-crop", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert seen["config"].stitch is False
    assert seen["config"].crop is False


def _canvas(
    path: Path,
    size: Tuple[int, int],
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
) -> None:
    """Save a black frame with one solid rectangle. ``box`` is a crop box."""
    image = Image.new("RGB", size, (0, 0, 0))
    left, top, right, bottom = box
    image.paste(Image.new("RGB", (right - left, bottom - top), color), (left, top))
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, "JPEG", quality=100)
    else:
        image.save(path)
