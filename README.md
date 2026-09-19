# sportball

I shoot a lot of rec / youth sports. After a Saturday you end up with one dump folder: a few thousand JPEGs from two or three games, plus warmups, plus the game after yours that you accidentally kept shooting. Sorting that by hand is miserable.

This repo exists to take that dump and put each game in its own folder.

It does a few other vision things (faces, YOLO, jersey colors, quality scores) but the thing I actually needed first was: **folder of images in, folders of games out**.

## How the split works

Each photo's capture time comes from EXIF (`DateTimeOriginal`, then related capture tags) via [fast-exif-rs-py](https://github.com/dapperfu/fast-exif-rs-py). Filenames are not used. Photos without a usable EXIF datetime are skipped. After that we sort and look at the gaps.

- A **gap** of about 10 minutes is treated as a possible game boundary.
- Halftime / water breaks are usually shorter than the gap between games, so they stay in the same folder. If a later gap is much larger, a smaller gap in the middle is left alone.
- A stretch only becomes a game if it lasts at least **30 minutes** and has at least **50 photos**. That keeps a 12-shot burst of kids posing from becoming "Game 7".

Output folders look like:

```
Game1_20Sep2025_090012-102348/
Game2_20Sep2025_113005-124410/
```

By default those folders are **symlinks** back to the originals. The dump stays put. Pass `--copy` if you want real copies.

If the auto-split is wrong, drop timestamps in a text file (one `YYYY-MM-DD HH:MM:SS` or `HH:MM:SS` per line) and pass `--split-file`.

## Install

Needs Python 3.8+, a venv, and Rust (sidecar writes go through `image-sidecar-rust`; no Python fallback).

```bash
git clone <this-repo>
cd sportball_photography
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -e .
```

GPU (optional):

```bash
venv/bin/pip install -e ".[cuda]"
```

Same thing via Make: `make install` or `make install-cuda`.

Entry points after install: `sportball` and `sb`.

## Split a folder of images into game folders

Preview first (no folders created):

```bash
sportball games split /path/to/dump /path/to/games --analyze-only
```

Do the split:

```bash
sportball games split /path/to/dump /path/to/games
```

Useful knobs:

```bash
# only files matching a glob (default is *)
sportball games split /path/to/dump /path/to/games --pattern "20250920_*"

# copy instead of symlink
sportball games split /path/to/dump /path/to/games --copy

# force splits at known times
sportball games split /path/to/dump /path/to/games --split-file splits.txt

# if your games are short / sparse
sportball games split /path/to/dump /path/to/games --min-duration 20 --min-gap 8 --min-photos 30

# after games are already time-split, further split by jersey color
sportball games split /path/to/dump /path/to/games --split-by-jersey
```

`splits.txt` is just:

```
20250920_102500
20250920_124500
```

Python, if you want it:

```python
from pathlib import Path
from sportball import SportballCore

core = SportballCore()
print(core.detect_games(Path("/path/to/dump")))
```

The CLI is what actually builds the output folders.

## Other commands that exist

These work to varying degrees. I use them after the dump is already split.

```bash
sportball face detect /path/to/game
sportball object detect /path/to/game --classes "person,sports ball"
sportball quality assess /path/to/game
sportball util sidecar-summary /path/to/game
```

`sportball --help` and `sportball games split --help` have the rest.

## Layout

- `sportball/` — the package and CLI
- `tests/` — pytest
- `development/` — old standalone scripts from before this was a package. Don't start there.
