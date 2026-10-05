# sportball

A Saturday of rec sports is a few thousand frames in one dump: warmup, game one, the gap while you packed up, game two, and whatever you kept shooting on the field after yours. A month of that is worse. Sorting it by hand is miserable.

**sportball** (`sb`) reads capture times from EXIF, finds the gaps between sessions, and builds one folder per game. Those folders are **symlinks** back to the originals. The dump stays where it is. Lightroom, Capture One, album generators, and `ffmpeg` can all work from the game folders without a second copy of the files.

`sportball` and `sb` are the same command.

## What you get

```
~/Pictures/2025/04_Apr/          ← originals stay here
~/Pictures/2025/05_May/
~/Pictures/2025/Games/           ← views into those dumps
    Game01_20Sep2025_090012-102348/
    Game02_20Sep2025_113005-124410/
    Game03_27Sep2025_091104-103822/
```

Each `Game##` folder looks like a normal album. The files inside it are links, not copies. Delete `Games/` and nothing in the dump is touched. Re-run `sb split` with a different gap and you get a new set of views, still pointing at the same files.

Use those folders to:

- Import a single game into Lightroom or Capture One
- Hand a game folder to an album / proofing / slideshow tool
- Encode a one-minute MP4 contact sheet of the game (`sb animate`)

Pass `--copy` only if you actually want a second set of files on disk.

## How the commands work together

The usual sequence is **look → split → use**.

```
month folders / card dumps
        │
        ▼
   sb analyze          dry run: albums, gaps, leftover clusters
        │
        ▼
   sb split            writes Games/ as symlink albums
        │
        ├──  editor / album tool
        ├──  sb animate          one MP4 per game
        └──  sb pano             action panoramas for Hugin
```

### 1. `sb analyze` — see the split before you commit

Reads EXIF, clusters shots by time, and prints the albums it *would* create. Nothing is written.

```bash
sb analyze 04_Apr
sb analyze 04_Apr 05_May
sb analyze --output SpringGames *
```

You get:

- A timeline of games and the breaks between them
- Photo count, duration, and shots-per-hour for each cluster
- Leftover clusters that did not make the cut (warmup dribbles, walking back to the car)
- A histogram of **seconds between consecutive shots**, so you can pick `--min-gap`

`--output` here is only so a previous `Games` folder is skipped when you pass `*`. Analyze does not write into it.

Tune on this pass. Then split with the same flags.

### 2. `sb split` — build the game albums

Same clustering as analyze, then it creates the folders.

```bash
sb split 04_Apr 05_May
sb split --output SpringGames 04_Apr 05_May
sb split --output SpringGames *
```

Output defaults to `./Games`. Games are numbered in capture-time order across every input directory, so April game 12 and May game 1 become `Game12` and `Game13` in one album, not two competing `Game01`s.

`*` is every directory in the current folder. The output directory itself is skipped, so a previous `Games` tree is not ingested again.

By default each photo in a game folder is a **symlink** to the original. `--copy` writes real files instead.

```bash
sb split --copy --output GamesBackup 04_Apr
```

If auto-split is wrong, put known break times in a text file (one timestamp per line) and pass `--split-file`.

### 3. `sb animate` — a quick movie of each game

Needs `ffmpeg` on `PATH`. Each `Game##_<date>_<start>-<end>` folder becomes a sibling `.mp4` next to that folder. Default is one minute per game (`FPS = photo_count / 60`).

```bash
sb animate Games
sb animate --duration 30 Games
sb animate --fps 12.5 Games
sb animate --duration 30 Game44_12Oct2025_144301-144303
sb animate --size 5568x3712 Games
sb animate --size 5568x Games
sb animate --size x1080 Games
```

`--size` is `WIDTHxHEIGHT`, `WIDTHx`, or `xHEIGHT`. `5568x3712` is an exact frame. `5568x` and `x1080` set one side and keep the photo aspect ratio (`x1080` is 1080p). Omit `--size` to keep the native photo resolution.

`--dry-run` prints the planned encodes. `--force` overwrites existing MP4s.

Because the game folders are symlinks, ffmpeg reads the originals. You are not encoding a duplicate tree.

### 4. `sb pano` — action panoramas for Hugin

Needs Hugin's `pto_gen`, `cpfind`, `pto_var`, `autooptimiser` (`hugin-tools`), and `hugin_executor` (the `hugin` package). Stitching is on by default and runs in process, with no batch window. Point it at a game folder or a dump of JPEGs. The Hugin project is not cropped or masked. You still choose the regions in Hugin.

Two kinds of folder, numbered together in capture order:

- `known_pano01_20Sep2025_090012-090018/` — two nearly uniform frames (lens covered, sky, or a zoom into the ground) after a good pan. The folder is the run before those frames, cut at the largest gap that has no control-point overlap.
- `guessed_pano02_20Sep2025_091440-091448/` — no marker. Neighbors overlap, and frames three apart do not, so the camera moved far enough to stitch.

Five shots in 2.5 seconds and five shots in 10 seconds are the same test. A gap longer than 10 seconds is two different shots. A panorama needs at least 5 photos. Marker frames are not copied into the folder. Each folder is symlinks plus a `.pto` whose control points were found on a preview and scaled back up.

Each input directory gets its own sibling folder. `Games/Game03_19Sep2026_120915-132501` writes `Games/Game03_19Sep2026_120915-132501-panos/`. Five input folders produce five `-panos` folders, so each game's panoramas stay next to that game. Numbering starts at 01 inside every sibling.

The median photo is the position and exposure anchor: the 3rd of 5, the 2nd of 4. Every project is then optimized for yaw, pitch, roll, and field of view (`y, p, r, v`) and nothing else. Stitching is the next step, unless you pass `--no-stitch`. After every sibling has been written, `hugin_executor` stitches each project in process. The image is a JPEG named like the `.pto` and lands in that panorama folder. Nothing is queued, and no batch window opens.

Cropping is also on unless you pass `--no-crop`. After the stitched JPEG exists (`guessed_pano01_20Sep2025_090012-090018.jpg`), Sportball writes `guessed_pano01_20Sep2025_090012-090018_cropped.jpg` next to the panorama folders. That file is the bounding box of everything that is not black canvas. The full stitch stays, so you can still see the black border and clean it up by hand.

```bash
sb pano Games/Game03_19Sep2026_120915-132501
sb pano Game01 Game02 Game03 Game04 Game05
sb pano --no-stitch --no-crop Games/Game03_19Sep2026_120915-132501
sb pano --dry-run Games/Game03_19Sep2026_120915-132501
sb pano --marker-var 80 --split-points 0 --overlap-points 15 Games
```

`--dry-run` prints a tqdm bar and a line per frame while it scores, then every frame's grayscale mean and variance and the control-point count on each link. Nothing is written. `--quiet` hides the progress. Use the report to set `--marker-var`, `--split-points`, `--overlap-points`, and `--far-points`. A marker is a frame whose pixels barely vary around their mean gray level, so a covered lens, a shot of sky, and a zoom into the ground all qualify.

## How a “game” is decided

Capture time comes from EXIF (`DateTimeOriginal`, then related capture tags) via [fast-exif-rs-py](https://github.com/dapperfu/fast-exif-rs-py). Filenames are ignored. Filesystem mtime is ignored. Photos without a usable EXIF datetime are skipped.

JPEG, HEIC, TIFF, and common RAW suffixes (CR2, NEF, ARW, DNG, RAF, ORF, RW2, and others) are included.

Then the shots are sorted and the gaps are inspected:

- A **gap of 10 minutes** (default) is a possible game boundary. `--min-gap` accepts fractions of a minute (`0.5` is 30 seconds). Use the analyze histogram to pick this.
- Halftime and water breaks are usually shorter than the gap between games, so they stay in the same folder. If a later gap is much larger, a smaller gap in the middle is left alone.
- A stretch only becomes a game if it has at least **10 photos** and shoots at least **100 photos per hour**. Twenty shots in 12 minutes is 100/hour and counts; the same 20 shots spread over an hour do not.
- `--min-photos` and `--min-rate` are mutually exclusive. Use `--min-photos` for short sports (wrestling: a 7-minute match with 10 frames still counts; rate is ignored). Use `--min-rate` when you only care about shooting density.

Folder names look like:

```
Game01_20Sep2025_090012-102348/
Game02_20Sep2025_113005-124410/
```

Date and start/end are from the first and last capture times in that cluster.

## Working from the symlink albums

Keep one canonical dump (card ingest, month folders, whatever you already use). Treat `Games/` as a derived index.

Most editors and album tools follow symlinks:

- Point Lightroom / Capture One at `Games/Game12_…` to work one match
- Point an album generator or slideshow at that same folder
- Delete or rebuild `Games/` at any time; the dump is unchanged

If a tool refuses to follow links, re-split that run with `--copy`, or work from the originals and use the game folder only as a file list.

On Windows, creating symlinks may require Developer Mode. Linux and macOS treat them as normal files.

## Knobs you will actually use

```bash
# only files matching a glob (default is *)
sb split -o Games 04_Apr --pattern "20250920_*"

# copy instead of symlink
sb split --copy 04_Apr 05_May

# force splits at known times
sb split --split-file splits.txt 04_Apr 05_May

# wrestling / short matches: absolute photo count, ignore hourly rate
sb split --min-gap 0.25 --min-photos 10 10-Oct

# field sports: hourly rate only (default is 100/h)
sb split --min-gap 0.25 --min-rate 100 04_Apr 05_May
```

`splits.txt` is one timestamp per line:

```
20250920_102500
20250920_124500
```

`YYYY-MM-DD HH:MM:SS` and `HH:MM:SS` also work.

`sb --help` and `sb split --help` list the rest.

## Install

Python 3.10+, a virtual environment, and Rust (EXIF reads go through [fast-exif-rs-py](https://github.com/dapperfu/fast-exif-rs-py); there is no fallback). `sb animate` also needs `ffmpeg`. `sb pano` also needs Hugin (`hugin-tools`: `pto_gen`, `cpfind`, `pto_var`, `autooptimiser`; stitching uses `hugin_executor` from the `hugin` package).

```bash
git clone <this-repo>
cd sportball_photography
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -e .
```

Same install via Make: `make install`.

After that, `sportball` and `sb` are on your PATH inside the venv.

## Rust port

A 1:1 native rewrite lives in [`rust/`](rust/README.md) (`sportball-rs` library + `sb-rs` CLI). It talks to [fast-exif-rs](https://github.com/dapperfu/fast-exif-rs) directly instead of the Python bindings.

```bash
cd rust
make test
make install
```

