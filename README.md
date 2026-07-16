# CVC Bats Thermal Detection

Thermal video blob detection and simple track filtering for bat monitoring.

The current workflow is GUI-first: choose an input video, draw ROI/exclude
zones and counting geometry from video frames, run detection, and review CSV/JSON
outputs plus an optional annotated video.

## Repository layout

```text
.
|-- src/
|   |-- thermal_blob_detector.py  # compatibility facade and module entry point
|   |-- gui.py                    # Tkinter parameter GUI
|   |-- thermal_bat/              # detector package
|   |   |-- cli.py                # command-line parser
|   |   |-- pipeline.py           # single-video and batch orchestration
|   |   |-- detector.py           # background, blob detection and tracking
|   |   |-- live_counting.py      # streaming line/AOI counting
|   |   |-- exports.py            # detector-specific CSV exports
|   |   |-- visualization.py      # preview and event-clip overlays
|   |   `-- models.py             # shared track/detection models
|   |-- counting_models.py        # counting/event data models
|   |-- counting_geometry.py      # pure line/AOI geometry
|   |-- gui_command.py            # Tk-independent CLI command builder
|   |-- thermal_blob_detector_mvp_v3_valid_tracks.py  # legacy CLI wrapper
|   `-- thermal_blob_detector_gui.py                  # legacy GUI wrapper
|-- docs/                         # detailed usage notes
|-- examples/                     # sample input video and counting config examples
|-- presets/                      # loadable GUI parameter presets
`-- outputs/                      # sample generated outputs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the two console commands shown below. If you change
`pyproject.toml`, run `pip install -e .` again so the editable install is
refreshed.

The built-in detector and GUI defaults follow `presets/working_detect.json`.
The preset browser opens the repository `presets/` directory by default.

For a no-install quick setup, install only the runtime requirements and use the
`python src/...` or `PYTHONPATH=src python -m ...` commands:

```bash
pip install -r requirements.txt
```

## Run the detector

Installed command, after `pip install -e .`:

```bash
thermal-blob-detector \
  --input examples/sample.mp4 \
  --output outputs/thermal_blob_valid_tracks.mp4 \
  --csv outputs/thermal_blob_track_points.csv \
  --summary-csv outputs/thermal_blob_track_summary.csv \
  --crossings-csv outputs/crossings.csv \
  --aoi-events-csv outputs/aoi_events.csv \
  --activity-csv outputs/activity_by_time.csv \
  --run-summary-json outputs/run_summary.json
```

Module form:

```bash
PYTHONPATH=src python -m thermal_blob_detector --input examples/sample.mp4
```

Direct source wrapper:

```bash
python src/thermal_blob_detector_mvp_v3_valid_tracks.py --input examples/sample.mp4
```

For fastest batch processing, omit `--show`. Passing `--output ""` disables
annotated video writing.

For maximum analysis throughput, use `--output ""` without `--show`. This
skips all per-frame track, HUD, and counting-geometry drawing while preserving
the final CSV/JSON analysis and optional post-run event clips. For a long debug
video, `--hide-inactive-tracks --trail-length 120` keeps overlay work bounded.
Using `--trail-length 0` draws full track histories and becomes slower as long
videos accumulate tracks.

The live overlay keeps an incremental index of valid tracks, so normal preview
and annotated-video rendering no longer revalidates every historical rejected
track on every frame. Periodic background recalibration still causes expected
short FPS dips because it samples and rebuilds the background model.
Progress logs report both cumulative FPS and `recent` FPS over the latest 100
frames. The recent value makes recalibration pauses distinguishable from actual
per-frame slowdown.

Tracks made exclusively from tiny noise blobs can be rejected with
`--min-track-max-blob-area`. Unlike an exclusion zone, this filter does not
depend on where activity occurs: at least one detection in the track must reach
the configured area. `0` disables it; for the analyzed `Drzewo.mp4` recording,
`14` is a useful starting value for the maximum-area requirement.

`--min-track-mean-blob-area` checks the average detection area over the whole
track. It is less sensitive to a single enlarged noise blob and is useful for
rejecting tracks assembled from tiny moving foliage or compression artefacts.
`0` disables it; `8` is the recommended starting value for `Drzewo.mp4`.
Bounding-box width and height filters were replaced by these area-based track
filters because small thermal targets may legitimately be narrow or streaked.
The former width/height CLI options remain accepted only for compatibility with
old scripts and no longer affect detection.

The two track-area filters should normally use different thresholds. The mean
area checks whether the object remains sufficiently large through the track,
while the maximum area requires at least one stronger detection. Recommended
starting profiles are:

| Target size | Mean track blob area | Max track blob area | Trade-off |
|---|---:|---:|---|
| Current `Drzewo.mp4` bats | 8 | 14 | Rejects the moving-branch tracks while retaining the bat at frame 44390 |
| Smaller or more distant bats | 6 | 10 | More sensitive, with a moderate increase in tiny-noise tracks |
| Very small experimental targets | 4 | 8 | High sensitivity; expect more foliage and compression false positives |

As a rule, lower the mean threshold first and keep the maximum threshold about
1.5-2 times higher. Change one step at a time and review event clips. If an
object itself contains fewer than the detector-level `--min-area 3` pixels,
that earlier threshold must also be lowered, usually to `2`; the track-area
filters cannot restore detections rejected before tracking.

To analyze only an inclusive source-frame range, use for example:

```bash
thermal-blob-detector \
  --input input/Drzewo.mp4 \
  --start-frame 42000 \
  --end-frame 46000
```

`--end-frame 0` means the end of the video. Frame numbers stored in tracks,
CSV files and event-clip manifests remain absolute source-video frame numbers.
An exported MP4 necessarily has its own local frame numbering beginning at 1,
but the event-clip HUD shows the current absolute `Source frame`, the absolute
source window and the local clip-frame position. Manifests expose both the
existing `start_frame`/`end_frame` fields and explicit
`source_start_frame`/`source_end_frame` aliases.

### Event clips

For long videos, a full annotated output video can become large. Event clips
export only short annotated sections around detected activity:

```bash
thermal-blob-detector \
  --input input.mp4 \
  --output "" \
  --event-clips \
  --event-clips-dir outputs/event_clips \
  --event-clip-pre-frames 100 \
  --event-clip-post-frames 100 \
  --event-clip-merge-gap-frames 100
```

The full analysis runs first. The input is then re-opened and clips are drawn
from the stored track detections. Overlapping or nearby activity windows are
merged, so simultaneous tracks produce one clip rather than duplicates. The
directory also receives `event_clips_manifest.csv` and
`event_clips_manifest.json`. Both manifests include `start_date`, `start_time`,
`end_date`, and `end_time` for every clip. The source recording start is read
from its `creation_time` metadata or, as a fallback, from an unambiguous date
and time in the source filename. These fields remain empty when neither source
provides a reliable timestamp.

The default trigger is `valid_tracks`. Use `--event-clip-trigger` with
`all_tracks`, `crossings`, `aois`, or `all_events` to select other completed
analysis results. Event clips are disabled unless `--event-clips` is supplied,
and are independent of the normal `--output` annotated video.

When `--event-clips-dir` is omitted, clips use
`outputs/<video_stem>/<video_stem>_event_clips/`. A custom directory is used
exactly as supplied; avoid sharing one custom directory across a batch because
clip filenames can collide.

### Annotation styles

The live preview, full annotated video, and event clips use the same annotation
style. `bbox-trail` is the default and recommended QA view: it draws a colored
box enclosing the trajectory accumulated so far, plus a bright padded box
around the object detected in the current frame. Its trail can be disabled to
avoid drawing over the object.

Fast visual QA:

```bash
--annotation-style bbox-trail --track-line-thickness 1 --bbox-padding 4
```

Do not cover the object:

```bash
--annotation-style bbox
```

Minimal preview:

```bash
--annotation-style minimal
```

Other choices are `trail` (the original trail view), `thin-trail`, and `dot`.
Use `--no-show-track-id` to hide labels. Box, trail, and point dimensions can
also be adjusted with `--bbox-thickness`, `--track-line-thickness`, and
`--current-point-radius`. Set `--track-line-thickness 0` to disable the trail
entirely, including in `thin-trail` and `bbox-trail` modes.

Track annotations use stable per-track colors by default. To draw every track,
box, point, and ID in one color, select `fixed` and one of the 16 GUI palette
colors, or use for example:

```bash
--track-color-mode fixed \
--track-fixed-color cyan
```

#### Verification mode

Verification mode renders each event-clip frame twice, side by side, so two
annotation styles can be reviewed in one pass. For example, compare the full
track box on the left with the unannotated source frame on the right:

```bash
--event-clips \
--verification-mode \
--verification-left-style bbox-trail \
--verification-right-style raw \
--track-line-thickness 0
```

The two panels contain the same source frame. Verification mode affects event
clips only. The `raw` style contains no tracks, geometry, HUD, or panel label.
Live preview and the full annotated video keep using
`--annotation-style`.

### Batch processing

By default, batch results are written to an `outputs` folder beside the input
recordings (or inside the folder passed with `--input-dir`). Use
`--batch-output-dir` only when a different location is wanted.

Process an explicit list of videos:

```bash
thermal-blob-detector \
  --inputs data/night_001.mp4 data/night_002.mp4 \
  --batch-output-dir outputs \
  --event-clips
```

Or scan a folder and its subfolders using the configured video extensions:

```bash
thermal-blob-detector \
  --input-dir data/thermal_recordings \
  --recursive \
  --video-extensions .mp4,.avi,.mov,.mkv \
  --batch-output-dir outputs \
  --event-clips \
  --event-clip-pre-frames 100 \
  --event-clip-post-frames 100
```

Exactly one of `--input`, `--inputs`, or `--input-dir` is required. Batch mode
creates `outputs/<video_stem>/` for each input, using stem-prefixed CSV, JSON,
and optional annotated-video filenames. It also writes `batch_summary.csv` and
`batch_summary.json` in the batch output root. Use `--skip-existing` to skip a
video when its derived run summary exists, and `--continue-on-error` to record a
failure and continue with later inputs. The normal annotated video remains
independent from event clips: pass `--output ""` to disable full annotated
videos for a batch while still using `--event-clips`.

### Periodic background recalibration

Long thermal recordings can drift in apparent temperature or contrast. Enable
periodic recalibration when detections degrade over time; it is disabled by
default, so existing runs keep the original static background behaviour.

```bash
thermal-blob-detector --input recording.mp4 \
  --background-recalibrate-interval 2000 \
  --background-recalibrate-frames 200 \
  --background-recalibrate-stride 10 \
  --background-recalibrate-blend 0.7
```

Use blend `1.0` for sudden temperature/contrast changes or a gentler value such
as `0.2` for slow thermal drift. The terminal and preview HUD show frame count,
percentage, elapsed time, processing FPS and ETA when the video frame count is
available.

## Track-based counting and statistics

Counting uses tracks, not raw bright blobs. During processing the annotated
video/live preview draws counting lines and AOIs and shows a HUD with cumulative
line crossings, direction counts, AOI seen counts, and active AOI occupancy.
Line/AOI event CSV rows are streamed to disk during processing and final summary
files are written after the run.

By default only valid flying tracks are counted. Use `--count-all-tracks` only
for diagnostics.

Current counting behavior:

- Each track is counted at most once per counting line.
- Line events store direction labels such as `left_to_right` and `right_to_left`.
- AOI `seen` counts unique tracks that entered or started inside an AOI.
- AOI `in` in the HUD counts only currently active tracks whose latest point is
  inside the AOI.
- AOI exit rows include `start_frame`, `end_frame`, and `dwell_time_s` when an
  exit is observed.

Example with one vertical counting line and one rectangular AOI:

```bash
PYTHONPATH=src python -m thermal_blob_detector \
  --input examples/sample.mp4 \
  --output outputs/sample_valid_tracks.mp4 \
  --csv outputs/sample_track_points.csv \
  --summary-csv outputs/sample_track_summary.csv \
  --count-line midline,Middle,640,0,640,720,right_to_left,left_to_right \
  --count-aoi roost,Roost,100,100,250,180 \
  --crossings-csv outputs/sample_crossings.csv \
  --aoi-events-csv outputs/sample_aoi_events.csv \
  --activity-csv outputs/sample_activity_by_time.csv \
  --run-summary-json outputs/sample_run_summary.json
```

The GUI writes counting geometry to JSON and passes it with
`--counting-config`. You can also run that manually:

```bash
PYTHONPATH=src python -m thermal_blob_detector \
  --input examples/sample.mp4 \
  --counting-config examples/counting_config_example.json
```

The main statistics outputs are:

- `crossings.csv`
- `aoi_events.csv`
- `activity_by_time.csv`
- enhanced `track_summary.csv`
- `run_summary.json`

## Run the GUI

Quick launcher from the repository root:

```bash
./start.sh
```

Portable Python launcher:

```bash
python start.py
```

Installed command, after `pip install -e .`:

```bash
thermal-blob-detector-gui
```

Module form:

```bash
PYTHONPATH=src python -m gui
```

Direct source wrapper:

```bash
python src/thermal_blob_detector_gui.py
```

### GUI workflow

1. Choose an input video.
2. Use **ROI / exclude** to draw the detection ROI and exclude zones.
3. Use **Counting / Statistics** to draw counting lines and AOI polygons.
4. Keep **Preview window** on for live visual checks, or turn it off for faster
   processing.
5. Keep **Save annotated video** on when you need visual verification after the
   run. Turn it off for fastest batch processing.
6. Run the detector and review generated CSV/JSON files.

The GUI no longer exposes manual text boxes for ROI, exclude zones, counting
lines, or AOIs. These are drawn from video frames. Counting JSON paths are stored
as absolute paths so the detector can find them regardless of process working
directory.

### Presets

Loadable GUI presets live in `presets/` and can be opened with **Load preset
JSON**:

- `good_current_defaults.json`
- `loose_rescue_more_bats.json`
- `strict_remove_artefacts.json`
- `diagnostic_all_tracks.json`
- `quick_500_frame_test.json`
- `full_video.json`

## Documentation

- [Detector MVP notes](docs/thermal_blob_detector_MVP.md)
- [GUI notes](docs/thermal_blob_detector_GUI.md)
