# CVC Bats Thermal Detection

Thermal video blob detection and simple track filtering for bat monitoring.

The current workflow is GUI-first: choose an input video, draw ROI/exclude
zones and counting geometry from video frames, run detection, and review CSV/JSON
outputs plus an optional annotated video.

## Repository layout

```text
.
|-- src/
|   |-- thermal_blob_detector.py  # detector CLI and processing logic
|   |-- gui.py                    # Tkinter parameter GUI
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
