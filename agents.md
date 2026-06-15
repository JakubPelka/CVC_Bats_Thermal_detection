# AGENTS.md — CVC_Bats_Thermal_detection

## Project context

This repository is a dedicated proof-of-concept tool for detecting, tracking, counting and summarising bat activity in thermal video recordings.

Current project decision:

- Keep this as a **dedicated thermal bat tool** for now.
- Do **not** integrate directly into `ComputerVisionCounter_video` yet.
- Keep the internal data structures reasonably compatible with future CVC-style integration.
- The current working core is: thermal bright-blob detection → simple tracking → valid-track filtering → GUI parameter tuning → ROI/exclude-zone drawing.
- The next development goal is: **statistics and counting**.

This tool should be treated as a specialised thermal-motion tracker, not as a YOLO/object-detection app.

---

## Current working detection assumptions

The sample data appears to work with the current approach:

```text
static thermal video from tripod
bright moving bat-like blobs
dark/low-contrast background
many short and crossing trajectories
some thermal/camera artefacts
```

Detection is based on:

```text
thermal frame
→ static/median background model
→ positive brightness difference
→ threshold
→ morphology
→ connected components
→ centroid tracking
→ track validation
```

The detector should continue to work without model training.

---

## Parameter values that worked during manual testing

These are not universal final values, but they gave a useful detection/tracking level on the tested sample video.

Use them as the default `normal` preset unless later tests show better values.

```yaml
threshold: 18.0
motion_gate: false
motion_threshold: 5.0

min_area: 2
max_area: 1200
min_width: 1
min_height: 1
max_width: 80
max_height: 80

morph_open: 1
morph_dilate: 1

max_link_distance: 90.0
max_gap_frames: 4
min_track_lifetime: 3
use_prediction: true

min_track_displacement: 12.0
min_track_path_length: 18.0
min_mean_speed: 0.8
max_mean_speed: 120.0
min_directionality: 0.15
max_detections_per_frame: 40

background_frames: 200
background_stride: 10
background_percentile: 50.0

trail_length: 0  # full visible track
draw_valid_tracks_only: true
```

Important notes from testing:

- `max_link_distance = 90` helped recover more correct tracks.
- `max_area = 1200` helped keep bright/close/blurred bats that were previously rejected.
- Some weak visible bats were detected surprisingly well.
- Some camera artefacts were also detected, so track-level validation is important.
- The tool should keep presets such as:
  - `normal`
  - `loose / rescue more bats`
  - `strict / remove artefacts`
  - `diagnostic / draw all tracks`

Suggested loose preset:

```yaml
min_track_lifetime: 2
min_track_displacement: 6.0
min_track_path_length: 10.0
min_mean_speed: 0.4
min_directionality: 0.05
max_detections_per_frame: 80
max_link_distance: 110.0
max_gap_frames: 6
```

Suggested strict preset:

```yaml
min_track_lifetime: 4
min_track_displacement: 18.0
min_track_path_length: 30.0
min_mean_speed: 1.2
min_directionality: 0.25
max_detections_per_frame: 30
max_link_distance: 80.0
max_gap_frames: 3
```

---

## Main task: add statistics and counting

Implement a statistics/counting layer on top of the existing valid tracks.

Counting must be **track-based**, not raw-detection-based.

Do not count every bright blob. Count validated tracks and their spatial events.

---

## Functional goals

### 1. Line crossing counting

Add support for one or more counting lines.

Each line should have:

```yaml
id: string
name: string
p1: [x, y]
p2: [x, y]
direction_labels:
  positive: string
  negative: string
enabled: true
```

Counting logic:

- Use track centroid history.
- For each consecutive pair of points in a track, check whether the segment crosses the counting line.
- Count a given `track_id` only once per `line_id` per crossing event.
- Store direction.
- Store crossing frame and time.
- Do not use raw detections for final counts.
- Prefer valid tracks only by default.
- Allow diagnostic option to count all tracks.

Recommended robust method:

```text
side_before = sign(point_before relative to line)
side_after  = sign(point_after relative to line)

if side_before != side_after:
    crossing detected
```

Then direction can be derived from the side change.

If a point lies exactly on the line, handle it with a small epsilon to avoid duplicate/noisy crossings.

Suggested options:

```yaml
line_crossing_epsilon: 1.0
min_frames_between_same_line_crossing: 3
count_valid_tracks_only: true
```

### 2. AOI entry/exit counting

Add support for one or more AOIs.

Start with rectangular AOIs if that matches the current GUI structure. Polygon AOIs can be added later.

Each AOI should have:

```yaml
id: string
name: string
type: rectangle  # later polygon
coordinates: [x, y, w, h]
enabled: true
```

Counting logic:

```text
inside_before = point_in_aoi(previous_centroid)
inside_after  = point_in_aoi(current_centroid)

False → True  = entry
True  → False = exit
```

Store:

```text
event_id
track_id
aoi_id
aoi_name
event_type: entry/exit
frame
time_s
cx
cy
```

Avoid multiple duplicate events for the same track/AOI caused by jitter near the boundary.

Suggested debounce option:

```yaml
aoi_boundary_debounce_frames: 3
```

### 3. Activity over time

Add activity statistics per time bin.

Default bin size:

```yaml
activity_bin_seconds: 60
```

Generate at least:

```text
time_bin_start_s
time_bin_end_s
valid_track_count_started
valid_track_count_active
line_crossings_total
line_crossings_by_line
line_crossings_by_direction
aoi_entries_total
aoi_exits_total
```

Useful later:

```text
peak_activity_bin
peak_activity_count
activity per minute
activity per 5 minutes
```

### 4. Track summary statistics

For every track, summarise:

```text
track_id
valid
first_frame
last_frame
start_time_s
end_time_s
lifetime_frames
duration_s
start_x
start_y
end_x
end_y
net_displacement_px
path_length_px
mean_speed_px_per_frame
mean_speed_px_per_second
directionality
max_blob_area
mean_blob_area
max_score
mean_score
crossing_count
aoi_entry_count
aoi_exit_count
```

### 5. Run summary

Create a compact run summary, ideally JSON and/or CSV:

```text
input_video
fps
frame_count_processed
parameter_preset
total_tracks
valid_tracks
invalid_tracks
total_line_crossings
crossings_by_line_and_direction
total_aoi_entries
total_aoi_exits
activity_bin_seconds
peak_activity_bin
notes
```

---

## Expected output files

Add these outputs:

```text
track_points.csv          # existing or current equivalent
track_summary.csv         # existing/enhanced
crossings.csv             # new
aoi_events.csv            # new
activity_by_time.csv      # new
run_summary.json          # new
```

Optional but useful:

```text
run_summary.csv
counting_config.json
annotated_counting_video.mp4
```

---

## GUI goals

Extend the existing Tkinter GUI, but avoid making it too complex in one step.

Add a new tab:

```text
Counting / Statistics
```

The tab should allow:

- enable/disable line counting
- enable/disable AOI counting
- set activity bin size
- choose whether to count:
  - valid tracks only
  - all tracks for diagnostics
- define counting lines
- define AOIs
- save/load counting configuration as JSON
- run detector + statistics in one workflow

### Drawing interaction

The current GUI already supports drawing rectangular ROI/exclude zones from a video frame.

Extend the same idea for counting geometry:

#### Counting lines

Allow drawing a line with mouse:

```text
click/drag from point A to point B
```

Store:

```text
x1,y1,x2,y2
```

Allow user to name the line.

#### AOIs

Start with rectangular AOIs if easier.

Later polygon AOIs can be added, but do not block the first implementation on polygon support.

---

## Suggested code structure

If the current project is still a single-file MVP, do not over-refactor everything at once.

Recommended direction:

```text
src/
├─ detector/
│  ├─ thermal_blob_detector.py
│  └─ track_validation.py
│
├─ counting/
│  ├─ line_crossing.py
│  ├─ aoi_events.py
│  └─ geometry.py
│
├─ stats/
│  ├─ track_stats.py
│  └─ activity_stats.py
│
├─ export/
│  ├─ csv_export.py
│  └─ json_export.py
│
├─ gui/
│  └─ thermal_blob_detector_gui.py
│
└─ app.py
```

But for the next practical step, it is acceptable to add:

```text
counting_stats.py
```

and later refactor.

---

## Recommended implementation order

### Step 1 — counting module without GUI

Create pure Python functions/classes that operate on tracks.

Do not start with GUI.

Implement and test:

```text
line crossing from track points
AOI entry/exit from track points
track summary metrics
activity bins
CSV/JSON export
```

### Step 2 — synthetic tests

Create small synthetic track examples:

```text
track crossing line left→right
track crossing line right→left
track moving near line but not crossing
track crossing same line once
track entering AOI
track exiting AOI
track jittering at AOI border
```

Expected results must be clear and deterministic.

### Step 3 — connect to current detector output

Use existing in-memory tracks or exported `track_points.csv`.

Preferred design:

- allow analysis from in-memory tracks after detection,
- also allow analysis from CSV later.

### Step 4 — add GUI tab

Only after the counting code works without GUI.

### Step 5 — annotated video

Draw counting lines/AOIs and event markers on output video.

This is useful but not required for the first statistics implementation.

---

## Data model suggestion

Use dataclasses.

Example:

```python
@dataclass
class CountingLine:
    id: str
    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    enabled: bool = True
    positive_label: str = "positive"
    negative_label: str = "negative"
```

```python
@dataclass
class CrossingEvent:
    event_id: str
    track_id: int
    line_id: str
    line_name: str
    direction: str
    frame: int
    time_s: float
    cx: float
    cy: float
```

```python
@dataclass
class AoiEvent:
    event_id: str
    track_id: int
    aoi_id: str
    aoi_name: str
    event_type: str  # entry / exit
    frame: int
    time_s: float
    cx: float
    cy: float
```

---

## Acceptance criteria

A first acceptable implementation should:

- keep the project as a dedicated thermal bat tool,
- not integrate with `ComputerVisionCounter_video`,
- preserve existing detector behaviour,
- count only valid tracks by default,
- produce `crossings.csv`,
- produce `aoi_events.csv`,
- produce `activity_by_time.csv`,
- produce enhanced `track_summary.csv`,
- produce `run_summary.json`,
- support at least one counting line,
- support at least one rectangular AOI,
- avoid duplicate counts from jitter,
- include at least basic synthetic tests for geometry/counting logic,
- document the new workflow in README.

---

## Repository hygiene

Do not commit large sample videos or generated output videos.

Keep ignored:

```gitignore
samples/*
!samples/README.md

output/*
!output/README.md

*.mp4
*.avi
*.mov
*.mkv
*.csv
*.json
```

Exception:

- Small example config JSON files may be kept in `presets/` or `examples/`.
- Synthetic tiny CSV test files may be kept under `tests/data/`.

Do not commit private or sensitive wildlife monitoring locations unless deliberately anonymised.

---

## Notes for future CVC integration

Do not integrate now.

However, keep exports compatible with possible future CVC usage:

```text
frame_idx
time_s
track_id
class_name = bat_candidate
cx
cy
bbox
score
valid_track
```

Possible future architecture:

```text
CVC_video detector backend:
YOLO / ThermalBlob / other
```

But this repository remains the active development sandbox for the thermal bat workflow until the method stabilises.

---

## Developer style

- Prefer small, testable functions.
- Do not hide important thresholds in code only; expose them in config/CLI/GUI.
- Keep GUI as a launcher/control layer, not as the only place where logic exists.
- Avoid hard-coding sample-specific coordinates.
- Preserve existing working defaults.
- Make outputs easy to inspect in QGIS, Excel, Python and ordinary text editors.
- Use clear CSV column names.
- Keep comments practical and concise.
