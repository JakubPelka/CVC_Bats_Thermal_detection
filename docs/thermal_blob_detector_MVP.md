# Thermal Blob Detector MVP v2

Standalone test module for detecting and tracking bright moving blobs in tripod thermal video.

Scope:

- detect bright objects against a thermal background,
- create simple centroid tracks,
- export debug video,
- export CSV with track points,
- no line/AOI counting logic.

This version adds better diagnostics for the issue where very bright/fast objects may be detected but not drawn as confirmed tracks.

## Why bright bats may be missing as tracks

A bright object close to the camera can cross the frame very quickly. It may be visible for only 1-2 frames. In v1, the overlay mainly emphasized tracks that reached `--min-track-lifetime`, default 3 frames. That could make short but real events look like they were not tracked.

Other common causes:

- `--max-link-distance` too low for fast motion,
- `--max-area`, `--max-width`, or `--max-height` too low for close/blurred objects,
- `--morph-dilate` joining a bat with another hot region and making the component too large,
- `--motion-gate` too strict if enabled.

## Basic test

```bash
pip install -r requirements.txt

PYTHONPATH=src python -m thermal_blob_detector --input examples/sample.mp4 --output outputs/debug.mp4 --csv outputs/tracks.csv --show
```

## First diagnostic run

Shows raw accepted detections and temporary tracks before they are confirmed.

```bash
PYTHONPATH=src python -m thermal_blob_detector \
  --input examples/sample.mp4 \
  --output outputs/debug_diagnostic.mp4 \
  --csv outputs/tracks_diagnostic.csv \
  --draw-all-tracks \
  --min-track-lifetime 1 \
  --max-link-distance 120 \
  --max-area 1200 \
  --show
```

If the bright bats appear in this run, the detector sees them. The issue is then only filtering/confirmation/tracking thresholds.

## More conservative production-like run

```bash
PYTHONPATH=src python -m thermal_blob_detector \
  --input examples/sample.mp4 \
  --output outputs/debug_clean.mp4 \
  --csv outputs/tracks_clean.csv \
  --threshold 18 \
  --min-area 2 \
  --max-area 300 \
  --max-link-distance 45 \
  --min-track-lifetime 3

```

## Important parameters

| Parameter | Meaning | When to increase |
|---|---|---|
| `--max-link-distance` | Max distance between predicted/observed centroid and new detection | Fast bats break into new temporary tracks |
| `--min-track-lifetime` | How many detections before a track is treated as confirmed | Lower to 1-2 for short fast passes |
| `--max-area` | Largest accepted blob area | Close/bright bats are rejected |
| `--max-width`, `--max-height` | Largest accepted blob size | Motion blur creates elongated blobs |
| `--threshold` | Brightness above background | Too many weak/noisy detections |
| `--morph-dilate` | Expands detected blobs | If objects merge with hot background, lower to 0 |
| `--motion-gate` | Requires frame-to-frame movement | Use only after basic detection is stable |

## Current track drawing behavior

- Valid flight-like tracks are drawn by default.
- Use `--draw-all-tracks` to draw invalid/static/noisy tracks too.
- The tracker uses a simple velocity prediction by default.
- Use `--no-prediction` to return to pure nearest-neighbour tracking.
- Use `--hide-inactive-tracks` to hide inactive tracks.

## Integration note

The detector still produces detections with:

```python
as_cvc_detection()
```

This means it can later be used as an alternative detector backend for ComputerVisionCounter. Counting logic should remain outside this module.
