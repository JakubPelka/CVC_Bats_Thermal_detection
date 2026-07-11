#!/usr/bin/env python3
"""
thermal_blob_detector.py

Standalone MVP for detecting and tracking bright moving blobs in tripod thermal video.

Scope:
- Detect bright moving thermal blob candidates.
- Track centroid movement across frames.
- Filter tracks by flight-like movement metrics to reduce camera artefacts.
- Draw only thin track trails by default.
- Export CSV with detections, track metrics and validity flags.
- Track-based line/AOI counting and activity statistics run after tracking.

Typical use:
    python -m thermal_blob_detector --input examples/sample.mp4 --output outputs/tracks.mp4 --csv outputs/tracks.csv --show

Useful debug use:
    python -m thermal_blob_detector --input examples/sample.mp4 --draw-all-tracks --show
"""

from counting_models import AoiEvent, CountingAoi, CountingConfig, CrossingEvent
from event_clips import ClipWindow
from thermal_bat.config import Rect, ThermalBlobConfig
from thermal_bat.models import BBox, BlobDetection, Point, Track, TrackMetrics
from thermal_bat.validation import euclidean_distance, is_valid_flying_track, track_metrics
from thermal_bat.detector import ThermalBlobDetector, blend_background, point_in_rects
from thermal_bat.progress import build_progress_text, format_duration
from thermal_bat.live_counting import LiveCounting
from thermal_bat.visualization import (
    OverlayRenderer,
    color_for_track,
    draw_counting_geometry,
    draw_counting_hud,
    draw_debug_overlay,
    draw_event_clip_overlay,
    should_draw_track,
    tuple_int,
)
from thermal_bat.exports import write_track_points_csv, write_track_summary_csv
from thermal_bat.pipeline import (
    build_counting_config_from_args, parse_rect, parse_rect_list,
    process_batch, process_single_video, process_video,
)
from thermal_bat.cli import build_arg_parser, main


if __name__ == "__main__":
    main()
