"""Detector configuration with CLI-visible, serializable thresholds."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Rect = Tuple[int, int, int, int]


@dataclass
class ThermalBlobConfig:
    threshold: float = 30.0
    motion_threshold: float = 25.0
    use_motion_gate: bool = True
    min_area: int = 3
    max_area: int = 1200
    morph_open: int = 1
    morph_dilate: int = 1
    max_link_distance: float = 90.0
    max_gap_frames: int = 4
    min_track_lifetime: int = 3
    use_prediction: bool = True
    draw_valid_only: bool = True
    min_track_displacement: float = 12.0
    min_track_path_length: float = 18.0
    min_mean_speed: float = 0.8
    max_mean_speed: float = 120.0
    min_directionality: float = 0.15
    min_track_max_blob_area: int = 14
    min_track_mean_blob_area: float = 8.0
    max_detections_per_frame: int = 40
    background_frames: int = 200
    background_stride: int = 10
    background_percentile: float = 50.0
    background_recalibrate_interval: int = 1000
    background_recalibrate_frames: int = 200
    background_recalibrate_stride: int = 10
    background_recalibrate_blend: float = 0.5
    roi: Optional[Rect] = None
    exclude_zones: List[Rect] = field(default_factory=list)
    draw_inactive_tracks: bool = True
    trail_length: int = 0
    draw_roi: bool = True
    draw_exclude_zones: bool = True
