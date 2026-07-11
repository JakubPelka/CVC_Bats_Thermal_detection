"""Data models for counting configuration, events, and summaries."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]


@dataclass
class CountingLine:
    id: str
    name: str
    p1: Point
    p2: Point
    positive_label: str = "positive"
    negative_label: str = "negative"
    enabled: bool = True
    points: Optional[List[Point]] = None


@dataclass
class CountingAoi:
    id: str
    name: str
    coordinates: Any
    enabled: bool = True
    type: str = "rectangle"


@dataclass
class CountingConfig:
    lines: List[CountingLine] = field(default_factory=list)
    aois: List[CountingAoi] = field(default_factory=list)
    line_crossing_epsilon: float = 1.0
    min_frames_between_same_line_crossing: int = 3
    aoi_boundary_debounce_frames: int = 3
    activity_bin_seconds: float = 60.0
    count_valid_tracks_only: bool = True


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


@dataclass
class AoiEvent:
    event_id: str
    track_id: int
    aoi_id: str
    aoi_name: str
    event_type: str
    frame: int
    time_s: float
    cx: float
    cy: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    dwell_time_s: Optional[float] = None


@dataclass
class TrackSummary:
    track_id: int
    valid: bool
    first_frame: int
    last_frame: int
    start_time_s: float
    end_time_s: float
    lifetime_frames: int
    duration_s: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    net_displacement_px: float
    path_length_px: float
    mean_speed_px_per_frame: float
    mean_speed_px_per_second: float
    directionality: float
    max_blob_area: int
    mean_blob_area: float
    max_score: float
    mean_score: float
    crossing_count: int
    aoi_entry_count: int
    aoi_exit_count: int


@dataclass
class CountingResults:
    crossings: List[CrossingEvent]
    aoi_events: List[AoiEvent]
    activity_rows: List[Dict[str, Any]]
    track_summaries: List[TrackSummary]
    run_summary: Dict[str, Any]
