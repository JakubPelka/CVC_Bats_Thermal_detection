"""Track metrics and flight-like track validation."""

import math

from .config import ThermalBlobConfig
from .models import Point, Track, TrackMetrics


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def track_metrics(track: Track) -> TrackMetrics:
    if len(track.detections) < 2:
        return TrackMetrics()
    net_displacement = euclidean_distance(track.detections[0].centroid, track.detections[-1].centroid)
    frame_span = max(1, track.detections[-1].frame_idx - track.detections[0].frame_idx)
    return TrackMetrics(
        path_length=track.path_length,
        net_displacement=net_displacement,
        mean_speed=track.path_length / frame_span,
        max_step_speed=track.max_step_speed,
        directionality=net_displacement / track.path_length if track.path_length > 0 else 0.0,
        frame_span=frame_span,
    )


def is_valid_flying_track(track: Track, cfg: ThermalBlobConfig) -> bool:
    if track.lifetime < cfg.min_track_lifetime:
        return False
    if cfg.min_track_max_blob_area > 0 and track.max_blob_area < cfg.min_track_max_blob_area:
        return False
    if cfg.min_track_mean_blob_area > 0:
        mean_blob_area = track.blob_area_sum / max(1, track.lifetime)
        if mean_blob_area < cfg.min_track_mean_blob_area:
            return False
    metrics = track_metrics(track)
    return (
        metrics.net_displacement >= cfg.min_track_displacement
        and metrics.path_length >= cfg.min_track_path_length
        and metrics.mean_speed >= cfg.min_mean_speed
        and metrics.mean_speed <= cfg.max_mean_speed
        and metrics.directionality >= cfg.min_directionality
    )
