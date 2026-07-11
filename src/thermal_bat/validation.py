"""Track metrics and flight-like track validation."""

import math

from .config import ThermalBlobConfig
from .models import Point, Track, TrackMetrics


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def track_metrics(track: Track) -> TrackMetrics:
    points = track.points()
    if len(points) < 2:
        return TrackMetrics()
    path_length = 0.0
    max_step_speed = 0.0
    for index in range(1, len(points)):
        step_distance = euclidean_distance(points[index - 1], points[index])
        frame_delta = max(1, track.detections[index].frame_idx - track.detections[index - 1].frame_idx)
        path_length += step_distance
        max_step_speed = max(max_step_speed, step_distance / frame_delta)
    net_displacement = euclidean_distance(points[0], points[-1])
    frame_span = max(1, track.detections[-1].frame_idx - track.detections[0].frame_idx)
    return TrackMetrics(
        path_length=path_length,
        net_displacement=net_displacement,
        mean_speed=path_length / frame_span,
        max_step_speed=max_step_speed,
        directionality=net_displacement / path_length if path_length > 0 else 0.0,
        frame_span=frame_span,
    )


def is_valid_flying_track(track: Track, cfg: ThermalBlobConfig) -> bool:
    if track.lifetime < cfg.min_track_lifetime:
        return False
    if cfg.min_track_max_blob_area > 0 and max((item.area for item in track.detections), default=0) < cfg.min_track_max_blob_area:
        return False
    if cfg.min_track_mean_blob_area > 0:
        mean_blob_area = sum(item.area for item in track.detections) / max(1, track.lifetime)
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
