"""Detector-specific per-point and legacy per-track CSV exports."""

import csv
from pathlib import Path
from typing import Dict

from .config import ThermalBlobConfig
from .models import Track
from .validation import is_valid_flying_track, track_metrics


def write_track_points_csv(path: Path, tracks: Dict[int, Track], fps: float, cfg: ThermalBlobConfig) -> None:
    rows = []
    metrics_by_track_id = {track.track_id: track_metrics(track) for track in tracks.values()}

    for track in tracks.values():
        m = metrics_by_track_id[track.track_id]
        confirmed = int(track.lifetime >= cfg.min_track_lifetime)
        valid = int(is_valid_flying_track(track, cfg))

        for detection in track.detections:
            x, y, w, h = detection.bbox
            cx, cy = detection.centroid
            rows.append(
                {
                    "frame": detection.frame_idx,
                    "time_s": round(detection.frame_idx / fps, 4) if fps > 0 else "",
                    "track_id": track.track_id,
                    "track_lifetime_frames": track.lifetime,
                    "confirmed": confirmed,
                    "valid_flying_track": valid,
                    "cx": round(cx, 3),
                    "cy": round(cy, 3),
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "area_px": detection.area,
                    "score": round(detection.score, 4),
                    "mean_diff": round(detection.mean_diff, 3),
                    "max_diff": round(detection.max_diff, 3),
                    "track_path_length_px": round(m.path_length, 3),
                    "track_net_displacement_px": round(m.net_displacement, 3),
                    "track_mean_speed_px_frame": round(m.mean_speed, 3),
                    "track_max_step_speed_px_frame": round(m.max_step_speed, 3),
                    "track_directionality": round(m.directionality, 4),
                    "track_frame_span": m.frame_span,
                }
            )

    rows.sort(key=lambda r: (r["frame"], r["track_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame",
            "time_s",
            "track_id",
            "track_lifetime_frames",
            "confirmed",
            "valid_flying_track",
            "cx",
            "cy",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "area_px",
            "score",
            "mean_diff",
            "max_diff",
            "track_path_length_px",
            "track_net_displacement_px",
            "track_mean_speed_px_frame",
            "track_max_step_speed_px_frame",
            "track_directionality",
            "track_frame_span",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_track_summary_csv(path: Path, tracks: Dict[int, Track], cfg: ThermalBlobConfig) -> None:
    rows = []

    for track in tracks.values():
        if not track.detections:
            continue

        m = track_metrics(track)
        first = track.detections[0]
        last = track.detections[-1]
        rows.append(
            {
                "track_id": track.track_id,
                "valid_flying_track": int(is_valid_flying_track(track, cfg)),
                "confirmed": int(track.lifetime >= cfg.min_track_lifetime),
                "active_at_end": int(track.active),
                "lifetime_frames": track.lifetime,
                "first_frame": first.frame_idx,
                "last_frame": last.frame_idx,
                "frame_span": m.frame_span,
                "start_cx": round(first.centroid[0], 3),
                "start_cy": round(first.centroid[1], 3),
                "end_cx": round(last.centroid[0], 3),
                "end_cy": round(last.centroid[1], 3),
                "path_length_px": round(m.path_length, 3),
                "net_displacement_px": round(m.net_displacement, 3),
                "mean_speed_px_frame": round(m.mean_speed, 3),
                "max_step_speed_px_frame": round(m.max_step_speed, 3),
                "directionality": round(m.directionality, 4),
            }
        )

    rows.sort(key=lambda r: (r["valid_flying_track"], r["lifetime_frames"]), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "track_id",
            "valid_flying_track",
            "confirmed",
            "active_at_end",
            "lifetime_frames",
            "first_frame",
            "last_frame",
            "frame_span",
            "start_cx",
            "start_cy",
            "end_cx",
            "end_cy",
            "path_length_px",
            "net_displacement_px",
            "mean_speed_px_frame",
            "max_step_speed_px_frame",
            "directionality",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



