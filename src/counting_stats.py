"""Track-based counting and statistics for thermal blob tracks."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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


def load_counting_config(path: Path) -> CountingConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return counting_config_from_dict(raw)


def counting_config_from_dict(raw: Dict[str, Any]) -> CountingConfig:
    lines = [_line_from_dict(item, idx) for idx, item in enumerate(raw.get("lines", []), start=1)]
    aoi_items = raw.get("aois", raw.get("zones", []))
    aois = [_aoi_from_dict(item, idx) for idx, item in enumerate(aoi_items, start=1)]
    return CountingConfig(
        lines=lines,
        aois=aois,
        line_crossing_epsilon=float(raw.get("line_crossing_epsilon", 1.0)),
        min_frames_between_same_line_crossing=int(raw.get("min_frames_between_same_line_crossing", 3)),
        aoi_boundary_debounce_frames=int(raw.get("aoi_boundary_debounce_frames", 3)),
        activity_bin_seconds=float(raw.get("activity_bin_seconds", 60.0)),
        count_valid_tracks_only=bool(raw.get("count_valid_tracks_only", True)),
    )


def merge_counting_configs(base: CountingConfig, override: CountingConfig) -> CountingConfig:
    if override.lines:
        base.lines = override.lines
    if override.aois:
        base.aois = override.aois
    base.line_crossing_epsilon = override.line_crossing_epsilon
    base.min_frames_between_same_line_crossing = override.min_frames_between_same_line_crossing
    base.aoi_boundary_debounce_frames = override.aoi_boundary_debounce_frames
    base.activity_bin_seconds = override.activity_bin_seconds
    base.count_valid_tracks_only = override.count_valid_tracks_only
    return base


def line_from_cli(value: str) -> CountingLine:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) not in (6, 8):
        raise ValueError("count line must use id,name,x1,y1,x2,y2[,positive_label,negative_label]")
    positive = parts[6] if len(parts) == 8 else "positive"
    negative = parts[7] if len(parts) == 8 else "negative"
    return CountingLine(
        id=parts[0],
        name=parts[1],
        p1=(float(parts[2]), float(parts[3])),
        p2=(float(parts[4]), float(parts[5])),
        positive_label=positive,
        negative_label=negative,
    )


def aoi_from_cli(value: str) -> CountingAoi:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 6:
        raise ValueError("count AOI must use id,name,x,y,w,h")
    return CountingAoi(
        id=parts[0],
        name=parts[1],
        coordinates=(float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])),
    )



def counting_config_to_dict(cfg: CountingConfig) -> Dict[str, Any]:
    return {
        "lines": [
            {
                "id": line.id,
                "name": line.name,
                "p1": list(line.p1),
                "p2": list(line.p2),
                **({"pts": [list(point) for point in _line_points(line)]} if line.points else {}),
                "direction_labels": {
                    "positive": line.positive_label,
                    "negative": line.negative_label,
                },
                "enabled": line.enabled,
            }
            for line in cfg.lines
        ],
        "aois": [
            {
                "id": aoi.id,
                "name": aoi.name,
                "type": aoi.type,
                "coordinates": _json_coordinates(aoi.coordinates),
                "enabled": aoi.enabled,
            }
            for aoi in cfg.aois
        ],
        "line_crossing_epsilon": cfg.line_crossing_epsilon,
        "min_frames_between_same_line_crossing": cfg.min_frames_between_same_line_crossing,
        "aoi_boundary_debounce_frames": cfg.aoi_boundary_debounce_frames,
        "activity_bin_seconds": cfg.activity_bin_seconds,
        "count_valid_tracks_only": cfg.count_valid_tracks_only,
    }



def analyze_tracks(
    tracks: Iterable[Any],
    fps: float,
    cfg: CountingConfig,
    is_valid_track: Callable[[Any], bool],
    input_video: str = "",
    frame_count_processed: int = 0,
    parameter_preset: str = "",
    notes: str = "",
) -> CountingResults:
    selected_tracks = [track for track in tracks if _track_detections(track)]
    valid_by_id = {int(track.track_id): bool(is_valid_track(track)) for track in selected_tracks}
    countable_tracks = [
        track for track in selected_tracks if not cfg.count_valid_tracks_only or valid_by_id[int(track.track_id)]
    ]

    crossings = detect_line_crossings(countable_tracks, fps, cfg)
    aoi_events = detect_aoi_events(countable_tracks, fps, cfg)
    track_summaries = summarize_tracks(selected_tracks, fps, valid_by_id, crossings, aoi_events)
    activity_rows = build_activity_rows(countable_tracks, crossings, aoi_events, fps, cfg, frame_count_processed)
    run_summary = build_run_summary(
        input_video=input_video,
        fps=fps,
        frame_count_processed=frame_count_processed,
        parameter_preset=parameter_preset,
        total_tracks=len(selected_tracks),
        valid_tracks=sum(1 for valid in valid_by_id.values() if valid),
        crossings=crossings,
        aoi_events=aoi_events,
        activity_rows=activity_rows,
        cfg=cfg,
        notes=notes,
    )
    return CountingResults(
        crossings=crossings,
        aoi_events=aoi_events,
        activity_rows=activity_rows,
        track_summaries=track_summaries,
        run_summary=run_summary,
    )


def detect_line_crossings(tracks: Iterable[Any], fps: float, cfg: CountingConfig) -> List[CrossingEvent]:
    events: List[CrossingEvent] = []
    counters: Dict[Tuple[int, str], int] = {}
    counted_lines: set[Tuple[int, str]] = set()

    for track in tracks:
        detections = _track_detections(track)
        if len(detections) < 2:
            continue

        for line in cfg.lines:
            if not line.enabled:
                continue
            points = _line_points(line)
            if len(points) < 2:
                continue

            last_side: Optional[int] = None
            last_event_frame: Optional[int] = None

            for idx, detection in enumerate(detections):
                point = _centroid(detection)
                side = _polyline_side(points, point, cfg.line_crossing_epsilon)
                if idx == 0:
                    last_side = side if side != 0 else None
                    continue

                previous_point = _centroid(detections[idx - 1])
                direction_sign = _polyline_crossing_sign(previous_point, point, points, cfg.line_crossing_epsilon)
                if direction_sign is None:
                    if side == 0:
                        continue
                    if last_side is None:
                        last_side = side
                        continue
                    if side == last_side:
                        continue
                    direction_sign = 1 if side > last_side else -1

                frame = int(detection.frame_idx)
                if last_event_frame is not None and frame - last_event_frame < cfg.min_frames_between_same_line_crossing:
                    if side != 0:
                        last_side = side
                    continue

                key = (int(track.track_id), line.id)
                if key in counted_lines:
                    break
                counters[key] = counters.get(key, 0) + 1
                counted_lines.add(key)
                direction = line.positive_label if direction_sign > 0 else line.negative_label
                cx, cy = point
                events.append(
                    CrossingEvent(
                        event_id=f"crossing_{track.track_id}_{line.id}_{counters[key]}",
                        track_id=int(track.track_id),
                        line_id=line.id,
                        line_name=line.name,
                        direction=direction,
                        frame=frame,
                        time_s=_time_s(frame, fps),
                        cx=cx,
                        cy=cy,
                    )
                )
                last_event_frame = frame
                if side != 0:
                    last_side = side
                break

    events.sort(key=lambda event: (event.frame, event.track_id, event.line_id, event.event_id))
    return events


def detect_aoi_events(tracks: Iterable[Any], fps: float, cfg: CountingConfig) -> List[AoiEvent]:
    events: List[AoiEvent] = []
    counters: Dict[Tuple[int, str], int] = {}

    for track in tracks:
        detections = _track_detections(track)
        if len(detections) < 2:
            continue

        for aoi in cfg.aois:
            if not aoi.enabled:
                continue

            previous_inside: Optional[bool] = None
            last_event_frame: Optional[int] = None
            entry_frame: Optional[int] = None
            visit_completed = False
            for detection in detections:
                if visit_completed:
                    break
                point = _centroid(detection)
                inside = point_in_aoi(point, aoi)
                if previous_inside is None:
                    previous_inside = inside
                    if inside:
                        frame = int(detection.frame_idx)
                        entry_frame = frame
                        key = (int(track.track_id), aoi.id)
                        counters[key] = counters.get(key, 0) + 1
                        cx, cy = point
                        events.append(
                            AoiEvent(
                                event_id=f"aoi_{track.track_id}_{aoi.id}_{counters[key]}",
                                track_id=int(track.track_id),
                                aoi_id=aoi.id,
                                aoi_name=aoi.name,
                                event_type="entry",
                                frame=frame,
                                time_s=_time_s(frame, fps),
                                cx=cx,
                                cy=cy,
                                start_frame=frame,
                            )
                        )
                        last_event_frame = frame
                    continue
                if inside == previous_inside:
                    continue

                frame = int(detection.frame_idx)
                if last_event_frame is not None and frame - last_event_frame < cfg.aoi_boundary_debounce_frames:
                    previous_inside = inside
                    continue

                key = (int(track.track_id), aoi.id)
                counters[key] = counters.get(key, 0) + 1
                event_type = "entry" if inside else "exit"
                if event_type == "entry":
                    entry_frame = frame
                    start_frame = frame
                    end_frame = None
                    dwell_time_s = None
                else:
                    start_frame = entry_frame
                    end_frame = frame
                    dwell_time_s = None if entry_frame is None else _time_s(frame - entry_frame, fps)
                cx, cy = point
                events.append(
                    AoiEvent(
                        event_id=f"aoi_{track.track_id}_{aoi.id}_{counters[key]}",
                        track_id=int(track.track_id),
                        aoi_id=aoi.id,
                        aoi_name=aoi.name,
                        event_type=event_type,
                        frame=frame,
                        time_s=_time_s(frame, fps),
                        cx=cx,
                        cy=cy,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        dwell_time_s=dwell_time_s,
                    )
                )
                last_event_frame = frame
                previous_inside = inside
                if event_type == "exit":
                    entry_frame = None
                    visit_completed = True

    events.sort(key=lambda event: (event.frame, event.track_id, event.aoi_id, event.event_id))
    return events


def summarize_tracks(
    tracks: Iterable[Any],
    fps: float,
    valid_by_id: Dict[int, bool],
    crossings: Sequence[CrossingEvent],
    aoi_events: Sequence[AoiEvent],
) -> List[TrackSummary]:
    crossing_counts: Dict[int, int] = {}
    entry_counts: Dict[int, int] = {}
    exit_counts: Dict[int, int] = {}
    for event in crossings:
        crossing_counts[event.track_id] = crossing_counts.get(event.track_id, 0) + 1
    for event in aoi_events:
        if event.event_type == "entry":
            entry_counts[event.track_id] = entry_counts.get(event.track_id, 0) + 1
        elif event.event_type == "exit":
            exit_counts[event.track_id] = exit_counts.get(event.track_id, 0) + 1

    summaries: List[TrackSummary] = []
    for track in tracks:
        detections = _track_detections(track)
        if not detections:
            continue
        first = detections[0]
        last = detections[-1]
        first_frame = int(first.frame_idx)
        last_frame = int(last.frame_idx)
        path_length = _path_length(detections)
        net_displacement = _distance(_centroid(first), _centroid(last)) if len(detections) > 1 else 0.0
        frame_span = max(1, last_frame - first_frame)
        duration_s = frame_span / fps if fps > 0 else 0.0
        mean_speed_px_per_frame = path_length / frame_span if frame_span > 0 else 0.0
        mean_speed_px_per_second = path_length / duration_s if duration_s > 0 else 0.0
        directionality = net_displacement / path_length if path_length > 0 else 0.0
        areas = [int(det.area) for det in detections]
        scores = [float(det.score) for det in detections]
        sx, sy = _centroid(first)
        ex, ey = _centroid(last)
        summaries.append(
            TrackSummary(
                track_id=int(track.track_id),
                valid=bool(valid_by_id.get(int(track.track_id), False)),
                first_frame=first_frame,
                last_frame=last_frame,
                start_time_s=_time_s(first_frame, fps),
                end_time_s=_time_s(last_frame, fps),
                lifetime_frames=len(detections),
                duration_s=duration_s,
                start_x=sx,
                start_y=sy,
                end_x=ex,
                end_y=ey,
                net_displacement_px=net_displacement,
                path_length_px=path_length,
                mean_speed_px_per_frame=mean_speed_px_per_frame,
                mean_speed_px_per_second=mean_speed_px_per_second,
                directionality=directionality,
                max_blob_area=max(areas),
                mean_blob_area=sum(areas) / len(areas),
                max_score=max(scores),
                mean_score=sum(scores) / len(scores),
                crossing_count=crossing_counts.get(int(track.track_id), 0),
                aoi_entry_count=entry_counts.get(int(track.track_id), 0),
                aoi_exit_count=exit_counts.get(int(track.track_id), 0),
            )
        )
    summaries.sort(key=lambda item: (item.valid, item.lifetime_frames, item.track_id), reverse=True)
    return summaries


def build_activity_rows(
    tracks: Iterable[Any],
    crossings: Sequence[CrossingEvent],
    aoi_events: Sequence[AoiEvent],
    fps: float,
    cfg: CountingConfig,
    frame_count_processed: int,
) -> List[Dict[str, Any]]:
    bin_seconds = max(0.001, cfg.activity_bin_seconds)
    max_time = frame_count_processed / fps if fps > 0 and frame_count_processed > 0 else 0.0
    for track in tracks:
        detections = _track_detections(track)
        if detections:
            max_time = max(max_time, _time_s(int(detections[-1].frame_idx), fps))
    for event in list(crossings) + list(aoi_events):
        max_time = max(max_time, float(event.time_s))

    bin_count = max(1, int(math.floor(max_time / bin_seconds)) + 1)
    rows: List[Dict[str, Any]] = []
    for idx in range(bin_count):
        start = idx * bin_seconds
        end = start + bin_seconds
        rows.append(
            {
                "time_bin_start_s": round(start, 4),
                "time_bin_end_s": round(end, 4),
                "valid_track_count_started": 0,
                "valid_track_count_active": 0,
                "line_crossings_total": 0,
                "line_crossings_by_line": "{}",
                "line_crossings_by_direction": "{}",
                "aoi_entries_total": 0,
                "aoi_exits_total": 0,
            }
        )

    for track in tracks:
        detections = _track_detections(track)
        if not detections:
            continue
        start_time = _time_s(int(detections[0].frame_idx), fps)
        end_time = _time_s(int(detections[-1].frame_idx), fps)
        start_idx = _bin_index(start_time, bin_seconds, len(rows))
        rows[start_idx]["valid_track_count_started"] += 1
        for idx, row in enumerate(rows):
            if float(row["time_bin_start_s"]) <= end_time and float(row["time_bin_end_s"]) > start_time:
                row["valid_track_count_active"] += 1

    line_counts: List[Dict[str, int]] = [dict() for _ in rows]
    direction_counts: List[Dict[str, int]] = [dict() for _ in rows]
    for event in crossings:
        idx = _bin_index(float(event.time_s), bin_seconds, len(rows))
        rows[idx]["line_crossings_total"] += 1
        line_counts[idx][event.line_id] = line_counts[idx].get(event.line_id, 0) + 1
        direction_counts[idx][event.direction] = direction_counts[idx].get(event.direction, 0) + 1

    for event in aoi_events:
        idx = _bin_index(float(event.time_s), bin_seconds, len(rows))
        if event.event_type == "entry":
            rows[idx]["aoi_entries_total"] += 1
        elif event.event_type == "exit":
            rows[idx]["aoi_exits_total"] += 1

    for idx, row in enumerate(rows):
        row["line_crossings_by_line"] = json.dumps(line_counts[idx], sort_keys=True)
        row["line_crossings_by_direction"] = json.dumps(direction_counts[idx], sort_keys=True)
    return rows


def build_run_summary(
    input_video: str,
    fps: float,
    frame_count_processed: int,
    parameter_preset: str,
    total_tracks: int,
    valid_tracks: int,
    crossings: Sequence[CrossingEvent],
    aoi_events: Sequence[AoiEvent],
    activity_rows: Sequence[Dict[str, Any]],
    cfg: CountingConfig,
    notes: str,
) -> Dict[str, Any]:
    by_line_and_direction: Dict[str, Dict[str, int]] = {}
    for event in crossings:
        by_line_and_direction.setdefault(event.line_id, {})
        by_line_and_direction[event.line_id][event.direction] = by_line_and_direction[event.line_id].get(event.direction, 0) + 1

    peak_row = max(activity_rows, key=lambda row: int(row["line_crossings_total"]) + int(row["aoi_entries_total"]), default=None)
    return {
        "input_video": input_video,
        "fps": fps,
        "frame_count_processed": frame_count_processed,
        "parameter_preset": parameter_preset,
        "total_tracks": total_tracks,
        "valid_tracks": valid_tracks,
        "invalid_tracks": max(0, total_tracks - valid_tracks),
        "total_line_crossings": len(crossings),
        "crossings_by_line_and_direction": by_line_and_direction,
        "total_aoi_entries": sum(1 for event in aoi_events if event.event_type == "entry"),
        "total_aoi_exits": sum(1 for event in aoi_events if event.event_type == "exit"),
        "activity_bin_seconds": cfg.activity_bin_seconds,
        "peak_activity_bin": None if peak_row is None else {
            "time_bin_start_s": peak_row["time_bin_start_s"],
            "time_bin_end_s": peak_row["time_bin_end_s"],
        },
        "peak_activity_count": 0 if peak_row is None else int(peak_row["line_crossings_total"]) + int(peak_row["aoi_entries_total"]),
        "notes": notes,
    }


def write_crossings_csv(path: Path, events: Sequence[CrossingEvent]) -> None:
    _write_dataclass_csv(path, events, [
        "event_id", "track_id", "line_id", "line_name", "direction", "frame", "time_s", "cx", "cy"
    ])


def write_aoi_events_csv(path: Path, events: Sequence[AoiEvent]) -> None:
    _write_dataclass_csv(path, events, [
        "event_id", "track_id", "aoi_id", "aoi_name", "event_type", "frame", "time_s", "cx", "cy",
        "start_frame", "end_frame", "dwell_time_s",
    ])


def write_activity_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "time_bin_start_s",
        "time_bin_end_s",
        "valid_track_count_started",
        "valid_track_count_active",
        "line_crossings_total",
        "line_crossings_by_line",
        "line_crossings_by_direction",
        "aoi_entries_total",
        "aoi_exits_total",
    ]
    _write_dict_csv(path, rows, fieldnames)


def write_track_summary_csv(path: Path, summaries: Sequence[TrackSummary]) -> None:
    fieldnames = [
        "track_id",
        "valid",
        "first_frame",
        "last_frame",
        "start_time_s",
        "end_time_s",
        "lifetime_frames",
        "duration_s",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "net_displacement_px",
        "path_length_px",
        "mean_speed_px_per_frame",
        "mean_speed_px_per_second",
        "directionality",
        "max_blob_area",
        "mean_blob_area",
        "max_score",
        "mean_score",
        "crossing_count",
        "aoi_entry_count",
        "aoi_exit_count",
    ]
    _write_dataclass_csv(path, summaries, fieldnames)


def write_run_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def write_counting_config_json(path: Path, cfg: CountingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(counting_config_to_dict(cfg), f, indent=2, sort_keys=True)
        f.write("\n")



def point_in_rect(point: Point, rect: Rect) -> bool:
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh



def point_in_aoi(point: Point, aoi: CountingAoi) -> bool:
    if aoi.type == "polygon":
        return point_in_polygon(point, aoi.coordinates)
    return point_in_rect(point, aoi.coordinates)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for idx in range(len(polygon)):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            x_at_y = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            if x < x_at_y:
                inside = not inside
    return inside


def _line_from_dict(item: Dict[str, Any], idx: int) -> CountingLine:
    line_id = str(item.get("id", item.get("name", f"line_{idx}")))
    name = str(item.get("name", line_id))
    raw_points = item.get("pts")
    if raw_points is None and "a" in item and "b" in item:
        raw_points = [item["a"], item["b"]]
    if raw_points is not None:
        points = [_point(point) for point in raw_points]
        if len(points) < 2:
            raise ValueError(f"line '{line_id}' must contain at least two points")
        p1, p2 = points[0], points[-1]
    else:
        p1 = _point(item["p1"])
        p2 = _point(item["p2"])
        points = None
    labels = item.get("direction_labels", {})
    return CountingLine(
        id=line_id,
        name=name,
        p1=p1,
        p2=p2,
        positive_label=str(item.get("positive_label", labels.get("positive", "positive"))),
        negative_label=str(item.get("negative_label", labels.get("negative", "negative"))),
        enabled=bool(item.get("enabled", True)),
        points=points,
    )


def _aoi_from_dict(item: Dict[str, Any], idx: int) -> CountingAoi:
    aoi_id = str(item.get("id", item.get("name", f"aoi_{idx}")))
    name = str(item.get("name", aoi_id))
    aoi_type = str(item.get("type", "polygon" if "pts" in item else "rectangle"))
    if aoi_type == "polygon":
        raw_points = item.get("coordinates", item.get("pts", []))
        coordinates = [_point(point) for point in raw_points]
        if len(coordinates) < 3:
            raise ValueError(f"polygon AOI '{aoi_id}' must contain at least three points")
    else:
        raw_rect = item.get("coordinates")
        coordinates = tuple(float(v) for v in raw_rect)
        if len(coordinates) != 4:
            raise ValueError(f"rectangle AOI '{aoi_id}' must contain x,y,w,h")
    return CountingAoi(
        id=aoi_id,
        name=name,
        coordinates=coordinates,
        enabled=bool(item.get("enabled", True)),
        type=aoi_type,
    )


def _json_coordinates(coordinates: Any) -> Any:
    if isinstance(coordinates, tuple):
        return list(coordinates)
    if isinstance(coordinates, list):
        return [list(point) if isinstance(point, tuple) else point for point in coordinates]
    return coordinates


def _line_points(line: CountingLine) -> List[Point]:
    if line.points and len(line.points) >= 2:
        return [(float(x), float(y)) for x, y in line.points]
    return [line.p1, line.p2]


def _polyline_side(points: Sequence[Point], point: Point, epsilon: float) -> int:
    best_idx = 0
    best_dist = float("inf")
    for idx in range(len(points) - 1):
        dist = _point_to_segment_distance_sq(points[idx], points[idx + 1], point)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return _signed_side(points[best_idx], points[best_idx + 1], point, epsilon)


def _polyline_crossing_sign(previous_point: Point, current_point: Point, points: Sequence[Point], epsilon: float) -> Optional[int]:
    for idx in range(len(points) - 1):
        a = points[idx]
        b = points[idx + 1]
        if not _segments_intersect(previous_point, current_point, a, b):
            continue
        previous_side = _signed_side(a, b, previous_point, epsilon)
        current_side = _signed_side(a, b, current_point, epsilon)
        if previous_side < 0 and current_side > 0:
            return 1
        if previous_side > 0 and current_side < 0:
            return -1
    return None


def _signed_side(a: Point, b: Point, point: Point, epsilon: float) -> int:
    cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    if abs(cross) <= epsilon:
        return 0
    return 1 if cross > 0 else -1


def _point_to_segment_distance_sq(a: Point, b: Point, point: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    length_sq = vx * vx + vy * vy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    cx, cy = ax + t * vx, ay + t * vy
    return (px - cx) ** 2 + (py - cy) ** 2


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    def orient(a: Point, b: Point, c: Point) -> int:
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 1 if value > 0 else (-1 if value < 0 else 0)

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a[0], b[0]) - 1e-6 <= c[0] <= max(a[0], b[0]) + 1e-6
            and min(a[1], b[1]) - 1e-6 <= c[1] <= max(a[1], b[1]) + 1e-6
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, p2, q2):
        return True
    if o3 == 0 and on_segment(q1, q2, p1):
        return True
    if o4 == 0 and on_segment(q1, q2, p2):
        return True
    return False


def _line_side(point: Point, line: CountingLine, epsilon: float) -> int:
    x, y = point
    x1, y1 = line.p1
    x2, y2 = line.p2
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if abs(cross) <= epsilon:
        return 0
    return 1 if cross > 0 else -1


def _track_detections(track: Any) -> List[Any]:
    return list(getattr(track, "detections", []))


def _centroid(detection: Any) -> Point:
    cx, cy = detection.centroid
    return float(cx), float(cy)


def _point(value: Sequence[Any]) -> Point:
    if len(value) != 2:
        raise ValueError("point must contain exactly two values")
    return float(value[0]), float(value[1])


def _same_point(a: Point, b: Point) -> bool:
    return a[0] == b[0] and a[1] == b[1]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_length(detections: Sequence[Any]) -> float:
    total = 0.0
    for idx in range(1, len(detections)):
        total += _distance(_centroid(detections[idx - 1]), _centroid(detections[idx]))
    return total


def _time_s(frame: int, fps: float) -> float:
    return round(frame / fps, 4) if fps > 0 else 0.0


def _bin_index(time_s: float, bin_seconds: float, row_count: int) -> int:
    return max(0, min(row_count - 1, int(math.floor(time_s / bin_seconds))))


def _write_dataclass_csv(path: Path, rows: Sequence[Any], fieldnames: Sequence[str]) -> None:
    dict_rows = [_rounded_dict(asdict(row)) for row in rows]
    _write_dict_csv(path, dict_rows, fieldnames)


def _write_dict_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rounded_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        elif isinstance(value, bool):
            out[key] = int(value)
        else:
            out[key] = value
    return out
