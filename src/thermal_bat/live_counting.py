"""Streaming line and AOI counting used by the live preview pipeline."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from counting_models import AoiEvent, CountingAoi, CountingConfig, CrossingEvent
from counting_stats import point_in_aoi, _line_points, _polyline_crossing_sign, _polyline_side, _time_s
from .models import Point, Track


class LiveCounting:
    """Streaming line/AOI counter used while frames are processed."""

    CROSSING_FIELDS = ["event_id", "track_id", "line_id", "line_name", "direction", "frame", "time_s", "cx", "cy"]
    AOI_FIELDS = [
        "event_id", "track_id", "aoi_id", "aoi_name", "event_type", "frame", "time_s", "cx", "cy",
        "start_frame", "end_frame", "dwell_time_s",
    ]

    def __init__(
        self,
        cfg: CountingConfig,
        fps: float,
        crossings_csv_path: Optional[Path],
        aoi_events_csv_path: Optional[Path],
        is_countable_track,
    ) -> None:
        self.cfg = cfg
        self.fps = fps
        self.is_countable_track = is_countable_track
        self.crossings: List[CrossingEvent] = []
        self.aoi_events: List[AoiEvent] = []
        self.line_totals: Dict[str, int] = {}
        self.direction_totals: Dict[str, int] = {}
        self.line_direction_totals: Dict[Tuple[str, str], int] = {}
        self.aoi_seen_tracks: Dict[str, set[int]] = {aoi.id: set() for aoi in cfg.aois}
        self.aoi_active_tracks: Dict[str, set[int]] = {aoi.id: set() for aoi in cfg.aois}
        self.aoi_last_dwell_s: Dict[str, float] = {}
        self._line_state: Dict[Tuple[int, str], Dict[str, Optional[int]]] = {}
        self._aoi_state: Dict[Tuple[int, str], Dict[str, object]] = {}
        self._crossing_counts: Dict[Tuple[int, str], int] = {}
        self._aoi_counts: Dict[Tuple[int, str], int] = {}
        self._validity_cache: Dict[int, Tuple[int, bool]] = {}
        self._line_points_cache: Dict[str, List[Point]] = {
            line.id: _line_points(line) for line in cfg.lines if line.enabled
        }
        self._crossings_file = None
        self._aoi_file = None
        self._crossings_writer = None
        self._aoi_writer = None

        if crossings_csv_path and cfg.lines:
            crossings_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._crossings_file = crossings_csv_path.open("w", newline="", encoding="utf-8")
            self._crossings_writer = csv.DictWriter(self._crossings_file, fieldnames=self.CROSSING_FIELDS)
            self._crossings_writer.writeheader()
            self._crossings_file.flush()
        if aoi_events_csv_path and cfg.aois:
            aoi_events_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._aoi_file = aoi_events_csv_path.open("w", newline="", encoding="utf-8")
            self._aoi_writer = csv.DictWriter(self._aoi_file, fieldnames=self.AOI_FIELDS)
            self._aoi_writer.writeheader()
            self._aoi_file.flush()

    def close(self) -> None:
        for file_obj in (self._crossings_file, self._aoi_file):
            if file_obj is not None:
                file_obj.flush()
                file_obj.close()

    def update(
        self,
        tracks: Sequence[Track],
        frame_idx: int,
        refresh_aoi_occupancy: bool = True,
        occupancy_tracks: Optional[Sequence[Track]] = None,
    ) -> None:
        for track in tracks:
            if track.last_frame_idx != frame_idx or not self._is_countable(track):
                continue
            if self.cfg.lines:
                self._update_lines(track)
            if self.cfg.aois:
                self._update_aois(track)
        if self.cfg.aois and refresh_aoi_occupancy:
            self._refresh_active_aoi_tracks(occupancy_tracks if occupancy_tracks is not None else tracks)

    def _is_countable(self, track: Track) -> bool:
        if not self.cfg.count_valid_tracks_only:
            return True
        cached = self._validity_cache.get(int(track.track_id))
        if cached is not None and cached[0] == track.lifetime:
            return cached[1]
        valid = bool(self.is_countable_track(track))
        self._validity_cache[int(track.track_id)] = (track.lifetime, valid)
        return valid

    def _refresh_active_aoi_tracks(self, tracks: Sequence[Track]) -> None:
        active_by_aoi: Dict[str, set[int]] = {aoi.id: set() for aoi in self.cfg.aois}
        for track in tracks:
            if not track.active:
                continue
            if not self._is_countable(track):
                continue
            point = track.last_point
            if point is None:
                continue
            for aoi in self.cfg.aois:
                if aoi.enabled and point_in_aoi(point, aoi):
                    active_by_aoi.setdefault(aoi.id, set()).add(int(track.track_id))
        self.aoi_active_tracks = active_by_aoi

    def _update_lines(self, track: Track) -> None:
        if len(track.detections) < 2:
            return
        previous = track.detections[-2]
        current = track.detections[-1]
        previous_point = previous.centroid
        current_point = current.centroid
        frame = int(current.frame_idx)

        for line in self.cfg.lines:
            if not line.enabled:
                continue
            points = self._line_points_cache.get(line.id, [])
            if len(points) < 2:
                continue
            key = (int(track.track_id), line.id)
            if self._crossing_counts.get(key, 0) > 0:
                continue
            state = self._line_state.setdefault(key, {"last_side": None, "last_event_frame": None})
            if state["last_side"] is None:
                previous_side = _polyline_side(points, previous_point, self.cfg.line_crossing_epsilon)
                state["last_side"] = previous_side if previous_side != 0 else None

            side = _polyline_side(points, current_point, self.cfg.line_crossing_epsilon)
            direction_sign = _polyline_crossing_sign(previous_point, current_point, points, self.cfg.line_crossing_epsilon)
            if direction_sign is None:
                if side == 0:
                    continue
                last_side = state["last_side"]
                if last_side is None:
                    state["last_side"] = side
                    continue
                if side == last_side:
                    continue
                direction_sign = 1 if side > last_side else -1

            last_event_frame = state["last_event_frame"]
            if last_event_frame is not None and frame - int(last_event_frame) < self.cfg.min_frames_between_same_line_crossing:
                if side != 0:
                    state["last_side"] = side
                continue

            self._crossing_counts[key] = self._crossing_counts.get(key, 0) + 1
            direction = line.positive_label if direction_sign > 0 else line.negative_label
            cx, cy = current_point
            event = CrossingEvent(
                event_id=f"crossing_{track.track_id}_{line.id}_{self._crossing_counts[key]}",
                track_id=int(track.track_id),
                line_id=line.id,
                line_name=line.name,
                direction=direction,
                frame=frame,
                time_s=_time_s(frame, self.fps),
                cx=float(cx),
                cy=float(cy),
            )
            self.crossings.append(event)
            self.line_totals[line.id] = self.line_totals.get(line.id, 0) + 1
            self.direction_totals[direction] = self.direction_totals.get(direction, 0) + 1
            direction_key = (line.id, direction)
            self.line_direction_totals[direction_key] = self.line_direction_totals.get(direction_key, 0) + 1
            if self._crossings_writer is not None:
                self._crossings_writer.writerow(_csv_row(event, self.CROSSING_FIELDS))
                self._crossings_file.flush()
            state["last_event_frame"] = frame
            if side != 0:
                state["last_side"] = side

    def _update_aois(self, track: Track) -> None:
        current = track.detections[-1]
        point = current.centroid
        frame = int(current.frame_idx)
        for aoi in self.cfg.aois:
            if not aoi.enabled:
                continue
            inside = point_in_aoi(point, aoi)
            key = (int(track.track_id), aoi.id)
            state = self._aoi_state.setdefault(
                key,
                {"previous_inside": None, "last_event_frame": None, "entry_frame": None, "visit_completed": False},
            )
            if state.get("visit_completed"):
                continue
            previous_inside = state["previous_inside"]
            if previous_inside is None:
                state["previous_inside"] = inside
                if inside:
                    state["entry_frame"] = frame
                    self._record_aoi_event(track, aoi, "entry", frame, point, frame, None, None)
                continue
            if inside == previous_inside:
                continue

            last_event_frame = state["last_event_frame"]
            if last_event_frame is not None and frame - int(last_event_frame) < self.cfg.aoi_boundary_debounce_frames:
                state["previous_inside"] = inside
                continue

            if inside:
                state["entry_frame"] = frame
                self._record_aoi_event(track, aoi, "entry", frame, point, frame, None, None)
            else:
                entry_frame = state["entry_frame"]
                dwell_time_s = None if entry_frame is None else _time_s(frame - int(entry_frame), self.fps)
                self._record_aoi_event(track, aoi, "exit", frame, point, entry_frame, frame, dwell_time_s)
                state["entry_frame"] = None
                state["visit_completed"] = True
            state["last_event_frame"] = frame
            state["previous_inside"] = inside

    def _record_aoi_event(
        self,
        track: Track,
        aoi: CountingAoi,
        event_type: str,
        frame: int,
        point: Point,
        start_frame: Optional[int],
        end_frame: Optional[int],
        dwell_time_s: Optional[float],
    ) -> None:
        key = (int(track.track_id), aoi.id)
        self._aoi_counts[key] = self._aoi_counts.get(key, 0) + 1
        cx, cy = point
        event = AoiEvent(
            event_id=f"aoi_{track.track_id}_{aoi.id}_{self._aoi_counts[key]}",
            track_id=int(track.track_id),
            aoi_id=aoi.id,
            aoi_name=aoi.name,
            event_type=event_type,
            frame=frame,
            time_s=_time_s(frame, self.fps),
            cx=float(cx),
            cy=float(cy),
            start_frame=start_frame,
            end_frame=end_frame,
            dwell_time_s=dwell_time_s,
        )
        self.aoi_events.append(event)
        if event_type == "entry":
            self.aoi_seen_tracks.setdefault(aoi.id, set()).add(int(track.track_id))
            self.aoi_active_tracks.setdefault(aoi.id, set()).add(int(track.track_id))
        elif event_type == "exit":
            self.aoi_active_tracks.setdefault(aoi.id, set()).discard(int(track.track_id))
            if dwell_time_s is not None:
                self.aoi_last_dwell_s[aoi.id] = dwell_time_s
        if self._aoi_writer is not None:
            self._aoi_writer.writerow(_csv_row(event, self.AOI_FIELDS))
            self._aoi_file.flush()


def _csv_row(event, fieldnames: Sequence[str]) -> Dict[str, object]:
    row = asdict(event)
    out: Dict[str, object] = {}
    for key in fieldnames:
        value = row.get(key)
        out[key] = round(value, 4) if isinstance(value, float) else value
    return out



