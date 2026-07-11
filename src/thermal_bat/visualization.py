"""OpenCV overlays for live preview, annotated video, and event clips."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from counting_geometry import line_points
from counting_models import CountingConfig
from event_clips import ClipWindow
from .config import ThermalBlobConfig
from .models import Point, Track
from .validation import is_valid_flying_track


class OverlayRenderer:
    """Stateful renderer that raster-caches completed valid trajectories."""

    def __init__(self) -> None:
        self._trail_layer: Optional[np.ndarray] = None
        self._trail_mask: Optional[np.ndarray] = None
        self._cached_track_ids: set[int] = set()
        self._cached_y = np.empty(0, dtype=np.intp)
        self._cached_x = np.empty(0, dtype=np.intp)
        self._cached_alpha = np.empty((0, 1), dtype=np.float32)
        self._cached_colors = np.empty((0, 3), dtype=np.float32)

    def render(self, frame: np.ndarray, detections: list, detector: Any, frame_idx: int,
               counting_cfg: Optional[CountingConfig] = None, live_counter: Any = None,
               progress_text: Optional[str] = None) -> np.ndarray:
        out = frame.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        if self._trail_layer is None or self._trail_layer.shape != out.shape:
            self._trail_layer = np.zeros_like(out)
            self._trail_mask = np.zeros(out.shape[:2], dtype=np.uint8)
            self._cached_track_ids.clear()
            self._clear_sparse_cache()

        cfg = detector.config
        active_tracks = []
        cache_changed = False
        for track in detector.drawable_tracks():
            if track.active or not cfg.draw_inactive_tracks:
                active_tracks.append(track)
                continue
            if track.track_id in self._cached_track_ids:
                continue
            self._draw_track(self._trail_layer, track, cfg, thickness=1)
            points = track.recent_points(cfg.trail_length)
            if len(points) >= 2:
                polyline = np.asarray([tuple_int(point) for point in points], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(self._trail_mask, [polyline], False, 255, 1, cv2.LINE_AA)
                cache_changed = True
            self._cached_track_ids.add(track.track_id)

        if cache_changed:
            self._refresh_sparse_cache()
        if cfg.draw_inactive_tracks and self._cached_y.size:
            background = out[self._cached_y, self._cached_x].astype(np.float32)
            out[self._cached_y, self._cached_x] = (
                self._cached_colors * self._cached_alpha
                + background * (1.0 - self._cached_alpha)
            ).astype(np.uint8)
        for track in active_tracks:
            self._draw_track(out, track, cfg, thickness=1)
        _draw_common_overlays(out, cfg, counting_cfg, live_counter, frame_idx, progress_text)
        return out

    def _clear_sparse_cache(self) -> None:
        self._cached_y = np.empty(0, dtype=np.intp)
        self._cached_x = np.empty(0, dtype=np.intp)
        self._cached_alpha = np.empty((0, 1), dtype=np.float32)
        self._cached_colors = np.empty((0, 3), dtype=np.float32)

    def _refresh_sparse_cache(self) -> None:
        """Index trail pixels only when a completed trajectory is added."""
        assert self._trail_layer is not None and self._trail_mask is not None
        self._cached_y, self._cached_x = np.nonzero(self._trail_mask)
        if not self._cached_y.size:
            self._clear_sparse_cache()
            return
        self._cached_alpha = (
            self._trail_mask[self._cached_y, self._cached_x].astype(np.float32)[:, None] / 255.0
        )
        self._cached_colors = self._trail_layer[self._cached_y, self._cached_x].astype(np.float32)

    @staticmethod
    def _draw_track(target: np.ndarray, track: Track, cfg: ThermalBlobConfig, thickness: int) -> None:
        points = track.recent_points(cfg.trail_length)
        if len(points) < 2:
            return
        polyline = np.asarray([tuple_int(point) for point in points], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(target, [polyline], False, color_for_track(track.track_id), thickness, cv2.LINE_AA)


def tuple_int(point: Point) -> Tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


@lru_cache(maxsize=None)
def color_for_track(track_id: int) -> Tuple[int, int, int]:
    color = np.random.default_rng(track_id * 9973).integers(80, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def should_draw_track(track: Track, cfg: ThermalBlobConfig) -> bool:
    return (cfg.draw_inactive_tracks or track.active) and (not cfg.draw_valid_only or is_valid_flying_track(track, cfg))


def draw_debug_overlay(frame: np.ndarray, detections: list, detector: Any, frame_idx: int,
                       counting_cfg: Optional[CountingConfig] = None, live_counter: Any = None,
                       progress_text: Optional[str] = None) -> np.ndarray:
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cfg = detector.config
    for track in detector.drawable_tracks():
        points = track.recent_points(cfg.trail_length)
        if len(points) >= 2:
            polyline = np.asarray([tuple_int(point) for point in points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [polyline], False, color_for_track(track.track_id), 1, cv2.LINE_AA)
    _draw_common_overlays(out, cfg, counting_cfg, live_counter, frame_idx, progress_text)
    return out


def _draw_common_overlays(out: np.ndarray, cfg: ThermalBlobConfig,
                          counting_cfg: Optional[CountingConfig], live_counter: Any,
                          frame_idx: int, progress_text: Optional[str]) -> None:
    if cfg.draw_roi and cfg.roi is not None:
        x, y, w, h = cfg.roi
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 1)
    if cfg.draw_exclude_zones:
        for x, y, w, h in cfg.exclude_zones:
            cv2.rectangle(out, (x, y), (x + w, y + h), (120, 120, 120), 1)
    if counting_cfg is not None:
        draw_counting_geometry(out, counting_cfg, live_counter)
    if live_counter is not None:
        draw_counting_hud(out, live_counter, frame_idx)
    if progress_text:
        y = max(18, out.shape[0] - 12)
        cv2.putText(out, progress_text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_counting_geometry(frame: np.ndarray, cfg: CountingConfig, live_counter: Any = None) -> None:
    overlay = frame.copy() if cfg.aois else None
    for aoi in cfg.aois:
        if not aoi.enabled:
            continue
        if aoi.type == "polygon":
            points = np.array([[tuple_int(point) for point in aoi.coordinates]], dtype=np.int32)
            cv2.polylines(frame, points, True, (0, 180, 255), 2, cv2.LINE_AA)
            cv2.fillPoly(overlay, points, (0, 90, 160))
            label_point = tuple_int(_polygon_label_point(aoi.coordinates))
        else:
            x, y, w, h = (int(round(value)) for value in aoi.coordinates)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 90, 160), -1)
            label_point = (x + 4, y + 16)
        seen = 0 if live_counter is None else len(live_counter.aoi_seen_tracks.get(aoi.id, set()))
        active = 0 if live_counter is None else len(live_counter.aoi_active_tracks.get(aoi.id, set()))
        cv2.putText(frame, f"{aoi.name} seen:{seen} in:{active}", label_point, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 210, 255), 1, cv2.LINE_AA)
    if overlay is not None:
        cv2.addWeighted(overlay, .18, frame, .82, 0, frame)
    for line in cfg.lines:
        if not line.enabled:
            continue
        points = [tuple_int(point) for point in line_points(line)]
        for index in range(1, len(points)):
            cv2.line(frame, points[index - 1], points[index], (255, 210, 0), 2, cv2.LINE_AA)
        total = 0 if live_counter is None else live_counter.line_totals.get(line.id, 0)
        cv2.putText(frame, f"{line.name}: {total}", (points[0][0] + 4, points[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 230, 60), 1, cv2.LINE_AA)


def draw_counting_hud(frame: np.ndarray, live_counter: Any, frame_idx: int) -> None:
    seen_total = sum(len(track_ids) for track_ids in live_counter.aoi_seen_tracks.values())
    rows = [f"frame {frame_idx}  crossings {len(live_counter.crossings)}  AOI seen {seen_total}"]
    for line in live_counter.cfg.lines[:4]:
        rows.append(f"L {line.name}: {live_counter.line_totals.get(line.id, 0)}  {line.positive_label}:{live_counter.line_direction_totals.get((line.id, line.positive_label), 0)} {line.negative_label}:{live_counter.line_direction_totals.get((line.id, line.negative_label), 0)}")
    for aoi in live_counter.cfg.aois[:4]:
        dwell = live_counter.aoi_last_dwell_s.get(aoi.id)
        rows.append(f"AOI {aoi.name}: seen {len(live_counter.aoi_seen_tracks.get(aoi.id, set()))} in {len(live_counter.aoi_active_tracks.get(aoi.id, set()))}{'' if dwell is None else f' last {dwell:.1f}s'}")
    width, height = min(frame.shape[1] - 12, 520), 10 + 20 * len(rows)
    region = frame[6:min(frame.shape[0], 6 + height), 6:6 + width]
    cv2.addWeighted(np.zeros_like(region), .72, region, .28, 0, region)
    for index, row in enumerate(rows):
        cv2.putText(frame, row, (14, 26 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1, cv2.LINE_AA)


def draw_event_clip_overlay(frame: np.ndarray, frame_idx: int, window: ClipWindow,
                            tracks: Dict[int, Track], counting_cfg: CountingConfig,
                            cfg: ThermalBlobConfig, clip_idx: int, clip_count: int) -> np.ndarray:
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for track_id in sorted(window.track_ids):
        track = tracks.get(track_id)
        if track is None:
            continue
        detections = [item for item in track.detections if window.start_frame <= item.frame_idx <= frame_idx]
        if cfg.trail_length > 0:
            detections = detections[-cfg.trail_length:]
        color = color_for_track(track_id)
        if len(detections) >= 2:
            polyline = np.asarray([tuple_int(item.centroid) for item in detections], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [polyline], False, color, 2, cv2.LINE_AA)
        current = next((item for item in reversed(detections) if item.frame_idx == frame_idx), None)
        if current is not None:
            cv2.circle(out, tuple_int(current.centroid), 4, color, -1, cv2.LINE_AA)
            cv2.putText(out, str(track_id), tuple_int(current.centroid), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    draw_counting_geometry(out, counting_cfg)
    hud = f"Clip {clip_idx} / {clip_count} | Frame {window.start_frame}-{window.end_frame} | tracks: {len(window.track_ids)} | source: {','.join(sorted(window.sources))}"
    cv2.putText(out, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _polygon_label_point(points: Sequence[Point]) -> Point:
    return (0.0, 0.0) if not points else (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
