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

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from counting_stats import (
    AoiEvent,
    CountingAoi,
    CountingConfig,
    CrossingEvent,
    analyze_tracks,
    aoi_from_cli,
    line_from_cli,
    load_counting_config,
    point_in_aoi,
    write_activity_csv,
    write_aoi_events_csv,
    write_counting_config_json,
    write_crossings_csv,
    write_run_summary_json,
    write_track_summary_csv as write_counting_track_summary_csv,
    _line_points,
    _polyline_crossing_sign,
    _polyline_side,
    _time_s,
)


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]  # x, y, w, h
Rect = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class BlobDetection:
    frame_idx: int
    centroid: Point
    bbox: BBox
    area: int
    mean_diff: float
    max_diff: float
    score: float
    class_name: str = "thermal_blob"

    def as_cvc_detection(self) -> dict:
        """
        Adapter-style dict for later CVC integration.
        BBox is returned as x1,y1,x2,y2.
        """
        x, y, w, h = self.bbox
        cx, cy = self.centroid
        return {
            "class_name": self.class_name,
            "confidence": self.score,
            "bbox": [x, y, x + w, y + h],
            "centroid": [cx, cy],
            "area": self.area,
            "frame_idx": self.frame_idx,
            "mean_diff": self.mean_diff,
            "max_diff": self.max_diff,
        }


@dataclass
class TrackMetrics:
    path_length: float = 0.0
    net_displacement: float = 0.0
    mean_speed: float = 0.0
    max_step_speed: float = 0.0
    directionality: float = 0.0
    frame_span: int = 0


@dataclass
class Track:
    track_id: int
    detections: List[BlobDetection] = field(default_factory=list)
    missed_frames: int = 0
    active: bool = True

    def add_detection(self, detection: BlobDetection) -> None:
        self.detections.append(detection)
        self.missed_frames = 0

    @property
    def last_detection(self) -> Optional[BlobDetection]:
        if not self.detections:
            return None
        return self.detections[-1]

    @property
    def last_point(self) -> Optional[Point]:
        det = self.last_detection
        return det.centroid if det else None

    @property
    def last_frame_idx(self) -> int:
        det = self.last_detection
        return det.frame_idx if det else -1

    @property
    def lifetime(self) -> int:
        return len(self.detections)

    def points(self) -> List[Point]:
        return [d.centroid for d in self.detections]

    def recent_points(self, limit: int) -> List[Point]:
        if limit <= 0:
            return self.points()
        return [d.centroid for d in self.detections[-limit:]]

    def predicted_point_for_frame(self, frame_idx: int) -> Optional[Point]:
        """
        Predict next centroid using last observed velocity.

        This is intentionally simple. It helps when bright/near objects move fast
        and jump far between frames.
        """
        if not self.detections:
            return None
        if len(self.detections) < 2:
            return self.last_point

        prev = self.detections[-2]
        last = self.detections[-1]
        dt = max(1, last.frame_idx - prev.frame_idx)
        vx = (last.centroid[0] - prev.centroid[0]) / dt
        vy = (last.centroid[1] - prev.centroid[1]) / dt
        ahead = max(1, frame_idx - last.frame_idx)
        return last.centroid[0] + vx * ahead, last.centroid[1] + vy * ahead


@dataclass
class ThermalBlobConfig:
    # Detection thresholds. Values depend strongly on camera/video encoding.
    threshold: float = 18.0
    motion_threshold: float = 5.0
    use_motion_gate: bool = False

    # Blob geometry filters.
    min_area: int = 2
    max_area: int = 1200
    min_width: int = 1
    min_height: int = 1
    max_width: int = 80
    max_height: int = 80

    # Morphological cleanup. Small values are intentional for tiny objects.
    morph_open: int = 1
    morph_dilate: int = 1

    # Simple tracker settings.
    max_link_distance: float = 90.0
    max_gap_frames: int = 4
    min_track_lifetime: int = 3
    use_prediction: bool = True

    # Track-level artefact filtering.
    # These are deliberately mild defaults. Tighten them after looking at CSV/debug output.
    draw_valid_only: bool = True
    min_track_displacement: float = 12.0
    min_track_path_length: float = 18.0
    min_mean_speed: float = 0.8
    max_mean_speed: float = 120.0
    min_directionality: float = 0.15

    # If a camera correction/NUC/compression glitch creates too many blobs in one frame,
    # skip detections for that frame.
    max_detections_per_frame: int = 40

    # Static background model settings.
    background_frames: int = 200
    background_stride: int = 10
    background_percentile: float = 50.0  # 50 = median

    # Optional rectangular processing ROI and exclusion zones.
    roi: Optional[Rect] = None
    exclude_zones: List[Rect] = field(default_factory=list)

    # Debug drawing.
    draw_inactive_tracks: bool = True
    trail_length: int = 0  # 0 = full track history
    draw_roi: bool = True
    draw_exclude_zones: bool = True


class ThermalBlobDetector:
    """
    Detector + simple centroid tracker for bright moving thermal blobs.

    Main integration point:
        detections, diff_u8, mask = detector.detect(frame, frame_idx)

    For CVC integration:
        [d.as_cvc_detection() for d in detections]
    """

    def __init__(self, config: ThermalBlobConfig) -> None:
        self.config = config
        self.background: Optional[np.ndarray] = None
        self.previous_gray: Optional[np.ndarray] = None
        self.tracks: Dict[int, Track] = {}
        self.next_track_id: int = 1
        self.skipped_detection_frames: int = 0

    @staticmethod
    def frame_to_gray_float(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return gray.astype(np.float32)

    def build_background(self, cap: cv2.VideoCapture) -> np.ndarray:
        """
        Build a static background model from sampled frames.

        For tripod thermal footage, median background is often more stable than
        frame-to-frame difference because short-lived flying objects disappear
        from the median.
        """
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        samples: List[np.ndarray] = []

        idx = 0
        while len(samples) < self.config.background_frames:
            frame_pos = idx * self.config.background_stride
            if frame_count > 0 and frame_pos >= frame_count:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ok, frame = cap.read()
            if not ok:
                break

            samples.append(self.frame_to_gray_float(frame))
            idx += 1

        if not samples:
            raise RuntimeError("Could not sample frames for background model.")

        stack = np.stack(samples, axis=0)
        self.background = np.percentile(
            stack,
            self.config.background_percentile,
            axis=0,
        ).astype(np.float32)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.previous_gray = None
        return self.background

    def detect(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[BlobDetection], np.ndarray, np.ndarray]:
        """
        Detect bright thermal blobs in one frame.

        Returns:
            detections: list of BlobDetection
            diff_u8: positive difference image for diagnostics
            mask: final binary mask used for connected components
        """
        if self.background is None:
            raise RuntimeError("Background is missing. Call build_background() first.")

        gray = self.frame_to_gray_float(frame)

        # Positive difference: only objects brighter than learned background.
        diff = gray - self.background
        positive_diff = np.maximum(diff, 0)
        bright_mask = (positive_diff >= self.config.threshold).astype(np.uint8) * 255

        if self.config.use_motion_gate and self.previous_gray is not None:
            motion = cv2.absdiff(gray, self.previous_gray)
            motion_mask = (motion >= self.config.motion_threshold).astype(np.uint8) * 255
            mask = cv2.bitwise_and(bright_mask, motion_mask)
        else:
            mask = bright_mask

        mask = self._apply_roi(mask)
        mask = self._apply_exclude_zones(mask)
        mask = self._cleanup_mask(mask)

        detections = self._connected_components_to_detections(mask, positive_diff, frame_idx)

        if self.config.max_detections_per_frame > 0 and len(detections) > self.config.max_detections_per_frame:
            # Likely camera correction, massive noise burst or compression artefact.
            detections = []
            self.skipped_detection_frames += 1

        self.previous_gray = gray
        diff_u8 = np.clip(positive_diff, 0, 255).astype(np.uint8)
        return detections, diff_u8, mask

    def update_tracks(self, detections: List[BlobDetection]) -> None:
        """
        Simple nearest-neighbour tracker.

        Good enough for sparse point-like objects. Later replacement candidate:
        Hungarian matching + Kalman prediction.
        """
        active_tracks = [t for t in self.tracks.values() if t.active]
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = {t.track_id for t in active_tracks}
        candidate_pairs: List[Tuple[float, int, int]] = []

        for track in active_tracks:
            for det_idx, detection in enumerate(detections):
                if self.config.use_prediction:
                    reference_point = track.predicted_point_for_frame(detection.frame_idx)
                else:
                    reference_point = track.last_point

                if reference_point is None:
                    continue

                distance = euclidean_distance(reference_point, detection.centroid)
                if distance <= self.config.max_link_distance:
                    candidate_pairs.append((distance, track.track_id, det_idx))

        candidate_pairs.sort(key=lambda item: item[0])

        for _, track_id, det_idx in candidate_pairs:
            if track_id not in unmatched_tracks or det_idx not in unmatched_detections:
                continue
            self.tracks[track_id].add_detection(detections[det_idx])
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(det_idx)

        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.config.max_gap_frames:
                track.active = False

        for det_idx in unmatched_detections:
            track = Track(track_id=self.next_track_id)
            track.add_detection(detections[det_idx])
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.lifetime >= self.config.min_track_lifetime]

    def valid_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if is_valid_flying_track(t, self.config)]

    def _apply_roi(self, mask: np.ndarray) -> np.ndarray:
        if self.config.roi is None:
            return mask

        x, y, w, h = self.config.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y : y + h, x : x + w] = 255
        return cv2.bitwise_and(mask, roi_mask)

    def _apply_exclude_zones(self, mask: np.ndarray) -> np.ndarray:
        if not self.config.exclude_zones:
            return mask

        out = mask.copy()
        for x, y, w, h in self.config.exclude_zones:
            out[y : y + h, x : x + w] = 0
        return out

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.config.morph_open > 0:
            k = max(1, self.config.morph_open)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if self.config.morph_dilate > 0:
            k = max(1, self.config.morph_dilate)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _connected_components_to_detections(
        self,
        mask: np.ndarray,
        positive_diff: np.ndarray,
        frame_idx: int,
    ) -> List[BlobDetection]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: List[BlobDetection] = []

        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if area < self.config.min_area or area > self.config.max_area:
                continue
            if w < self.config.min_width or h < self.config.min_height:
                continue
            if w > self.config.max_width or h > self.config.max_height:
                continue

            cx, cy = centroids[label_id]
            if point_in_rects((float(cx), float(cy)), self.config.exclude_zones):
                continue

            component_mask = labels == label_id
            component_diff = positive_diff[component_mask]
            if component_diff.size == 0:
                continue

            mean_diff = float(np.mean(component_diff))
            max_diff = float(np.max(component_diff))

            # Simple confidence-like score for sorting/debugging, not ML probability.
            score = min(1.0, max_diff / max(1.0, self.config.threshold * 4.0))

            detections.append(
                BlobDetection(
                    frame_idx=frame_idx,
                    centroid=(float(cx), float(cy)),
                    bbox=(x, y, w, h),
                    area=area,
                    mean_diff=mean_diff,
                    max_diff=max_diff,
                    score=score,
                )
            )

        return detections


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tuple_int(point: Point) -> Tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def point_in_rects(point: Point, rects: Sequence[Rect]) -> bool:
    x, y = point
    for rx, ry, rw, rh in rects:
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return True
    return False


def color_for_track(track_id: int) -> Tuple[int, int, int]:
    # Deterministic pseudo-random BGR color.
    rng = np.random.default_rng(track_id * 9973)
    color = rng.integers(80, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def track_metrics(track: Track) -> TrackMetrics:
    points = track.points()
    if len(points) < 2:
        return TrackMetrics()

    path_length = 0.0
    max_step_speed = 0.0

    for i in range(1, len(points)):
        step_distance = euclidean_distance(points[i - 1], points[i])
        frame_delta = max(1, track.detections[i].frame_idx - track.detections[i - 1].frame_idx)
        step_speed = step_distance / frame_delta
        path_length += step_distance
        max_step_speed = max(max_step_speed, step_speed)

    net_displacement = euclidean_distance(points[0], points[-1])
    first_frame = track.detections[0].frame_idx
    last_frame = track.detections[-1].frame_idx
    frame_span = max(1, last_frame - first_frame)
    mean_speed = path_length / frame_span
    directionality = net_displacement / path_length if path_length > 0 else 0.0

    return TrackMetrics(
        path_length=path_length,
        net_displacement=net_displacement,
        mean_speed=mean_speed,
        max_step_speed=max_step_speed,
        directionality=directionality,
        frame_span=frame_span,
    )


def is_valid_flying_track(track: Track, cfg: ThermalBlobConfig) -> bool:
    """
    Track-level artefact filter.

    The idea is not to decide whether a blob visually looks like a bat.
    The idea is to reject tracks that do not behave like a moving flying object.
    """
    if track.lifetime < cfg.min_track_lifetime:
        return False

    m = track_metrics(track)

    if m.net_displacement < cfg.min_track_displacement:
        return False
    if m.path_length < cfg.min_track_path_length:
        return False
    if m.mean_speed < cfg.min_mean_speed:
        return False
    if m.mean_speed > cfg.max_mean_speed:
        return False
    if m.directionality < cfg.min_directionality:
        return False

    return True


def should_draw_track(track: Track, cfg: ThermalBlobConfig) -> bool:
    if not cfg.draw_inactive_tracks and not track.active:
        return False
    if cfg.draw_valid_only and not is_valid_flying_track(track, cfg):
        return False
    return True


def draw_debug_overlay(
    frame: np.ndarray,
    detections: List[BlobDetection],
    detector: ThermalBlobDetector,
    frame_idx: int,
    counting_cfg: Optional[CountingConfig] = None,
    live_counter: Optional["LiveCounting"] = None,
) -> np.ndarray:
    """
    Debug visualization mode.

    Draws thin track trails plus optional counting geometry/HUD. By default,
    only valid flying tracks are drawn. Use --draw-all-tracks to see also
    short/static/noisy tracks.
    """
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    cfg = detector.config

    for track in detector.tracks.values():
        if not should_draw_track(track, cfg):
            continue

        points = track.recent_points(cfg.trail_length)
        if len(points) < 2:
            continue

        color = color_for_track(track.track_id)

        for i in range(1, len(points)):
            cv2.line(
                out,
                tuple_int(points[i - 1]),
                tuple_int(points[i]),
                color,
                1,
                cv2.LINE_AA,
            )

    if cfg.draw_roi and cfg.roi is not None:
        x, y, w, h = cfg.roi
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 1)

    if cfg.draw_exclude_zones and cfg.exclude_zones:
        for x, y, w, h in cfg.exclude_zones:
            cv2.rectangle(out, (x, y), (x + w, y + h), (120, 120, 120), 1)

    if counting_cfg is not None:
        draw_counting_geometry(out, counting_cfg, live_counter)
    if live_counter is not None:
        draw_counting_hud(out, live_counter, frame_idx)


    return out


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

    def update(self, tracks: Sequence[Track], frame_idx: int) -> None:
        for track in tracks:
            if track.last_frame_idx != frame_idx:
                continue
            if self.cfg.count_valid_tracks_only and not self.is_countable_track(track):
                continue
            self._update_lines(track)
            self._update_aois(track)
        self._refresh_active_aoi_tracks(tracks)

    def _refresh_active_aoi_tracks(self, tracks: Sequence[Track]) -> None:
        active_by_aoi: Dict[str, set[int]] = {aoi.id: set() for aoi in self.cfg.aois}
        for track in tracks:
            if not track.active:
                continue
            if self.cfg.count_valid_tracks_only and not self.is_countable_track(track):
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
            points = _line_points(line)
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


def draw_counting_geometry(frame: np.ndarray, cfg: CountingConfig, live_counter: Optional[LiveCounting]) -> None:
    overlay = frame.copy()
    for aoi in cfg.aois:
        if not aoi.enabled:
            continue
        if aoi.type == "polygon":
            pts = np.array([[tuple_int(point) for point in aoi.coordinates]], dtype=np.int32)
            cv2.polylines(frame, pts, isClosed=True, color=(0, 180, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.fillPoly(overlay, pts, color=(0, 90, 160))
            label_point = tuple_int(_polygon_label_point(aoi.coordinates))
        else:
            x, y, w, h = (int(round(v)) for v in aoi.coordinates)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 90, 160), -1)
            label_point = (x + 4, y + 16)
        seen = 0 if live_counter is None else len(live_counter.aoi_seen_tracks.get(aoi.id, set()))
        active = 0 if live_counter is None else len(live_counter.aoi_active_tracks.get(aoi.id, set()))
        cv2.putText(frame, f"{aoi.name} seen:{seen} in:{active}", label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)
    if cfg.aois:
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for line in cfg.lines:
        if not line.enabled:
            continue
        points = _line_points(line)
        if len(points) < 2:
            continue
        pts = [tuple_int(point) for point in points]
        for idx in range(1, len(pts)):
            cv2.line(frame, pts[idx - 1], pts[idx], (255, 210, 0), 2, cv2.LINE_AA)
        total = 0 if live_counter is None else live_counter.line_totals.get(line.id, 0)
        label = f"{line.name}: {total}"
        cv2.putText(frame, label, (pts[0][0] + 4, pts[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 60), 1, cv2.LINE_AA)


def draw_counting_hud(frame: np.ndarray, live_counter: LiveCounting, frame_idx: int) -> None:
    aoi_seen_total = sum(len(track_ids) for track_ids in live_counter.aoi_seen_tracks.values())
    rows = [f"frame {frame_idx}  crossings {len(live_counter.crossings)}  AOI seen {aoi_seen_total}"]
    for line in live_counter.cfg.lines[:4]:
        total = live_counter.line_totals.get(line.id, 0)
        positive = live_counter.line_direction_totals.get((line.id, line.positive_label), 0)
        negative = live_counter.line_direction_totals.get((line.id, line.negative_label), 0)
        rows.append(f"L {line.name}: {total}  {line.positive_label}:{positive} {line.negative_label}:{negative}")
    for aoi in live_counter.cfg.aois[:4]:
        seen = len(live_counter.aoi_seen_tracks.get(aoi.id, set()))
        active = len(live_counter.aoi_active_tracks.get(aoi.id, set()))
        dwell = live_counter.aoi_last_dwell_s.get(aoi.id)
        dwell_text = "" if dwell is None else f" last {dwell:.1f}s"
        rows.append(f"AOI {aoi.name}: seen {seen} in {active}{dwell_text}")
    if not rows:
        return
    width = min(frame.shape[1] - 12, 520)
    height = 10 + 20 * len(rows)
    hud = frame.copy()
    cv2.rectangle(hud, (6, 6), (6 + width, 6 + height), (0, 0, 0), -1)
    cv2.addWeighted(hud, 0.72, frame, 0.28, 0, frame)
    y = 26
    for row in rows:
        cv2.putText(frame, row, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        y += 20


def _polygon_label_point(points: Sequence[Point]) -> Point:
    if not points:
        return 0.0, 0.0
    return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)


def _csv_row(event, fieldnames: Sequence[str]) -> Dict[str, object]:
    row = asdict(event)
    out: Dict[str, object] = {}
    for key in fieldnames:
        value = row.get(key)
        out[key] = round(value, 4) if isinstance(value, float) else value
    return out


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


def parse_rect(value: Optional[str], argument_name: str = "rectangle") -> Optional[Rect]:
    if not value:
        return None
    parts = [int(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"{argument_name} must use format x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def parse_rect_list(values: Optional[List[str]], argument_name: str = "rectangle") -> List[Rect]:
    if not values:
        return []

    rects: List[Rect] = []
    for value in values:
        rect = parse_rect(value, argument_name)
        if rect is not None:
            rects.append(rect)
    return rects


def process_video(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = Path(args.output) if args.output else None
    csv_path = Path(args.csv) if args.csv else None
    summary_csv_path = Path(args.summary_csv) if args.summary_csv else None
    crossings_csv_path = Path(args.crossings_csv) if args.crossings_csv else None
    aoi_events_csv_path = Path(args.aoi_events_csv) if args.aoi_events_csv else None
    activity_csv_path = Path(args.activity_csv) if args.activity_csv else None
    run_summary_json_path = Path(args.run_summary_json) if args.run_summary_json else None
    counting_config_out_path = Path(args.counting_config_out) if args.counting_config_out else None

    cfg = ThermalBlobConfig(
        threshold=args.threshold,
        motion_threshold=args.motion_threshold,
        use_motion_gate=args.motion_gate,
        min_area=args.min_area,
        max_area=args.max_area,
        min_width=args.min_width,
        min_height=args.min_height,
        max_width=args.max_width,
        max_height=args.max_height,
        morph_open=args.morph_open,
        morph_dilate=args.morph_dilate,
        max_link_distance=args.max_link_distance,
        max_gap_frames=args.max_gap_frames,
        min_track_lifetime=args.min_track_lifetime,
        use_prediction=not args.no_prediction,
        draw_valid_only=not args.draw_all_tracks,
        min_track_displacement=args.min_track_displacement,
        min_track_path_length=args.min_track_path_length,
        min_mean_speed=args.min_mean_speed,
        max_mean_speed=args.max_mean_speed,
        min_directionality=args.min_directionality,
        max_detections_per_frame=args.max_detections_per_frame,
        background_frames=args.background_frames,
        background_stride=args.background_stride,
        background_percentile=args.background_percentile,
        roi=parse_rect(args.roi, "ROI"),
        exclude_zones=parse_rect_list(args.exclude_zone, "exclude zone"),
        draw_inactive_tracks=not args.hide_inactive_tracks,
        trail_length=args.trail_length,
        draw_roi=not args.hide_roi_rectangle,
        draw_exclude_zones=not args.hide_exclude_zones,
    )

    detector = ThermalBlobDetector(cfg)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path}")
    print(f"Video: {width}x{height}, fps={fps:.3f}, frames={frame_count}")
    print("Building background model...")
    detector.build_background(cap)
    print("Background ready.")

    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

    counting_cfg = build_counting_config_from_args(args)
    live_counter = LiveCounting(
        cfg=counting_cfg,
        fps=fps,
        crossings_csv_path=crossings_csv_path,
        aoi_events_csv_path=aoi_events_csv_path,
        is_countable_track=lambda track: is_valid_flying_track(track, cfg),
    )
    processed_frames = 0

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections, _diff_u8, _mask = detector.detect(frame, frame_idx)
            detector.update_tracks(detections)
            live_counter.update(list(detector.tracks.values()), frame_idx)
            debug_frame = draw_debug_overlay(frame, detections, detector, frame_idx, counting_cfg, live_counter)

            if writer is not None:
                writer.write(debug_frame)

            if args.show:
                cv2.imshow("Thermal blob detector MVP - valid tracks", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("Stopped by user.")
                    break

            processed_frames += 1
            if args.max_frames and processed_frames >= args.max_frames:
                break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        live_counter.close()
        if args.show:
            cv2.destroyAllWindows()

    if csv_path:
        write_track_points_csv(csv_path, detector.tracks, fps, cfg)
        print(f"CSV written: {csv_path}")

    counting_results = analyze_tracks(
        tracks=detector.tracks.values(),
        fps=fps,
        cfg=counting_cfg,
        is_valid_track=lambda track: is_valid_flying_track(track, cfg),
        input_video=str(input_path),
        frame_count_processed=processed_frames,
        parameter_preset=args.parameter_preset,
        notes=f"Skipped noisy detection frames: {detector.skipped_detection_frames}",
    )

    if summary_csv_path:
        write_counting_track_summary_csv(summary_csv_path, counting_results.track_summaries)
        print(f"Enhanced track summary CSV written: {summary_csv_path}")

    if crossings_csv_path:
        write_crossings_csv(crossings_csv_path, counting_results.crossings)
        print(f"Crossings CSV written: {crossings_csv_path}")

    if aoi_events_csv_path:
        write_aoi_events_csv(aoi_events_csv_path, counting_results.aoi_events)
        print(f"AOI events CSV written: {aoi_events_csv_path}")

    if activity_csv_path:
        write_activity_csv(activity_csv_path, counting_results.activity_rows)
        print(f"Activity CSV written: {activity_csv_path}")

    if run_summary_json_path:
        write_run_summary_json(run_summary_json_path, counting_results.run_summary)
        print(f"Run summary JSON written: {run_summary_json_path}")

    if counting_config_out_path:
        write_counting_config_json(counting_config_out_path, counting_cfg)
        print(f"Counting config JSON written: {counting_config_out_path}")

    if output_path:
        print(f"Debug video written: {output_path}")

    all_tracks = list(detector.tracks.values())
    confirmed = detector.confirmed_tracks()
    valid = detector.valid_tracks()
    print("Done.")
    print(f"Frames processed: {processed_frames}")
    print(f"Skipped noisy detection frames: {detector.skipped_detection_frames}")
    print(f"All tracks: {len(all_tracks)}")
    print(f"Confirmed tracks >= {cfg.min_track_lifetime} detections: {len(confirmed)}")
    print(f"Valid flying tracks: {len(valid)}")
    print(f"Line crossings: {len(counting_results.crossings)}")
    print(f"AOI events: {len(counting_results.aoi_events)}")


def build_counting_config_from_args(args: argparse.Namespace) -> CountingConfig:
    counting_cfg = CountingConfig()

    if args.counting_config:
        counting_cfg = load_counting_config(Path(args.counting_config))

    for value in args.count_line or []:
        counting_cfg.lines.append(line_from_cli(value))
    for value in args.count_aoi or []:
        counting_cfg.aois.append(aoi_from_cli(value))

    if args.count_all_tracks:
        counting_cfg.count_valid_tracks_only = False
    if args.activity_bin_seconds is not None:
        counting_cfg.activity_bin_seconds = args.activity_bin_seconds
    if args.line_crossing_epsilon is not None:
        counting_cfg.line_crossing_epsilon = args.line_crossing_epsilon
    if args.min_frames_between_same_line_crossing is not None:
        counting_cfg.min_frames_between_same_line_crossing = args.min_frames_between_same_line_crossing
    if args.aoi_boundary_debounce_frames is not None:
        counting_cfg.aoi_boundary_debounce_frames = args.aoi_boundary_debounce_frames
    return counting_cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and track bright moving blobs in tripod thermal video. "
            "Draws valid flight-like tracks by default and can export track-based line/AOI counting statistics."
        )
    )
    parser.add_argument("--input", required=True, help="Input thermal video file")
    parser.add_argument("--output", default="thermal_blob_valid_tracks.mp4", help="Output debug video path")
    parser.add_argument("--csv", default="thermal_blob_track_points.csv", help="Output CSV path for per-detection track points")
    parser.add_argument("--summary-csv", default="thermal_blob_track_summary.csv", help="Enhanced output CSV path for per-track summary")
    parser.add_argument("--crossings-csv", default="crossings.csv", help="Output CSV path for line crossing events")
    parser.add_argument("--aoi-events-csv", default="aoi_events.csv", help="Output CSV path for AOI entry/exit events")
    parser.add_argument("--activity-csv", default="activity_by_time.csv", help="Output CSV path for activity time bins")
    parser.add_argument("--run-summary-json", default="run_summary.json", help="Output JSON path for compact run summary")
    parser.add_argument("--counting-config-out", default="", help="Optional path to save the effective counting config JSON")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional processing limit for quick tests")

    parser.add_argument("--threshold", type=float, default=18.0, help="Brightness difference threshold above background")
    parser.add_argument("--motion-gate", action="store_true", help="Require both bright-above-background and frame-to-frame motion")
    parser.add_argument("--motion-threshold", type=float, default=5.0, help="Frame-to-frame motion threshold")

    parser.add_argument("--min-area", type=int, default=2, help="Minimum blob area in pixels")
    parser.add_argument("--max-area", type=int, default=1200, help="Maximum blob area in pixels")
    parser.add_argument("--min-width", type=int, default=1, help="Minimum blob width")
    parser.add_argument("--min-height", type=int, default=1, help="Minimum blob height")
    parser.add_argument("--max-width", type=int, default=80, help="Maximum blob width")
    parser.add_argument("--max-height", type=int, default=80, help="Maximum blob height")
    parser.add_argument("--morph-open", type=int, default=1, help="Morphological opening kernel size; 0 disables")
    parser.add_argument("--morph-dilate", type=int, default=1, help="Dilation kernel size; 0 disables")

    parser.add_argument("--max-link-distance", type=float, default=90.0, help="Maximum centroid/predicted-centroid distance for track linking")
    parser.add_argument("--max-gap-frames", type=int, default=4, help="How many frames a track may be missing before closing")
    parser.add_argument("--min-track-lifetime", type=int, default=3, help="Minimum detections before a track can be valid")
    parser.add_argument("--no-prediction", action="store_true", help="Disable simple velocity prediction during track linking")

    parser.add_argument("--draw-all-tracks", action="store_true", help="Draw all tracks, including invalid/static/noisy ones")
    parser.add_argument("--min-track-displacement", type=float, default=12.0, help="Minimum start-to-end displacement for valid flying track")
    parser.add_argument("--min-track-path-length", type=float, default=18.0, help="Minimum total path length for valid flying track")
    parser.add_argument("--min-mean-speed", type=float, default=0.8, help="Minimum mean speed in pixels/frame for valid flying track")
    parser.add_argument("--max-mean-speed", type=float, default=120.0, help="Maximum mean speed in pixels/frame for valid flying track")
    parser.add_argument("--min-directionality", type=float, default=0.15, help="Minimum net_displacement/path_length ratio for valid flying track")
    parser.add_argument("--max-detections-per-frame", type=int, default=40, help="Skip frame detections above this count; 0 disables")

    parser.add_argument("--background-frames", type=int, default=200, help="Number of sampled frames for background model")
    parser.add_argument("--background-stride", type=int, default=10, help="Frame step between background samples")
    parser.add_argument("--background-percentile", type=float, default=50.0, help="Background percentile; 50=median")

    parser.add_argument("--roi", default=None, help="Optional rectangular ROI as x,y,w,h")
    parser.add_argument(
        "--exclude-zone",
        action="append",
        default=None,
        help="Optional exclusion rectangle x,y,w,h. Can be repeated for fixed camera artefact zones.",
    )
    parser.add_argument("--hide-inactive-tracks", action="store_true", help="Do not draw inactive tracks")
    parser.add_argument("--trail-length", type=int, default=0, help="Recent points drawn per track. 0 = full track history")
    parser.add_argument("--hide-roi-rectangle", action="store_true", help="Do not draw ROI rectangle")
    parser.add_argument("--hide-exclude-zones", action="store_true", help="Do not draw exclusion-zone rectangles")

    parser.add_argument("--counting-config", default="", help="Optional JSON file with counting lines, AOIs and counting settings")
    parser.add_argument(
        "--count-line",
        action="append",
        default=None,
        help="Counting line as id,name,x1,y1,x2,y2[,positive_label,negative_label]. Can be repeated.",
    )
    parser.add_argument(
        "--count-aoi",
        action="append",
        default=None,
        help="Rectangular counting AOI as id,name,x,y,w,h. Can be repeated.",
    )
    parser.add_argument("--count-all-tracks", action="store_true", help="Diagnostic counting mode: count all tracks, not only valid flying tracks")
    parser.add_argument("--activity-bin-seconds", type=float, default=None, help="Activity statistics bin size in seconds")
    parser.add_argument("--line-crossing-epsilon", type=float, default=None, help="Pixel-side tolerance for line crossing tests")
    parser.add_argument("--min-frames-between-same-line-crossing", type=int, default=None, help="Debounce repeated crossings of the same line by one track")
    parser.add_argument("--aoi-boundary-debounce-frames", type=int, default=None, help="Debounce repeated AOI entry/exit events near a boundary")
    parser.add_argument("--parameter-preset", default="custom", help="Preset label written to run_summary.json")

    parser.add_argument("--fourcc", default="mp4v", help="Output video codec fourcc, e.g. mp4v or XVID")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
