#!/usr/bin/env python3
"""
thermal_blob_detector_mvp_v3_valid_tracks.py

Standalone MVP for detecting and tracking bright moving blobs in tripod thermal video.

Scope:
- Detect bright moving thermal blob candidates.
- Track centroid movement across frames.
- Filter tracks by flight-like movement metrics to reduce camera artefacts.
- Draw only thin track trails by default.
- Export CSV with detections, track metrics and validity flags.
- No line/AOI counting logic is included here.

Typical use:
    python thermal_blob_detector_mvp_v3_valid_tracks.py --input sample.mp4 --output tracks.mp4 --csv tracks.csv --show

Useful debug use:
    python thermal_blob_detector_mvp_v3_valid_tracks.py --input sample.mp4 --draw-all-tracks --show
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


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
) -> np.ndarray:
    """
    Clean visualization mode.

    Draws only thin track trails:
    - no centroid dots
    - no IDs / labels
    - no bounding boxes
    - no HUD text

    By default, only valid flying tracks are drawn.
    Use --draw-all-tracks to see also short/static/noisy tracks.
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

    processed_frames = 0

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections, _diff_u8, _mask = detector.detect(frame, frame_idx)
            detector.update_tracks(detections)
            debug_frame = draw_debug_overlay(frame, detections, detector, frame_idx)

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
        if args.show:
            cv2.destroyAllWindows()

    if csv_path:
        write_track_points_csv(csv_path, detector.tracks, fps, cfg)
        print(f"CSV written: {csv_path}")

    if summary_csv_path:
        write_track_summary_csv(summary_csv_path, detector.tracks, cfg)
        print(f"Summary CSV written: {summary_csv_path}")

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and track bright moving blobs in tripod thermal video. "
            "Draws valid flight-like tracks only by default. No line/AOI counting included."
        )
    )
    parser.add_argument("--input", required=True, help="Input thermal video file")
    parser.add_argument("--output", default="thermal_blob_valid_tracks.mp4", help="Output debug video path")
    parser.add_argument("--csv", default="thermal_blob_track_points.csv", help="Output CSV path for per-detection track points")
    parser.add_argument("--summary-csv", default="thermal_blob_track_summary.csv", help="Output CSV path for per-track summary")
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
    parser.add_argument("--fourcc", default="mp4v", help="Output video codec fourcc, e.g. mp4v or XVID")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
