"""Background modelling, thermal blob detection, and centroid tracking."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Rect, ThermalBlobConfig
from .models import BlobDetection, Point, Track
from .validation import euclidean_distance, is_valid_flying_track


def blend_background(old_background: Optional[np.ndarray], new_background: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, float(alpha)))
    if old_background is None:
        return new_background
    return ((1.0 - alpha) * old_background + alpha * new_background).astype(np.float32)


def point_in_rects(point: Point, rects: Sequence[Rect]) -> bool:
    x, y = point
    return any(rx <= x <= rx + rw and ry <= y <= ry + rh for rx, ry, rw, rh in rects)


class ThermalBlobDetector:
    """Detect bright moving blobs and link their centroids into tracks."""

    def __init__(self, config: ThermalBlobConfig) -> None:
        self.config = config
        self.background: Optional[np.ndarray] = None
        self.previous_gray: Optional[np.ndarray] = None
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.skipped_detection_frames = 0
        self.discarded_invalid_tracks = 0
        self._track_validity_cache: Dict[int, Tuple[int, bool]] = {}
        self._valid_track_ids: set[int] = set()

    @staticmethod
    def frame_to_gray_float(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return gray.astype(np.float32)

    def build_background_window(self, cap: cv2.VideoCapture, start_frame: int,
                                sample_count: int, stride: int) -> np.ndarray:
        if sample_count <= 0:
            raise ValueError("Background sample count must be greater than zero.")
        if stride <= 0:
            raise ValueError("Background sample stride must be greater than zero.")
        original_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        samples: List[np.ndarray] = []
        try:
            for index in range(sample_count):
                frame_pos = start_frame + index * stride
                if frame_count > 0 and frame_pos >= frame_count:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ok, frame = cap.read()
                if not ok:
                    break
                samples.append(self.frame_to_gray_float(frame))
        finally:
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
        if not samples:
            raise RuntimeError("Could not sample frames for background model.")
        return np.percentile(np.stack(samples), self.config.background_percentile, axis=0).astype(np.float32)

    def build_background(self, cap: cv2.VideoCapture) -> np.ndarray:
        self.background = self.build_background_window(
            cap, 0, self.config.background_frames, self.config.background_stride
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.previous_gray = None
        return self.background

    def recalibrate_background(self, cap: cv2.VideoCapture, start_frame: int) -> np.ndarray:
        new_background = self.build_background_window(
            cap, start_frame, self.config.background_recalibrate_frames,
            self.config.background_recalibrate_stride,
        )
        self.background = blend_background(
            self.background, new_background, self.config.background_recalibrate_blend
        )
        return self.background

    def detect(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[BlobDetection], np.ndarray, np.ndarray]:
        if self.background is None:
            raise RuntimeError("Background is missing. Call build_background() first.")
        gray = self.frame_to_gray_float(frame)
        positive_diff = np.maximum(gray - self.background, 0)
        mask = (positive_diff >= self.config.threshold).astype(np.uint8) * 255
        if self.config.use_motion_gate and self.previous_gray is not None:
            motion = cv2.absdiff(gray, self.previous_gray)
            mask = cv2.bitwise_and(mask, (motion >= self.config.motion_threshold).astype(np.uint8) * 255)
        mask = self._cleanup_mask(self._apply_exclude_zones(self._apply_roi(mask)))
        detections = self._connected_components_to_detections(mask, positive_diff, frame_idx)
        if self.config.max_detections_per_frame > 0 and len(detections) > self.config.max_detections_per_frame:
            detections = []
            self.skipped_detection_frames += 1
        self.previous_gray = gray
        return detections, np.clip(positive_diff, 0, 255).astype(np.uint8), mask

    def update_tracks(self, detections: List[BlobDetection]) -> List[Track]:
        active_tracks = [track for track in self.tracks.values() if track.active]
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = {track.track_id for track in active_tracks}
        candidates = []
        updated_ids: set[int] = set()
        for track in active_tracks:
            for detection_index, detection in enumerate(detections):
                reference = (
                    track.predicted_point_for_frame(detection.frame_idx)
                    if self.config.use_prediction else track.last_point
                )
                if reference is None:
                    continue
                distance = euclidean_distance(reference, detection.centroid)
                if distance <= self.config.max_link_distance:
                    candidates.append((distance, track.track_id, detection_index))
        for _, track_id, detection_index in sorted(candidates):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            self.tracks[track_id].add_detection(detections[detection_index])
            updated_ids.add(track_id)
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.config.max_gap_frames:
                track.active = False
        for detection_index in unmatched_detections:
            track = Track(track_id=self.next_track_id)
            track.add_detection(detections[detection_index])
            self.tracks[track.track_id] = track
            updated_ids.add(track.track_id)
            self.next_track_id += 1
        updated = [self.tracks[track_id] for track_id in updated_ids]
        for track in updated:
            self._refresh_track_validity(track)
        if not self.config.retain_invalid_tracks:
            self._discard_closed_invalid_tracks()
        return updated

    def _discard_closed_invalid_tracks(self) -> None:
        """Release finalized invalid tracks and their detection histories."""
        discarded_ids = [
            track_id for track_id, track in self.tracks.items()
            if not track.active and not self.is_valid_track_cached(track)
        ]
        for track_id in discarded_ids:
            del self.tracks[track_id]
            self._track_validity_cache.pop(track_id, None)
            self._valid_track_ids.discard(track_id)
        self.discarded_invalid_tracks += len(discarded_ids)

    def _refresh_track_validity(self, track: Track) -> bool:
        valid = is_valid_flying_track(track, self.config)
        self._track_validity_cache[track.track_id] = (track.lifetime, valid)
        (self._valid_track_ids.add if valid else self._valid_track_ids.discard)(track.track_id)
        return valid

    def is_valid_track_cached(self, track: Track) -> bool:
        cached = self._track_validity_cache.get(track.track_id)
        return cached[1] if cached and cached[0] == track.lifetime else self._refresh_track_validity(track)

    def drawable_tracks(self) -> List[Track]:
        tracks = (
            (self.tracks[track_id] for track_id in self._valid_track_ids)
            if self.config.draw_valid_only else self.tracks.values()
        )
        return list(tracks) if self.config.draw_inactive_tracks else [track for track in tracks if track.active]

    def confirmed_tracks(self) -> List[Track]:
        return [track for track in self.tracks.values() if track.lifetime >= self.config.min_track_lifetime]

    def valid_tracks(self) -> List[Track]:
        return [self.tracks[track_id] for track_id in self._valid_track_ids]

    @property
    def valid_track_count(self) -> int:
        return len(self._valid_track_ids)

    @property
    def active_track_count(self) -> int:
        return sum(track.active for track in self.tracks.values())

    def _apply_roi(self, mask: np.ndarray) -> np.ndarray:
        if self.config.roi is None:
            return mask
        x, y, width, height = self.config.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y:y + height, x:x + width] = 255
        return cv2.bitwise_and(mask, roi_mask)

    def _apply_exclude_zones(self, mask: np.ndarray) -> np.ndarray:
        if not self.config.exclude_zones:
            return mask
        result = mask.copy()
        for x, y, width, height in self.config.exclude_zones:
            result[y:y + height, x:x + width] = 0
        return result

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.config.morph_open > 0:
            size = max(1, self.config.morph_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        if self.config.morph_dilate > 0:
            size = max(1, self.config.morph_dilate)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        return mask

    def _connected_components_to_detections(self, mask: np.ndarray, positive_diff: np.ndarray,
                                             frame_idx: int) -> List[BlobDetection]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections = []
        for label_id in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label_id])
            if area < self.config.min_area or area > self.config.max_area:
                continue
            cx, cy = centroids[label_id]
            if point_in_rects((float(cx), float(cy)), self.config.exclude_zones):
                continue
            component_diff = positive_diff[labels == label_id]
            if component_diff.size == 0:
                continue
            mean_diff, max_diff = float(np.mean(component_diff)), float(np.max(component_diff))
            detections.append(BlobDetection(
                frame_idx, (float(cx), float(cy)), (x, y, width, height), area,
                mean_diff, max_diff, min(1.0, max_diff / max(1.0, self.config.threshold * 4.0)),
            ))
        return detections
