"""Data models shared by detection, tracking, analysis, and exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


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
        x, y, w, h = self.bbox
        cx, cy = self.centroid
        return {
            "class_name": self.class_name, "confidence": self.score,
            "bbox": [x, y, x + w, y + h], "centroid": [cx, cy],
            "area": self.area, "frame_idx": self.frame_idx,
            "mean_diff": self.mean_diff, "max_diff": self.max_diff,
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
        return self.detections[-1] if self.detections else None

    @property
    def last_point(self) -> Optional[Point]:
        detection = self.last_detection
        return detection.centroid if detection else None

    @property
    def last_frame_idx(self) -> int:
        detection = self.last_detection
        return detection.frame_idx if detection else -1

    @property
    def lifetime(self) -> int:
        return len(self.detections)

    def points(self) -> List[Point]:
        return [detection.centroid for detection in self.detections]

    def recent_points(self, limit: int) -> List[Point]:
        detections = self.detections if limit <= 0 else self.detections[-limit:]
        return [detection.centroid for detection in detections]

    def predicted_point_for_frame(self, frame_idx: int) -> Optional[Point]:
        if not self.detections:
            return None
        if len(self.detections) < 2:
            return self.last_point
        previous, last = self.detections[-2:]
        delta = max(1, last.frame_idx - previous.frame_idx)
        vx = (last.centroid[0] - previous.centroid[0]) / delta
        vy = (last.centroid[1] - previous.centroid[1]) / delta
        ahead = max(1, frame_idx - last.frame_idx)
        return last.centroid[0] + vx * ahead, last.centroid[1] + vy * ahead
