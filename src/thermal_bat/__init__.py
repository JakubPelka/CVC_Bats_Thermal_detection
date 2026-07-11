"""Core package for thermal bat detection, tracking, and export."""

from .config import ThermalBlobConfig
from .detector import ThermalBlobDetector
from .models import BlobDetection, Track, TrackMetrics
from .validation import is_valid_flying_track, track_metrics

__all__ = [
    "BlobDetection", "Track", "TrackMetrics", "ThermalBlobConfig", "ThermalBlobDetector",
    "is_valid_flying_track", "track_metrics",
]
