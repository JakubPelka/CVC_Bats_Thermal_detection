import unittest
import sys
import types
import inspect

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = types.SimpleNamespace(CAP_PROP_FRAME_COUNT=7, CAP_PROP_POS_FRAMES=1)
    sys.modules.setdefault("cv2", cv2)

from thermal_blob_detector import (
    BlobDetection,
    Track,
    ThermalBlobConfig,
    ThermalBlobDetector,
    blend_background,
    build_progress_text,
    format_duration,
    is_valid_flying_track,
    draw_counting_geometry,
)


class FakeCapture:
    def __init__(self, values):
        self.frames = [np.full((2, 2), value, dtype=np.uint8) for value in values]
        self.position = 0.0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = float(value)
            return True
        return False

    def read(self):
        index = int(self.position)
        if index >= len(self.frames):
            return False, None
        self.position += 1
        return True, self.frames[index]


class BackgroundAndProgressTests(unittest.TestCase):
    def test_background_window_restores_capture_position(self):
        cap = FakeCapture([0, 10, 20, 30, 40])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
        detector = ThermalBlobDetector(ThermalBlobConfig(background_percentile=50.0))

        background = detector.build_background_window(cap, start_frame=1, sample_count=2, stride=2)

        self.assertEqual(cap.get(cv2.CAP_PROP_POS_FRAMES), 2)
        np.testing.assert_array_equal(background, np.full((2, 2), 20, dtype=np.float32))

    def test_blend_background_clamps_alpha(self):
        old = np.full((1, 1), 10, dtype=np.float32)
        new = np.full((1, 1), 30, dtype=np.float32)
        self.assertEqual(float(blend_background(old, new, 0.25)[0, 0]), 15.0)
        self.assertEqual(float(blend_background(old, new, 2.0)[0, 0]), 30.0)

    def test_progress_with_known_and_unknown_total(self):
        self.assertEqual(format_duration(3661.9), "01:01:01")
        known = build_progress_text(49, 100, start_time=10.0, current_time=20.0)
        self.assertEqual(known, "Frame 50 / 100 (50.0%) | elapsed 00:00:10 | ETA 00:00:10 | 5.0 fps")
        unknown = build_progress_text(49, 0, start_time=10.0, current_time=20.0)
        self.assertEqual(unknown, "Frame 50 | elapsed 00:00:10 | 5.0 fps")

    def _area_filter_config(self):
        return ThermalBlobConfig(
            min_track_lifetime=2, min_track_displacement=0.0,
            min_track_path_length=0.0, min_mean_speed=0.0,
            max_mean_speed=1000.0, min_directionality=0.0,
            min_track_max_blob_area=8,
            min_track_mean_blob_area=0.0,
        )

    def test_min_track_max_blob_area_rejects_tiny_noise_track(self):
        track = Track(track_id=1, detections=[
            BlobDetection(0, (0.0, 0.0), (0, 0, 2, 2), 4, 20.0, 30.0, 0.5),
            BlobDetection(1, (10.0, 0.0), (10, 0, 2, 2), 4, 20.0, 30.0, 0.5),
        ])
        self.assertFalse(is_valid_flying_track(track, self._area_filter_config()))

    def test_min_track_max_blob_area_accepts_one_large_detection(self):
        track = Track(track_id=1, detections=[
            BlobDetection(0, (0.0, 0.0), (0, 0, 2, 2), 4, 20.0, 30.0, 0.5),
            BlobDetection(1, (10.0, 0.0), (10, 0, 3, 3), 9, 20.0, 30.0, 0.5),
        ])
        self.assertTrue(is_valid_flying_track(track, self._area_filter_config()))

    def test_min_track_mean_blob_area_rejects_track_made_from_tiny_blobs(self):
        cfg = self._area_filter_config()
        cfg.min_track_max_blob_area = 0
        cfg.min_track_mean_blob_area = 8.0
        track = Track(track_id=1, detections=[
            BlobDetection(0, (0.0, 0.0), (0, 0, 2, 2), 4, 20.0, 30.0, 0.5),
            BlobDetection(1, (10.0, 0.0), (10, 0, 3, 3), 9, 20.0, 30.0, 0.5),
        ])
        self.assertFalse(is_valid_flying_track(track, cfg))

    def test_min_track_mean_blob_area_accepts_sustained_larger_blobs(self):
        cfg = self._area_filter_config()
        cfg.min_track_max_blob_area = 0
        cfg.min_track_mean_blob_area = 8.0
        track = Track(track_id=1, detections=[
            BlobDetection(0, (0.0, 0.0), (0, 0, 3, 3), 8, 20.0, 30.0, 0.5),
            BlobDetection(1, (10.0, 0.0), (10, 0, 3, 3), 10, 20.0, 30.0, 0.5),
        ])
        self.assertTrue(is_valid_flying_track(track, cfg))

    def test_closed_invalid_track_is_discarded(self):
        cfg = self._area_filter_config()
        cfg.max_gap_frames = 0
        detector = ThermalBlobDetector(cfg)
        detector.tracks[1] = Track(track_id=1, detections=[
            BlobDetection(0, (0.0, 0.0), (0, 0, 1, 1), 1, 1.0, 1.0, 0.1),
        ])

        detector.update_tracks([])

        self.assertNotIn(1, detector.tracks)
        self.assertEqual(detector.discarded_invalid_tracks, 1)

    def test_diagnostic_mode_retains_closed_invalid_track(self):
        cfg = self._area_filter_config()
        cfg.max_gap_frames = 0
        cfg.retain_invalid_tracks = True
        detector = ThermalBlobDetector(cfg)
        detector.tracks[1] = Track(track_id=1, detections=[
            BlobDetection(0, (0.0, 0.0), (0, 0, 1, 1), 1, 1.0, 1.0, 0.1),
        ])

        detector.update_tracks([])

        self.assertIn(1, detector.tracks)
        self.assertEqual(detector.discarded_invalid_tracks, 0)

    def test_counting_geometry_live_counter_is_optional_for_event_clips(self):
        parameter = inspect.signature(draw_counting_geometry).parameters["live_counter"]
        self.assertIsNone(parameter.default)


if __name__ == "__main__":
    unittest.main()
