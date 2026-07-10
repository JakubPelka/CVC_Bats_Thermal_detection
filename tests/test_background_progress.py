import unittest
import sys
import types

import numpy as np

cv2 = types.SimpleNamespace(CAP_PROP_FRAME_COUNT=7, CAP_PROP_POS_FRAMES=1)
sys.modules.setdefault("cv2", cv2)

from thermal_blob_detector import (
    ThermalBlobConfig,
    ThermalBlobDetector,
    blend_background,
    build_progress_text,
    format_duration,
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


if __name__ == "__main__":
    unittest.main()
