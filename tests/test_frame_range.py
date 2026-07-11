import unittest

from thermal_bat.pipeline import resolve_frame_range


class FrameRangeTests(unittest.TestCase):
    def test_full_video_range(self):
        self.assertEqual(resolve_frame_range(100, 0, 0), (0, 99, 100))

    def test_inclusive_partial_range(self):
        self.assertEqual(resolve_frame_range(100000, 42000, 46000), (42000, 46000, 4001))

    def test_end_and_max_frames_are_clamped(self):
        self.assertEqual(resolve_frame_range(100, 90, 200), (90, 99, 10))
        self.assertEqual(resolve_frame_range(100, 20, 80, 5), (20, 24, 5))

    def test_invalid_range_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_frame_range(100, 50, 40)


if __name__ == "__main__":
    unittest.main()
