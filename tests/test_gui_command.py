import sys
import tempfile
import unittest
from pathlib import Path

from gui_command import build_detector_command


class GuiCommandTests(unittest.TestCase):
    def test_builds_single_file_command_without_tkinter(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.touch()
            paths = {
                "script": "thermal_blob_detector", "input_mode": "single", "input": str(video),
                "inputs": "", "input_dir": "", "batch_output_dir": "outputs",
                "video_extensions": ".mp4", "output": "out.mp4", "csv": "points.csv",
                "summary_csv": "summary.csv", "crossings_csv": "crossings.csv",
                "aoi_events_csv": "aoi.csv", "activity_csv": "activity.csv",
                "run_summary_json": "run.json", "event_clips_dir": "",
                "event_clip_pre_frames": "100", "event_clip_post_frames": "100",
                "event_clip_merge_gap_frames": "100", "event_clip_trigger": "valid_tracks",
                "event_clip_fourcc": "mp4v", "roi": "", "exclude_zones": "",
                "counting_config": "",
            }
            command = build_detector_command(
                paths, {"threshold": "18"}, {"save_annotated_video": True},
                [("threshold", "--threshold", "float", "18", "Threshold", "")],
                [], "thermal_blob_detector", Path,
            )
        self.assertEqual(command[:4], [sys.executable, "-u", "-m", "thermal_blob_detector"])
        self.assertIn("--input", command)
        self.assertIn("--threshold", command)


if __name__ == "__main__":
    unittest.main()
