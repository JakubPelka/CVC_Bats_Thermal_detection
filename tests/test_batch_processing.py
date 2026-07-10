from types import SimpleNamespace
from pathlib import Path

from batch_processing import build_output_paths, collect_input_videos, safe_stem


def args(**overrides):
    values = {
        "input": "", "inputs": None, "input_dir": "", "recursive": False,
        "video_extensions": ".mp4,.avi,.mov,.mkv", "event_clips_dir": "",
        "output": "enabled", "csv": "enabled", "summary_csv": "enabled",
        "crossings_csv": "enabled", "aoi_events_csv": "enabled",
        "activity_csv": "enabled", "run_summary_json": "enabled",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_collects_one_input_file():
    assert collect_input_videos(args(input="night.mp4")) == [Path("night.mp4")]


def test_collects_multiple_explicit_files():
    assert [str(path) for path in collect_input_videos(args(inputs=["a.mp4", "b.avi"]))] == ["a.mp4", "b.avi"]


def test_folder_scan_filters_extensions(tmp_path):
    (tmp_path / "a.MP4").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.avi").touch()
    found = collect_input_videos(args(input_dir=str(tmp_path)))
    assert [path.name for path in found] == ["a.MP4"]


def test_recursive_folder_scan(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "video.mov").touch()
    assert collect_input_videos(args(input_dir=str(tmp_path), recursive=True)) == [nested / "video.mov"]


def test_rejects_multiple_input_modes():
    try:
        collect_input_videos(args(input="a.mp4", inputs=["b.mp4"]))
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("multiple input modes should be rejected")


def test_safe_stem_and_per_video_output_paths(tmp_path):
    namespace = args()
    paths = build_output_paths(Path("Bat night #1.mp4"), tmp_path, namespace)
    assert safe_stem(Path("Bat night #1.mp4")) == "Bat_night__1"
    assert paths["output_dir"] == tmp_path / "Bat_night__1"
    assert paths["track_points_csv"] == tmp_path / "Bat_night__1" / "Bat_night__1_track_points.csv"
    assert paths["event_clips_dir"] == tmp_path / "Bat_night__1" / "Bat_night__1_event_clips"


def test_custom_event_clip_directory_is_used_exactly(tmp_path):
    custom = tmp_path / "clips"
    paths = build_output_paths(Path("night.mp4"), tmp_path, args(event_clips_dir=str(custom)))
    assert paths["event_clips_dir"] == custom
