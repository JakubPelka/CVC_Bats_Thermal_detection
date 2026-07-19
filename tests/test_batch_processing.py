from types import SimpleNamespace
from pathlib import Path

from batch_processing import build_output_paths, collect_input_videos, safe_stem
from cvc_bats_auto import build_analysis_command, initialize_state, iter_source_videos, load_json


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
    assert paths["output_dir"] == tmp_path
    assert paths["track_points_csv"] == tmp_path / "Bat_night__1_track_points.csv"
    assert paths["event_clips_dir"] == tmp_path


def test_optional_per_input_folder_keeps_results_together(tmp_path):
    paths = build_output_paths(
        Path("Bat night #1.mp4"), tmp_path, args(output_per_input_folder=True)
    )
    assert paths["output_dir"] == tmp_path / "Bat_night__1"
    assert paths["track_points_csv"] == tmp_path / "Bat_night__1" / "Bat_night__1_track_points.csv"
    assert paths["event_clips_dir"] == tmp_path / "Bat_night__1"


def test_custom_event_clip_directory_is_used_exactly(tmp_path):
    custom = tmp_path / "clips"
    paths = build_output_paths(Path("night.mp4"), tmp_path, args(event_clips_dir=str(custom)))
    assert paths["event_clips_dir"] == custom


def test_auto_scan_is_recursive_and_ignores_output_directories(tmp_path):
    nested = tmp_path / "camera"
    nested.mkdir()
    source = nested / "night.mp4"
    source.touch()
    output = nested / "output-default" / "night"
    output.mkdir(parents=True)
    (output / "clip.mp4").touch()

    found = list(iter_source_videos(tmp_path, {".mp4"}))

    assert found == [source.resolve()]


def test_auto_initialize_marks_existing_videos_as_baseline(tmp_path):
    video = tmp_path / "existing.mp4"
    video.write_bytes(b"video")
    state_path = tmp_path / "state.json"
    config = {"watch_directory": str(tmp_path), "video_extensions": ".mp4"}

    assert initialize_state(config, state_path) == 1

    entry = load_json(state_path)["files"][str(video.resolve())]
    assert entry["status"] == "baseline"
    assert entry["size"] == 5


def test_auto_command_overrides_preset_runtime_paths(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    video = tmp_path / "camera" / "night.mp4"
    video.parent.mkdir()
    video.touch()
    config = {
        "repo_directory": str(repo),
        "preset_path": str(repo / "presets" / "default.json"),
        "output_directory_name": "output-default",
    }

    command = build_analysis_command(video, config)

    assert command[command.index("--inputs") + 1] == str(video)
    assert command[command.index("--batch-output-dir") + 1] == str(video.parent / "output-default")
    assert command[command.index("--parameter-preset") + 1] == "default"
    assert str(repo / "outputs") not in command
