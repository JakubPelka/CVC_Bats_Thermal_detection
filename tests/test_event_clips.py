import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from counting_models import CountingConfig
from event_clips import (
    ClipWindow, _clip_datetime_fields, _datetime_from_filename,
    build_clip_filename, build_clip_windows, merge_clip_windows, write_clip_manifest,
)
from thermal_bat.config import ThermalBlobConfig
from thermal_bat.visualization import color_for_track, draw_verification_event_clip_overlay


def track(track_id, frames):
    return SimpleNamespace(
        track_id=track_id,
        detections=[SimpleNamespace(frame_idx=frame) for frame in frames],
    )


def test_one_track_creates_one_clamped_window():
    windows = build_clip_windows([track(1, [5, 10])], [], [], "valid_tracks", 10, 100, 50, 3, lambda _: True)
    assert [(item.start_frame, item.end_frame, item.track_ids) for item in windows] == [(0, 49, {1})]


def test_overlapping_windows_merge_and_keep_metadata():
    windows = [
        ClipWindow(10, 30, {1}, sources={"valid_track"}),
        ClipWindow(20, 40, {2}, sources={"crossing"}),
    ]
    merged = merge_clip_windows(windows, 0)
    assert len(merged) == 1
    assert (merged[0].start_frame, merged[0].end_frame) == (10, 40)
    assert merged[0].track_ids == {1, 2}
    assert merged[0].sources == {"valid_track", "crossing"}


def test_windows_within_merge_gap_merge():
    assert len(merge_clip_windows([ClipWindow(10, 20), ClipWindow(25, 30)], 5)) == 1


def test_distant_windows_remain_separate():
    assert len(merge_clip_windows([ClipWindow(10, 20), ClipWindow(26, 30)], 5)) == 2


def test_all_tracks_excludes_short_noise_tracks():
    windows = build_clip_windows(
        [track(1, [2]), track(2, [4, 5, 6])], [], [], "all_tracks", 0, 0, 20, 3, lambda _: False
    )
    assert [item.track_ids for item in windows] == [{2}]


def test_datetime_is_read_from_unambiguous_filename():
    assert _datetime_from_filename(Path("camera_2026-07-16_21-04-05.mp4")) == datetime(2026, 7, 16, 21, 4, 5)
    assert _datetime_from_filename(Path("camera_20260716_210405.mp4")) == datetime(2026, 7, 16, 21, 4, 5)


def test_ambiguous_or_incomplete_filename_datetime_is_ignored():
    assert _datetime_from_filename(Path("camera_2026-07-16.mp4")) is None
    assert _datetime_from_filename(Path("20260716_210405_copy_20260717_210405.mp4")) is None


def test_clip_datetime_fields_use_absolute_frames_and_handle_midnight():
    fields = _clip_datetime_fields(datetime(2026, 7, 16, 23, 59, 58), 20, 40, 10.0)
    assert fields == {
        "start_date": "2026-07-17", "start_time": "00:00:00",
        "end_date": "2026-07-17", "end_time": "00:00:02",
    }


def test_clip_datetime_fields_are_empty_without_source_time():
    assert _clip_datetime_fields(None, 20, 40, 10.0) == {
        "start_date": "", "start_time": "", "end_date": "", "end_time": "",
    }


def test_raw_right_verification_pane_is_unchanged_source_frame():
    frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape((12, 16, 3))
    cfg = ThermalBlobConfig(verification_left_style="minimal", verification_right_style="raw")

    output = draw_verification_event_clip_overlay(
        frame, 4, ClipWindow(4, 4), {}, CountingConfig(), cfg, 1, 1,
    )

    assert output.shape == (12, 32, 3)
    assert np.array_equal(output[:, 16:], frame)


def test_fixed_track_color_is_shared_by_all_tracks():
    cfg = ThermalBlobConfig(track_color_mode="fixed", track_fixed_color="cyan")
    assert color_for_track(1, cfg) == (255, 229, 0)
    assert color_for_track(99, cfg) == color_for_track(1, cfg)


def test_random_track_colors_remain_deterministic_and_distinct():
    cfg = ThermalBlobConfig(track_color_mode="random")
    assert color_for_track(1, cfg) == color_for_track(1, cfg)
    assert color_for_track(1, cfg) != color_for_track(2, cfg)


def test_clip_manifest_csv_is_excel_friendly_and_keeps_lists_in_one_cell(tmp_path):
    rows = [{
        "clip_id": 1, "track_ids": [12, 34], "event_ids": ["a", "b"],
        "start_date": "2026-07-16", "start_time": "00:36:36",
    }]

    write_clip_manifest(tmp_path, rows)

    raw = (tmp_path / "event_clips_manifest.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    with (tmp_path / "event_clips_manifest.csv").open(newline="", encoding="utf-8-sig") as file_obj:
        parsed = list(csv.DictReader(file_obj, delimiter=";"))
    assert parsed == [{
        "clip_id": "1", "start_date": "2026-07-16", "start_time": "00:36:36",
        "event_ids": "a|b", "track_ids": "12|34",
    }]


def test_clip_filename_contains_recording_name_and_clock_time():
    filename = build_clip_filename(
        Path("Camera night.mp4"), 3, ClipWindow(20, 40),
        datetime(2026, 7, 16, 21, 4, 5), 10.0,
    )
    assert filename == "Camera_night_clip_0003_21-04-07.mp4"


def test_track_ids_is_last_manifest_field_in_csv_and_json(tmp_path):
    rows = [{
        "clip_id": 1, "track_ids": [2], "filename": "clip.mp4",
        "start_date": "2026-07-16", "start_time": "21:04:05",
        "end_date": "2026-07-16", "end_time": "21:04:06",
    }]
    write_clip_manifest(tmp_path, rows, "camera")
    with (tmp_path / "camera_event_clips_manifest.csv").open(encoding="utf-8-sig") as file_obj:
        header = file_obj.readline().strip().split(";")
    assert header[:5] == ["clip_id", "start_date", "start_time", "end_date", "end_time"]
    assert header[-1] == "track_ids"
    payload = __import__("json").loads((tmp_path / "camera_event_clips_manifest.json").read_text())
    assert list(payload[0])[-1] == "track_ids"
