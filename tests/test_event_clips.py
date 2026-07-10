from types import SimpleNamespace

from event_clips import ClipWindow, build_clip_windows, merge_clip_windows


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

