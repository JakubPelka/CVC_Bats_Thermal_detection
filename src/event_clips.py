"""Post-run event clip window construction and manifest helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Set

@dataclass
class ClipWindow:
    start_frame: int
    end_frame: int
    track_ids: Set[int] = field(default_factory=set)
    event_ids: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    crossing_event_ids: Set[str] = field(default_factory=set)
    aoi_event_ids: Set[str] = field(default_factory=set)


def _clamped_window(start: int, end: int, total_frames: int, **kwargs: Any) -> ClipWindow:
    last_frame = max(0, total_frames - 1)
    return ClipWindow(max(0, start), min(last_frame, end), **kwargs)


def build_clip_windows(
    tracks: Iterable[Any],
    crossings: Iterable[Any],
    aoi_events: Iterable[Any],
    trigger: str,
    pre_frames: int,
    post_frames: int,
    total_frames: int,
    min_track_lifetime: int,
    is_valid_track: Callable[[Any], bool],
) -> List[ClipWindow]:
    """Build unmerged, buffered windows from finalized tracks and events."""
    windows: List[ClipWindow] = []
    include_valid = trigger in {"valid_tracks", "all_events"}
    include_all = trigger == "all_tracks"
    include_crossings = trigger in {"crossings", "all_events"}
    include_aois = trigger in {"aois", "all_events"}

    if include_valid or include_all:
        for track in tracks:
            detections = list(getattr(track, "detections", []))
            if not detections:
                continue
            if include_valid and not is_valid_track(track):
                continue
            if include_all and len(detections) < min_track_lifetime:
                continue
            track_id = int(track.track_id)
            windows.append(_clamped_window(
                int(detections[0].frame_idx) - pre_frames,
                int(detections[-1].frame_idx) + post_frames,
                total_frames,
                track_ids={track_id},
                sources={"valid_track" if include_valid else "all_track"},
            ))

    if include_crossings:
        for event in crossings:
            event_id = str(event.event_id)
            windows.append(_clamped_window(
                int(event.frame) - pre_frames, int(event.frame) + post_frames, total_frames,
                track_ids={int(event.track_id)}, event_ids={event_id},
                crossing_event_ids={event_id}, sources={"crossing"},
            ))

    if include_aois:
        for event in aoi_events:
            event_id = str(event.event_id)
            start = event.start_frame if event.start_frame is not None else event.frame
            end = event.end_frame if event.end_frame is not None else event.frame
            windows.append(_clamped_window(
                int(start) - pre_frames, int(end) + post_frames, total_frames,
                track_ids={int(event.track_id)}, event_ids={event_id},
                aoi_event_ids={event_id}, sources={"aoi"},
            ))
    return windows


def merge_clip_windows(windows: Iterable[ClipWindow], merge_gap_frames: int) -> List[ClipWindow]:
    """Merge overlapping/nearby windows while retaining all provenance."""
    merged: List[ClipWindow] = []
    for window in sorted(windows, key=lambda item: (item.start_frame, item.end_frame)):
        item = ClipWindow(
            window.start_frame, window.end_frame, set(window.track_ids), set(window.event_ids),
            set(window.sources), set(window.crossing_event_ids), set(window.aoi_event_ids),
        )
        if not merged or item.start_frame > merged[-1].end_frame + merge_gap_frames:
            merged.append(item)
            continue
        current = merged[-1]
        current.end_frame = max(current.end_frame, item.end_frame)
        current.track_ids.update(item.track_ids)
        current.event_ids.update(item.event_ids)
        current.sources.update(item.sources)
        current.crossing_event_ids.update(item.crossing_event_ids)
        current.aoi_event_ids.update(item.aoi_event_ids)
    return merged


def build_clip_filename(clip_idx: int, window: ClipWindow) -> str:
    ids = sorted(window.track_ids)
    tracks = str(len(ids))
    return f"clip_{clip_idx:04d}_f{window.start_frame:06d}_f{window.end_frame:06d}_tracks_{tracks}.mp4"


def export_event_clips(
    input_path: Path, output_dir: Path, windows: Sequence[ClipWindow], fps: float,
    width: int, height: int, fourcc: str,
    annotate: Callable[[Any, int, ClipWindow, int, int], Any],
    valid_track_ids: Set[int],
) -> List[dict]:
    import cv2

    if len(fourcc) != 4:
        raise ValueError("event clip fourcc must contain exactly four characters")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not re-open video for event clips: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    try:
        for clip_idx, window in enumerate(windows, start=1):
            filename = build_clip_filename(clip_idx, window)
            output_path = output_dir / filename
            temporary_path = output_dir / f".{output_path.stem}.part{output_path.suffix}"
            print(f"Writing event clip {clip_idx}/{len(windows)}: {output_path}")
            temporary_path.unlink(missing_ok=True)
            writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not create event clip: {temporary_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)
            frame_idx = window.start_frame
            written_frames = 0
            try:
                while frame_idx <= window.end_frame:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(annotate(frame, frame_idx, window, clip_idx, len(windows)))
                    frame_idx += 1
                    written_frames += 1
            except BaseException:
                writer.release()
                temporary_path.unlink(missing_ok=True)
                raise
            else:
                writer.release()
            if written_frames == 0:
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(f"No frames could be read for event clip starting at frame {window.start_frame}")
            temporary_path.replace(output_path)
            actual_end = max(window.start_frame, frame_idx - 1)
            duration_frames = actual_end - window.start_frame + 1
            manifest.append({
                "clip_id": clip_idx, "filename": filename,
                "start_frame": window.start_frame, "end_frame": actual_end,
                "source_start_frame": window.start_frame,
                "source_end_frame": actual_end,
                "source_frame_offset": window.start_frame,
                "duration_frames": duration_frames,
                "start_time_s": window.start_frame / fps, "end_time_s": actual_end / fps,
                "duration_s": duration_frames / fps, "source_types": sorted(window.sources),
                "track_ids": sorted(window.track_ids), "event_ids": sorted(window.event_ids),
                "valid_track_count": len(window.track_ids & valid_track_ids),
                "crossing_count": len(window.crossing_event_ids),
                "aoi_event_count": len(window.aoi_event_ids),
            })
    finally:
        cap.release()
    return manifest


def write_clip_manifest(output_dir: Path, rows: Sequence[dict]) -> None:
    json_path = output_dir / "event_clips_manifest.json"
    csv_path = output_dir / "event_clips_manifest.csv"
    json_path.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ";".join(map(str, value)) if isinstance(value, list) else value for key, value in row.items()})
