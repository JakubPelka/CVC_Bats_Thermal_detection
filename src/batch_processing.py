"""Pure input collection and output-path helpers for detector batch runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def safe_stem(path: Path) -> str:
    stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in path.stem)
    return stem or "video"


def _video_extensions(value: str) -> set[str]:
    return {
        extension if extension.startswith(".") else f".{extension}"
        for item in value.split(",")
        if (extension := item.strip().lower())
    }


def collect_input_videos(args: Any) -> List[Path]:
    modes = [bool(args.input), bool(args.inputs), bool(args.input_dir)]
    if sum(modes) != 1:
        raise ValueError("Use exactly one of --input, --inputs, or --input-dir")
    if args.input:
        return [Path(args.input)]
    if args.inputs:
        return [Path(value) for value in args.inputs]

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    extensions = _video_extensions(args.video_extensions)
    pattern = "**/*" if args.recursive else "*"
    return sorted(
        path for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def build_output_paths(input_path: Path, batch_output_dir: Path, args: Any) -> Dict[str, Any]:
    stem = safe_stem(input_path)
    output_dir = batch_output_dir / stem
    custom_clips = getattr(args, "event_clips_dir", "")
    return {
        "stem": stem,
        "output_dir": output_dir,
        "annotated_video": output_dir / f"{stem}_annotated.mp4" if args.output else None,
        "track_points_csv": output_dir / f"{stem}_track_points.csv" if args.csv else None,
        "track_summary_csv": output_dir / f"{stem}_track_summary.csv" if args.summary_csv else None,
        "crossings_csv": output_dir / f"{stem}_crossings.csv" if args.crossings_csv else None,
        "aoi_events_csv": output_dir / f"{stem}_aoi_events.csv" if args.aoi_events_csv else None,
        "activity_csv": output_dir / f"{stem}_activity_by_time.csv" if args.activity_csv else None,
        "run_summary_json": output_dir / f"{stem}_run_summary.json" if args.run_summary_json else None,
        "counting_config_out": None,
        "event_clips_dir": Path(custom_clips) if custom_clips else output_dir / f"{stem}_event_clips",
    }


BATCH_FIELDS = [
    "input_path", "status", "error", "output_dir", "run_summary_json",
    "track_points_csv", "track_summary_csv", "crossings_csv", "aoi_events_csv",
    "activity_csv", "event_clips_dir", "processed_frames", "valid_tracks",
    "line_crossings", "aoi_events", "event_clip_count", "elapsed_seconds",
]


def write_batch_summary(output_dir: Path, rows: Sequence[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = [{field: row.get(field, "") for field in BATCH_FIELDS} for row in rows]
    (output_dir / "batch_summary.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    with (output_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=BATCH_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
