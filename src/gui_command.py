"""Tk-independent construction of detector CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence


def build_detector_command(
    paths: Mapping[str, str],
    numeric: Mapping[str, str],
    boolean: Mapping[str, bool],
    numeric_params: Sequence[tuple],
    boolean_params: Sequence[tuple],
    detector_module: str,
    resolve_counting_config: Callable[[str], Path],
) -> list[str]:
    script = paths["script"].strip()
    if not script:
        raise ValueError("Detector script/module is required.")
    script_path = Path(script)
    if script_path.exists():
        command = [sys.executable, "-u", str(script_path)]
    elif script == detector_module or "." in script:
        command = [sys.executable, "-u", "-m", script]
    else:
        raise ValueError(f"Detector script/module not found:\n{script}")

    mode = paths["input_mode"]
    if mode == "single":
        input_path = paths["input"].strip()
        if not input_path or not Path(input_path).is_file():
            raise ValueError(f"Input video not found:\n{input_path}")
        command += ["--input", input_path]
    elif mode == "multiple":
        inputs = [item.strip() for item in paths["inputs"].replace("\n", ";").split(";") if item.strip()]
        if not inputs:
            raise ValueError("Select at least one input video.")
        missing = next((item for item in inputs if not Path(item).is_file()), None)
        if missing:
            raise ValueError(f"Input video not found:\n{missing}")
        command += ["--inputs", *inputs]
    elif mode == "folder":
        input_dir = paths["input_dir"].strip()
        if not input_dir or not Path(input_dir).is_dir():
            raise ValueError(f"Input folder not found:\n{input_dir}")
        command += ["--input-dir", input_dir]
    else:
        raise ValueError("Choose one input mode: one file, multiple files, or folder.")

    if paths["batch_output_dir"].strip():
        command += ["--batch-output-dir", paths["batch_output_dir"].strip()]
    if paths["video_extensions"].strip():
        command += ["--video-extensions", paths["video_extensions"].strip()]
    annotation_style = paths.get("annotation_style", "bbox-trail").strip() or "bbox-trail"
    command += ["--annotation-style", annotation_style]
    command += [
        "--track-color-mode", paths.get("track_color_mode", "random").strip() or "random",
        "--track-fixed-color", paths.get("track_fixed_color", "cyan").strip() or "cyan",
    ]
    command += [
        "--verification-left-style", paths.get("verification_left_style", "bbox-trail").strip() or "bbox-trail",
        "--verification-right-style", paths.get("verification_right_style", "raw").strip() or "raw",
    ]
    for key, flag in (
        ("recursive", "--recursive"), ("continue_on_error", "--continue-on-error"),
        ("skip_existing", "--skip-existing"), ("output_per_input_folder", "--output-per-input-folder"),
    ):
        if boolean.get(key):
            command.append(flag)

    output_map = (
        ("csv", "--csv"), ("summary_csv", "--summary-csv"),
        ("crossings_csv", "--crossings-csv"), ("aoi_events_csv", "--aoi-events-csv"),
        ("activity_csv", "--activity-csv"), ("run_summary_json", "--run-summary-json"),
    )
    if boolean.get("save_annotated_video"):
        if paths["output"].strip():
            command += ["--output", paths["output"].strip()]
    else:
        command += ["--output", ""]
    for key, flag in output_map:
        if paths[key].strip():
            command += [flag, paths[key].strip()]

    if boolean.get("event_clips"):
        command.append("--event-clips")
        if paths["event_clips_dir"].strip():
            command += ["--event-clips-dir", paths["event_clips_dir"].strip()]
        for key, flag in (
            ("event_clip_pre_frames", "--event-clip-pre-frames"),
            ("event_clip_post_frames", "--event-clip-post-frames"),
            ("event_clip_merge_gap_frames", "--event-clip-merge-gap-frames"),
        ):
            value = paths[key].strip()
            try:
                if int(value) < 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"{key} must be a non-negative integer.") from exc
            command += [flag, value]
        fourcc = paths["event_clip_fourcc"].strip()
        if len(fourcc) != 4:
            raise ValueError("Event clip FourCC must contain exactly four characters.")
        command += ["--event-clip-trigger", paths["event_clip_trigger"].strip(), "--event-clip-fourcc", fourcc]

    numeric_meta = {key: (flag, kind, label) for key, flag, kind, _default, label, _help in numeric_params}
    for key, value in numeric.items():
        value = value.strip()
        if not value:
            continue
        flag, kind, label = numeric_meta[key]
        try:
            int(value) if kind == "int" else float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid value for '{label}': {value}") from exc
        command += [flag, value]
    for key, flag, _label, _default, _help in boolean_params:
        if boolean.get(key):
            command.append(flag)
        elif key == "motion_gate":
            command.append("--no-motion-gate")
        elif key == "show_track_id":
            command.append("--no-show-track-id")

    roi = paths["roi"].strip()
    if roi:
        _validate_rect(roi, "ROI")
        command += ["--roi", roi]
    zones = paths["exclude_zones"].strip().replace(";", "\n")
    for zone in (line.strip() for line in zones.splitlines() if line.strip()):
        _validate_rect(zone, "Exclude zone")
        command += ["--exclude-zone", zone]
    if paths["counting_config"].strip():
        command += ["--counting-config", str(resolve_counting_config(paths["counting_config"].strip()))]
    return command


def _validate_rect(value: str, label: str) -> None:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{label} must use format x,y,w,h. Got: {value}")
    try:
        [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{label} must contain integers only. Got: {value}") from exc
