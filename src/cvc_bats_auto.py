"""Sequential folder automation for unattended CVC Bats analysis."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from batch_processing import _video_extensions


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "cvc-bats-auto" / "config.json"
DEFAULT_STATE_PATH = Path.home() / ".local" / "state" / "cvc-bats-auto" / "state.json"


@dataclass(frozen=True)
class FileFingerprint:
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "FileFingerprint":
        stat = path.stat()
        return cls(size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def as_dict(self) -> Dict[str, int]:
        return {"size": self.size, "mtime_ns": self.mtime_ns}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def iter_source_videos(watch_directory: Path, extensions: set[str]) -> Iterable[Path]:
    for path in sorted(watch_directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        try:
            relative_parts = path.relative_to(watch_directory).parts[:-1]
        except ValueError:
            continue
        if any(part.lower().startswith("output-") for part in relative_parts):
            continue
        yield path.resolve()


def fingerprint_matches(entry: Dict[str, Any], fingerprint: FileFingerprint) -> bool:
    return entry.get("size") == fingerprint.size and entry.get("mtime_ns") == fingerprint.mtime_ns


def initialize_state(config: Dict[str, Any], state_path: Path, force: bool = False) -> int:
    if state_path.exists() and not force:
        print(f"State already exists; baseline unchanged: {state_path}")
        return 0
    watch_directory = Path(config["watch_directory"]).expanduser().resolve()
    extensions = _video_extensions(config.get("video_extensions", ".mp4,.avi,.mov,.mkv"))
    entries: Dict[str, Any] = {}
    for path in iter_source_videos(watch_directory, extensions):
        fingerprint = FileFingerprint.from_path(path)
        entries[str(path)] = {**fingerprint.as_dict(), "status": "baseline"}
    write_json_atomic(state_path, {"version": 1, "files": entries})
    print(f"Initialized baseline with {len(entries)} existing video(s): {state_path}")
    return len(entries)


def build_analysis_command(video_path: Path, config: Dict[str, Any]) -> list[str]:
    repo_directory = Path(config["repo_directory"]).expanduser().resolve()
    preset_path = Path(config["preset_path"]).expanduser().resolve()
    preset = load_json(preset_path)
    paths = dict(preset.get("paths", {}))
    numeric = dict(preset.get("numeric", {}))
    boolean = dict(preset.get("boolean", {}))

    # Runtime paths always come from the discovered source video. Output enablement
    # still follows whether the corresponding preset path is empty or non-empty.
    paths.update({
        "script": "thermal_blob_detector",
        "input_mode": "multiple",
        "input": "",
        "inputs": str(video_path),
        "input_dir": "",
        "batch_output_dir": str(video_path.parent / config.get("output_directory_name", "output-default")),
        "event_clips_dir": "",
    })
    counting_config = paths.get("counting_config", "").strip()
    if counting_config:
        candidate = Path(counting_config).expanduser()
        if not candidate.is_absolute():
            candidate = repo_directory / candidate
        paths["counting_config"] = str(candidate.resolve())

    from gui_command import build_detector_command

    numeric_params = [
        (key, f"--{key.replace('_', '-')}", "float", value, key, "")
        for key, value in numeric.items()
    ]
    boolean_flags = {
        "show": "--show", "motion_gate": "--motion-gate", "no_prediction": "--no-prediction",
        "draw_all_tracks": "--draw-all-tracks", "hide_inactive_tracks": "--hide-inactive-tracks",
        "hide_roi_rectangle": "--hide-roi-rectangle", "hide_exclude_zones": "--hide-exclude-zones",
        "count_all_tracks": "--count-all-tracks", "show_track_id": "--show-track-id",
        "verification_mode": "--verification-mode",
    }
    boolean_params = [
        (key, flag, key, False, "") for key, flag in boolean_flags.items()
    ]

    def resolve_counting_config(value: str) -> Path:
        candidate = Path(value).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(f"Counting config not found: {candidate}")
        return candidate

    command = build_detector_command(
        paths, numeric, boolean, numeric_params, boolean_params,
        "thermal_blob_detector", resolve_counting_config,
    )
    command[0] = str(repo_directory / ".venv" / "bin" / "python")
    command += ["--parameter-preset", preset_path.stem]
    return command


class AutoProcessor:
    def __init__(self, config: Dict[str, Any], state_path: Path) -> None:
        self.config = config
        self.state_path = state_path
        self.watch_directory = Path(config["watch_directory"]).expanduser().resolve()
        self.extensions = _video_extensions(config.get("video_extensions", ".mp4,.avi,.mov,.mkv"))
        self.scan_interval = max(1, int(config.get("scan_interval_seconds", 60)))
        self.stable_for = max(0, int(config.get("stable_for_seconds", 180)))
        self.retry_after = max(self.scan_interval, int(config.get("retry_after_seconds", 3600)))
        self.observed: Dict[str, tuple[FileFingerprint, float]] = {}
        self.stop_requested = False

    def request_stop(self, *_args: Any) -> None:
        self.stop_requested = True

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            raise RuntimeError(
                f"Automation state is missing: {self.state_path}. Run the installer or use --initialize first."
            )
        state = load_json(self.state_path)
        state.setdefault("version", 1)
        state.setdefault("files", {})
        return state

    def _ready_videos(self, state: Dict[str, Any], now: float) -> Iterable[tuple[Path, FileFingerprint]]:
        entries = state["files"]
        present = set()
        for path in iter_source_videos(self.watch_directory, self.extensions):
            key = str(path)
            present.add(key)
            try:
                fingerprint = FileFingerprint.from_path(path)
            except FileNotFoundError:
                continue
            entry = entries.get(key, {})
            if entry.get("status") in {"baseline", "completed"} and fingerprint_matches(entry, fingerprint):
                self.observed.pop(key, None)
                continue
            if entry.get("status") == "failed" and fingerprint_matches(entry, fingerprint):
                if now - float(entry.get("attempted_at", 0)) < self.retry_after:
                    continue
            previous = self.observed.get(key)
            if previous is None or previous[0] != fingerprint:
                self.observed[key] = (fingerprint, now)
                continue
            if now - previous[1] >= self.stable_for:
                yield path, fingerprint
        for missing in set(self.observed) - present:
            self.observed.pop(missing, None)

    def run_once(self) -> int:
        state = self._load_state()
        ready = list(self._ready_videos(state, time.time()))
        if not ready:
            return 0
        video_path, fingerprint = ready[0]
        key = str(video_path)
        print(f"Starting automatic analysis: {video_path}", flush=True)
        command = build_analysis_command(video_path, self.config)
        attempted_at = time.time()
        result = subprocess.run(command, cwd=self.config["repo_directory"], check=False)
        status = "completed" if result.returncode == 0 else "failed"
        state["files"][key] = {
            **fingerprint.as_dict(), "status": status,
            "attempted_at": attempted_at, "return_code": result.returncode,
        }
        write_json_atomic(self.state_path, state)
        self.observed.pop(key, None)
        print(f"Automatic analysis {status}: {video_path}", flush=True)
        return 1

    def run_forever(self) -> None:
        print(f"Watching recursively: {self.watch_directory}", flush=True)
        print(f"Preset: {self.config['preset_path']}", flush=True)
        while not self.stop_requested:
            processed = self.run_once()
            if processed:
                continue
            for _ in range(self.scan_interval):
                if self.stop_requested:
                    break
                time.sleep(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch a folder and sequentially analyze new bat videos")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--initialize", action="store_true", help="Record existing videos as the initial baseline")
    parser.add_argument("--force-initialize", action="store_true", help="Replace the state baseline")
    parser.add_argument("--once", action="store_true", help="Run one scan instead of watching continuously")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_json(args.config.expanduser().resolve())
    watch_directory = Path(config["watch_directory"]).expanduser().resolve()
    watch_directory.mkdir(parents=True, exist_ok=True)
    if args.initialize or args.force_initialize:
        initialize_state(config, args.state.expanduser().resolve(), force=args.force_initialize)
        return
    processor = AutoProcessor(config, args.state.expanduser().resolve())
    signal.signal(signal.SIGTERM, processor.request_stop)
    signal.signal(signal.SIGINT, processor.request_stop)
    if args.once:
        processor.run_once()
    else:
        processor.run_forever()


if __name__ == "__main__":
    main()
