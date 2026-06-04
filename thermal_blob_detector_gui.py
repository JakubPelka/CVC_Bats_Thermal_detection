#!/usr/bin/env python3
"""
thermal_blob_detector_gui.py

Tkinter launcher for thermal_blob_detector_mvp_v3_valid_tracks.py.

Purpose:
- Change detector parameters from a small GUI.
- Run the detector script without editing Python code.
- Save/load parameter presets as JSON.
- Keep the detector itself separate and stable.

Place this file in the same folder as:
    thermal_blob_detector_mvp_v3_valid_tracks.py

Then run:
    python thermal_blob_detector_gui.py

By default, the OpenCV preview window is enabled.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


APP_TITLE = "Thermal Bat Blob Detector - Parameter GUI"

SCRIPT_DEFAULT_NAME = "thermal_blob_detector_mvp_v3_valid_tracks.py"


NUMERIC_PARAMS: List[Tuple[str, str, str, str, str]] = [
    # key, CLI flag, type, default, label
    ("max_frames", "--max-frames", "int", "0", "Max frames, 0 = full video"),
    ("threshold", "--threshold", "float", "18.0", "Brightness threshold"),
    ("motion_threshold", "--motion-threshold", "float", "5.0", "Motion threshold"),

    ("min_area", "--min-area", "int", "2", "Min blob area"),
    ("max_area", "--max-area", "int", "1200", "Max blob area"),
    ("min_width", "--min-width", "int", "1", "Min blob width"),
    ("min_height", "--min-height", "int", "1", "Min blob height"),
    ("max_width", "--max-width", "int", "80", "Max blob width"),
    ("max_height", "--max-height", "int", "80", "Max blob height"),
    ("morph_open", "--morph-open", "int", "1", "Morph open"),
    ("morph_dilate", "--morph-dilate", "int", "1", "Morph dilate"),

    ("max_link_distance", "--max-link-distance", "float", "90.0", "Max link distance"),
    ("max_gap_frames", "--max-gap-frames", "int", "4", "Max gap frames"),
    ("min_track_lifetime", "--min-track-lifetime", "int", "3", "Min track lifetime"),

    ("min_track_displacement", "--min-track-displacement", "float", "12.0", "Min track displacement"),
    ("min_track_path_length", "--min-track-path-length", "float", "18.0", "Min track path length"),
    ("min_mean_speed", "--min-mean-speed", "float", "0.8", "Min mean speed"),
    ("max_mean_speed", "--max-mean-speed", "float", "120.0", "Max mean speed"),
    ("min_directionality", "--min-directionality", "float", "0.15", "Min directionality"),
    ("max_detections_per_frame", "--max-detections-per-frame", "int", "40", "Max detections/frame, 0 = off"),

    ("background_frames", "--background-frames", "int", "200", "Background frames"),
    ("background_stride", "--background-stride", "int", "10", "Background stride"),
    ("background_percentile", "--background-percentile", "float", "50.0", "Background percentile"),

    ("trail_length", "--trail-length", "int", "0", "Trail length, 0 = full track"),
]

BOOLEAN_PARAMS: List[Tuple[str, str, str, bool]] = [
    # key, CLI flag, label, default
    ("show", "--show", "Show OpenCV preview window", True),
    ("motion_gate", "--motion-gate", "Use motion gate", False),
    ("no_prediction", "--no-prediction", "Disable prediction", False),
    ("draw_all_tracks", "--draw-all-tracks", "Draw all tracks, including invalid", False),
    ("hide_inactive_tracks", "--hide-inactive-tracks", "Hide inactive tracks", False),
    ("hide_roi_rectangle", "--hide-roi-rectangle", "Hide ROI rectangle", False),
    ("hide_exclude_zones", "--hide-exclude-zones", "Hide exclude-zone rectangles", False),
]


class ThermalDetectorGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1160x820")
        self.minsize(1050, 720)

        self.process: Optional[subprocess.Popen[str]] = None
        self.reader_thread: Optional[threading.Thread] = None

        self.path_vars: Dict[str, tk.StringVar] = {}
        self.num_vars: Dict[str, tk.StringVar] = {}
        self.bool_vars: Dict[str, tk.BooleanVar] = {}

        self._create_variables()
        self._create_widgets()
        self._refresh_command_preview()

    def _create_variables(self) -> None:
        script_path = Path(__file__).with_name(SCRIPT_DEFAULT_NAME)

        self.path_vars["script"] = tk.StringVar(value=str(script_path))
        self.path_vars["input"] = tk.StringVar(value="")
        self.path_vars["output"] = tk.StringVar(value="thermal_blob_valid_tracks.mp4")
        self.path_vars["csv"] = tk.StringVar(value="thermal_blob_track_points.csv")
        self.path_vars["summary_csv"] = tk.StringVar(value="thermal_blob_track_summary.csv")
        self.path_vars["roi"] = tk.StringVar(value="")
        self.path_vars["exclude_zones"] = tk.StringVar(value="")

        for key, _flag, _typ, default, _label in NUMERIC_PARAMS:
            self.num_vars[key] = tk.StringVar(value=default)

        for key, _flag, _label, default in BOOLEAN_PARAMS:
            self.bool_vars[key] = tk.BooleanVar(value=default)

        for var in list(self.path_vars.values()) + list(self.num_vars.values()):
            var.trace_add("write", lambda *_: self._refresh_command_preview())
        for var in self.bool_vars.values():
            var.trace_add("write", lambda *_: self._refresh_command_preview())

    def _create_widgets(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        self._build_file_section(top)

        middle = ttk.Frame(root)
        middle.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

        left = ttk.Frame(middle)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        right = ttk.Frame(middle)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self._build_parameter_section(left)
        self._build_command_and_log_section(right)

        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.X)
        self._build_buttons(bottom)

    def _build_file_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Files")
        frame.pack(fill=tk.X)

        self._path_row(frame, "Detector script", "script", self._browse_script, row=0)
        self._path_row(frame, "Input video", "input", self._browse_input, row=1)
        self._path_row(frame, "Output video", "output", self._browse_output, row=2)
        self._path_row(frame, "Track points CSV", "csv", self._browse_csv, row=3)
        self._path_row(frame, "Track summary CSV", "summary_csv", self._browse_summary_csv, row=4)

        frame.columnconfigure(1, weight=1)

    def _path_row(self, parent: ttk.Frame, label: str, key: str, command, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
        entry = ttk.Entry(parent, textvariable=self.path_vars[key])
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, padx=4, pady=3)

    def _build_parameter_section(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_detect = ttk.Frame(notebook, padding=8)
        tab_track = ttk.Frame(notebook, padding=8)
        tab_filter = ttk.Frame(notebook, padding=8)
        tab_bg = ttk.Frame(notebook, padding=8)
        tab_masks = ttk.Frame(notebook, padding=8)
        tab_flags = ttk.Frame(notebook, padding=8)

        notebook.add(tab_detect, text="Detection")
        notebook.add(tab_track, text="Tracking")
        notebook.add(tab_filter, text="Track filters")
        notebook.add(tab_bg, text="Background")
        notebook.add(tab_masks, text="ROI / exclude")
        notebook.add(tab_flags, text="Flags")

        self._params_grid(
            tab_detect,
            [
                "max_frames",
                "threshold",
                "motion_threshold",
                "min_area",
                "max_area",
                "min_width",
                "min_height",
                "max_width",
                "max_height",
                "morph_open",
                "morph_dilate",
            ],
        )
        self._params_grid(
            tab_track,
            [
                "max_link_distance",
                "max_gap_frames",
                "min_track_lifetime",
                "trail_length",
            ],
        )
        self._params_grid(
            tab_filter,
            [
                "min_track_displacement",
                "min_track_path_length",
                "min_mean_speed",
                "max_mean_speed",
                "min_directionality",
                "max_detections_per_frame",
            ],
        )
        self._params_grid(
            tab_bg,
            [
                "background_frames",
                "background_stride",
                "background_percentile",
            ],
        )

        self._build_mask_tab(tab_masks)
        self._build_flags_tab(tab_flags)

        presets = ttk.LabelFrame(parent, text="Quick presets", padding=8)
        presets.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(presets, text="Good current defaults", command=self._preset_current_defaults).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="Loose / rescue more bats", command=self._preset_loose).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="Strict / remove artefacts", command=self._preset_strict).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="Diagnostic all tracks", command=self._preset_diagnostic).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="Quick 500-frame test", command=lambda: self.num_vars["max_frames"].set("500")).pack(fill=tk.X, pady=2)
        ttk.Button(presets, text="Full video", command=lambda: self.num_vars["max_frames"].set("0")).pack(fill=tk.X, pady=2)

    def _params_grid(self, parent: ttk.Frame, keys: List[str]) -> None:
        meta = {key: (flag, typ, default, label) for key, flag, typ, default, label in NUMERIC_PARAMS}

        for row, key in enumerate(keys):
            _flag, typ, _default, label = meta[key]
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
            entry = ttk.Entry(parent, textvariable=self.num_vars[key], width=16)
            entry.grid(row=row, column=1, sticky="w", padx=4, pady=4)
            ttk.Label(parent, text=typ).grid(row=row, column=2, sticky="w", padx=4, pady=4)

        parent.columnconfigure(1, weight=1)

    def _build_mask_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="ROI, optional, format: x,y,w,h").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.path_vars["roi"]).pack(fill=tk.X, pady=(2, 10))

        text = (
            "Exclude zones, optional.\n"
            "Use one rectangle per line or separate with semicolon.\n"
            "Format: x,y,w,h\n\n"
            "Example:\n"
            "100,50,80,80\n"
            "600,20,50,120"
        )
        ttk.Label(parent, text=text, justify=tk.LEFT).pack(anchor="w")
        exclude_entry = tk.Text(parent, height=8, width=42)
        exclude_entry.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        def sync_from_var(*_args) -> None:
            current = exclude_entry.get("1.0", tk.END).strip()
            wanted = self.path_vars["exclude_zones"].get()
            if current != wanted:
                exclude_entry.delete("1.0", tk.END)
                exclude_entry.insert("1.0", wanted)

        def sync_to_var(*_args) -> None:
            value = exclude_entry.get("1.0", tk.END).strip()
            if self.path_vars["exclude_zones"].get() != value:
                self.path_vars["exclude_zones"].set(value)

        exclude_entry.bind("<KeyRelease>", sync_to_var)
        exclude_entry.bind("<FocusOut>", sync_to_var)
        self.path_vars["exclude_zones"].trace_add("write", sync_from_var)

    def _build_flags_tab(self, parent: ttk.Frame) -> None:
        for key, _flag, label, _default in BOOLEAN_PARAMS:
            cb = ttk.Checkbutton(parent, text=label, variable=self.bool_vars[key])
            cb.pack(anchor="w", pady=3)

    def _build_command_and_log_section(self, parent: ttk.Frame) -> None:
        command_frame = ttk.LabelFrame(parent, text="Generated command", padding=6)
        command_frame.pack(fill=tk.X)

        self.command_text = tk.Text(command_frame, height=5, wrap=tk.WORD)
        self.command_text.pack(fill=tk.X, expand=False)

        ttk.Button(command_frame, text="Copy command", command=self._copy_command).pack(anchor="e", pady=(4, 0))

        log_frame = ttk.LabelFrame(parent, text="Run log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.log_text = ScrolledText(log_frame, height=20, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_buttons(self, parent: ttk.Frame) -> None:
        self.run_button = ttk.Button(parent, text="Run detector", command=self._run_detector)
        self.run_button.pack(side=tk.LEFT, padx=(0, 6))

        self.stop_button = ttk.Button(parent, text="Stop", command=self._stop_detector, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=6)

        ttk.Button(parent, text="Clear log", command=self._clear_log).pack(side=tk.LEFT, padx=6)
        ttk.Button(parent, text="Save preset JSON", command=self._save_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(parent, text="Load preset JSON", command=self._load_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(parent, text="Open output folder", command=self._open_output_folder).pack(side=tk.LEFT, padx=6)

    def _browse_script(self) -> None:
        path = filedialog.askopenfilename(
            title="Select detector script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if path:
            self.path_vars["script"].set(path)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.path_vars["input"].set(path)
        self._suggest_outputs_from_input(Path(path))

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select output video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"), ("All files", "*.*")],
        )
        if path:
            self.path_vars["output"].set(path)

    def _browse_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select track points CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.path_vars["csv"].set(path)

    def _browse_summary_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select track summary CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.path_vars["summary_csv"].set(path)

    def _suggest_outputs_from_input(self, input_path: Path) -> None:
        stem = input_path.stem
        parent = input_path.parent
        self.path_vars["output"].set(str(parent / f"{stem}_valid_tracks.mp4"))
        self.path_vars["csv"].set(str(parent / f"{stem}_track_points.csv"))
        self.path_vars["summary_csv"].set(str(parent / f"{stem}_track_summary.csv"))

    def _preset_current_defaults(self) -> None:
        values = {
            "max_frames": "0",
            "threshold": "18.0",
            "motion_threshold": "5.0",
            "min_area": "2",
            "max_area": "1200",
            "min_width": "1",
            "min_height": "1",
            "max_width": "80",
            "max_height": "80",
            "morph_open": "1",
            "morph_dilate": "1",
            "max_link_distance": "90.0",
            "max_gap_frames": "4",
            "min_track_lifetime": "3",
            "min_track_displacement": "12.0",
            "min_track_path_length": "18.0",
            "min_mean_speed": "0.8",
            "max_mean_speed": "120.0",
            "min_directionality": "0.15",
            "max_detections_per_frame": "40",
            "background_frames": "200",
            "background_stride": "10",
            "background_percentile": "50.0",
            "trail_length": "0",
        }
        self._set_numeric_values(values)
        self.bool_vars["motion_gate"].set(False)
        self.bool_vars["draw_all_tracks"].set(False)
        self.bool_vars["no_prediction"].set(False)

    def _preset_loose(self) -> None:
        values = {
            "min_track_lifetime": "2",
            "min_track_displacement": "6.0",
            "min_track_path_length": "10.0",
            "min_mean_speed": "0.4",
            "min_directionality": "0.05",
            "max_detections_per_frame": "80",
            "max_link_distance": "110.0",
            "max_gap_frames": "6",
        }
        self._set_numeric_values(values)
        self.bool_vars["draw_all_tracks"].set(False)

    def _preset_strict(self) -> None:
        values = {
            "min_track_lifetime": "4",
            "min_track_displacement": "18.0",
            "min_track_path_length": "30.0",
            "min_mean_speed": "1.2",
            "min_directionality": "0.25",
            "max_detections_per_frame": "30",
            "max_link_distance": "80.0",
            "max_gap_frames": "3",
        }
        self._set_numeric_values(values)
        self.bool_vars["draw_all_tracks"].set(False)

    def _preset_diagnostic(self) -> None:
        values = {
            "min_track_lifetime": "1",
            "min_track_displacement": "0.0",
            "min_track_path_length": "0.0",
            "min_mean_speed": "0.0",
            "min_directionality": "0.0",
            "max_detections_per_frame": "0",
            "trail_length": "0",
        }
        self._set_numeric_values(values)
        self.bool_vars["draw_all_tracks"].set(True)

    def _set_numeric_values(self, values: Dict[str, str]) -> None:
        for key, value in values.items():
            if key in self.num_vars:
                self.num_vars[key].set(value)

    def _build_command(self) -> List[str]:
        script = Path(self.path_vars["script"].get().strip())
        input_video = self.path_vars["input"].get().strip()

        if not script.exists():
            raise ValueError(f"Detector script not found:\n{script}")
        if not input_video:
            raise ValueError("Input video is required.")
        if not Path(input_video).exists():
            raise ValueError(f"Input video not found:\n{input_video}")

        cmd: List[str] = [sys.executable, "-u", str(script), "--input", input_video]

        output = self.path_vars["output"].get().strip()
        csv_path = self.path_vars["csv"].get().strip()
        summary_csv = self.path_vars["summary_csv"].get().strip()

        if output:
            cmd += ["--output", output]
        if csv_path:
            cmd += ["--csv", csv_path]
        if summary_csv:
            cmd += ["--summary-csv", summary_csv]

        self._validate_numeric_params()

        meta = {key: (flag, typ) for key, flag, typ, _default, _label in NUMERIC_PARAMS}
        for key, var in self.num_vars.items():
            value = var.get().strip()
            if value == "":
                continue
            flag, _typ = meta[key]
            cmd += [flag, value]

        for key, flag, _label, _default in BOOLEAN_PARAMS:
            if self.bool_vars[key].get():
                cmd.append(flag)

        roi = self.path_vars["roi"].get().strip()
        if roi:
            self._validate_rect(roi, "ROI")
            cmd += ["--roi", roi]

        for zone in self._parse_exclude_zones():
            self._validate_rect(zone, "Exclude zone")
            cmd += ["--exclude-zone", zone]

        return cmd

    def _validate_numeric_params(self) -> None:
        meta = {key: (typ, label) for key, _flag, typ, _default, label in NUMERIC_PARAMS}

        for key, var in self.num_vars.items():
            value = var.get().strip()
            if value == "":
                continue

            typ, label = meta[key]
            try:
                if typ == "int":
                    int(value)
                elif typ == "float":
                    float(value)
                else:
                    raise ValueError(f"Unsupported type: {typ}")
            except ValueError as exc:
                raise ValueError(f"Invalid value for '{label}': {value}") from exc

    def _validate_rect(self, value: str, label: str) -> None:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 4:
            raise ValueError(f"{label} must use format x,y,w,h. Got: {value}")
        try:
            [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"{label} must contain integers only. Got: {value}") from exc

    def _parse_exclude_zones(self) -> List[str]:
        raw = self.path_vars["exclude_zones"].get().strip()
        if not raw:
            return []

        normalized = raw.replace(";", "\n")
        zones = [line.strip() for line in normalized.splitlines() if line.strip()]
        return zones

    def _refresh_command_preview(self) -> None:
        if not hasattr(self, "command_text"):
            return

        try:
            cmd = self._build_command()
            preview = subprocess.list2cmdline(cmd)
        except Exception as exc:
            preview = f"Command not ready: {exc}"

        self.command_text.configure(state=tk.NORMAL)
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert("1.0", preview)
        self.command_text.configure(state=tk.NORMAL)

    def _copy_command(self) -> None:
        text = self.command_text.get("1.0", tk.END).strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        self.append_log("Command copied to clipboard.\n")

    def _run_detector(self) -> None:
        if self.process is not None:
            messagebox.showinfo(APP_TITLE, "Detector is already running.")
            return

        try:
            cmd = self._build_command()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        script_path = Path(self.path_vars["script"].get().strip())
        cwd = str(script_path.parent)

        self.append_log("\n=== RUN START ===\n")
        self.append_log(subprocess.list2cmdline(cmd) + "\n\n")

        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            self.process = None
            self.run_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            messagebox.showerror(APP_TITLE, f"Could not start detector:\n{exc}")
            return

        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _read_process_output(self) -> None:
        assert self.process is not None

        proc = self.process
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    self.after(0, self.append_log, line)
        finally:
            return_code = proc.wait()
            self.after(0, self._process_finished, return_code)

    def _process_finished(self, return_code: int) -> None:
        self.append_log(f"\n=== RUN FINISHED, return code {return_code} ===\n")
        self.process = None
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def _stop_detector(self) -> None:
        if self.process is None:
            return

        self.append_log("\nStopping detector...\n")
        try:
            self.process.terminate()
        except Exception as exc:
            self.append_log(f"Could not terminate process: {exc}\n")

    def append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", tk.END)

    def _collect_settings(self) -> Dict[str, object]:
        return {
            "paths": {key: var.get() for key, var in self.path_vars.items()},
            "numeric": {key: var.get() for key, var in self.num_vars.items()},
            "boolean": {key: var.get() for key, var in self.bool_vars.items()},
        }

    def _apply_settings(self, data: Dict[str, object]) -> None:
        paths = data.get("paths", {})
        numeric = data.get("numeric", {})
        boolean = data.get("boolean", {})

        if isinstance(paths, dict):
            for key, value in paths.items():
                if key in self.path_vars:
                    self.path_vars[key].set(str(value))

        if isinstance(numeric, dict):
            for key, value in numeric.items():
                if key in self.num_vars:
                    self.num_vars[key].set(str(value))

        if isinstance(boolean, dict):
            for key, value in boolean.items():
                if key in self.bool_vars:
                    self.bool_vars[key].set(bool(value))

    def _save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save parameter preset",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        data = self._collect_settings()
        try:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.append_log(f"Preset saved: {path}\n")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not save preset:\n{exc}")

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Load parameter preset",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._apply_settings(data)
            self.append_log(f"Preset loaded: {path}\n")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not load preset:\n{exc}")

    def _open_output_folder(self) -> None:
        candidates = [
            self.path_vars["output"].get().strip(),
            self.path_vars["csv"].get().strip(),
            self.path_vars["summary_csv"].get().strip(),
            self.path_vars["input"].get().strip(),
        ]

        folder: Optional[Path] = None
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            folder = path if path.is_dir() else path.parent
            if str(folder):
                break

        if folder is None:
            messagebox.showinfo(APP_TITLE, "No output/input path selected.")
            return

        try:
            open_folder(folder)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open folder:\n{exc}")


def open_folder(path: Path) -> None:
    path = path.resolve()

    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def main() -> None:
    app = ThermalDetectorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
