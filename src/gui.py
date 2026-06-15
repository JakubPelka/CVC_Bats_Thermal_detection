#!/usr/bin/env python3
"""
gui.py

Tkinter launcher for thermal_blob_detector.

Purpose:
- Change detector parameters from a small GUI.
- Run the detector script without editing Python code.
- Save/load parameter presets as JSON.
- Keep the detector itself separate and stable.

Then run:
    python -m gui

By default, the OpenCV preview window is enabled. Parameter tabs include an explanation column.
"""

from __future__ import annotations

import base64
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

import cv2


APP_TITLE = "Thermal Bat Blob Detector - Parameter GUI"

DETECTOR_MODULE = "thermal_blob_detector"


NUMERIC_PARAMS: List[Tuple[str, str, str, str, str, str]] = [
    # key, CLI flag, type, default, label, explanation
    ("max_frames", "--max-frames", "int", "0", "Max frames, 0 = full video", "Limits processing for quick tests. Use 300-1000 while tuning; 0 processes the full video."),

    ("threshold", "--threshold", "float", "18.0", "Brightness threshold", "Minimum brightness above background. Increase to remove weak noise; decrease to catch faint bats."),
    ("motion_threshold", "--motion-threshold", "float", "5.0", "Motion threshold", "Used only with motion gate. Increase to require stronger frame-to-frame movement."),

    ("min_area", "--min-area", "int", "2", "Min blob area", "Smallest accepted blob in pixels. Increase to remove tiny hot pixels/noise."),
    ("max_area", "--max-area", "int", "1200", "Max blob area", "Largest accepted blob. Increase if close/bright bats are rejected; decrease to remove large artefacts."),
    ("min_width", "--min-width", "int", "1", "Min blob width", "Minimum blob width in pixels. Usually keep at 1 for small thermal targets."),
    ("min_height", "--min-height", "int", "1", "Min blob height", "Minimum blob height in pixels. Usually keep at 1 for small thermal targets."),
    ("max_width", "--max-width", "int", "80", "Max blob width", "Rejects wide blobs. Increase if bats become streaks; decrease to remove broad artefacts."),
    ("max_height", "--max-height", "int", "80", "Max blob height", "Rejects tall blobs. Increase if close bats are tall; decrease to remove large artefacts."),
    ("morph_open", "--morph-open", "int", "1", "Morph open", "Mask cleanup. Higher removes small noise but may delete very tiny/faint bats."),
    ("morph_dilate", "--morph-dilate", "int", "1", "Morph dilate", "Expands blobs slightly. Higher can connect broken pixels but may merge nearby objects."),

    ("max_link_distance", "--max-link-distance", "float", "90.0", "Max link distance", "Maximum pixel jump between frames for the same track. Increase for fast/near bats; decrease to avoid wrong linking."),
    ("max_gap_frames", "--max-gap-frames", "int", "4", "Max gap frames", "How many missing frames a track can survive. Increase if bats blink/disappear briefly."),
    ("min_track_lifetime", "--min-track-lifetime", "int", "3", "Min track lifetime", "Minimum detections before a track is confirmed. Increase to remove flashes; decrease to keep short fast passes."),

    ("min_track_displacement", "--min-track-displacement", "float", "12.0", "Min track displacement", "Minimum start-to-end movement. Increase to remove stationary artefacts; decrease if valid short tracks vanish."),
    ("min_track_path_length", "--min-track-path-length", "float", "18.0", "Min track path length", "Minimum total travelled path. Increase to remove small jitter; decrease for short visible passes."),
    ("min_mean_speed", "--min-mean-speed", "float", "0.8", "Min mean speed", "Minimum average speed in px/frame. Increase to remove slow drift/hot pixels."),
    ("max_mean_speed", "--max-mean-speed", "float", "120.0", "Max mean speed", "Maximum average speed in px/frame. Decrease if tracks jump unrealistically between artefacts."),
    ("min_directionality", "--min-directionality", "float", "0.15", "Min directionality", "Straightness ratio: net movement / total path. Low allows chaotic bat flight; higher removes jittering noise."),
    ("max_detections_per_frame", "--max-detections-per-frame", "int", "40", "Max detections/frame, 0 = off", "Rejects overloaded frames, e.g. camera calibration/noise bursts. Set 0 to disable."),

    ("background_frames", "--background-frames", "int", "200", "Background frames", "Number of sampled frames used to build the static background. More = stabler but slower start."),
    ("background_stride", "--background-stride", "int", "10", "Background stride", "Frame step between samples for background building. Higher samples a longer time range."),
    ("background_percentile", "--background-percentile", "float", "50.0", "Background percentile", "50 = median background. Lower/higher can help with unusual thermal backgrounds."),

    ("trail_length", "--trail-length", "int", "0", "Trail length, 0 = full track", "Length of drawn trail. 0 draws full history; small values draw only a moving tail."),
]

BOOLEAN_PARAMS: List[Tuple[str, str, str, bool, str]] = [
    # key, CLI flag, label, default, explanation
    ("show", "--show", "Show OpenCV preview window", True, "Opens live preview during processing. Useful while tuning parameters."),
    ("motion_gate", "--motion-gate", "Use motion gate", False, "Requires both brightness above background and frame-to-frame movement. Can remove static artefacts but may lose faint bats."),
    ("no_prediction", "--no-prediction", "Disable prediction", False, "Turns off simple velocity prediction in tracking. Usually keep unchecked."),
    ("draw_all_tracks", "--draw-all-tracks", "Draw all tracks, including invalid", False, "Diagnostic mode. Shows tracks before artefact filtering."),
    ("hide_inactive_tracks", "--hide-inactive-tracks", "Hide inactive tracks", False, "Draw only currently active tracks. Usually leave unchecked when reviewing full trajectories."),
    ("hide_roi_rectangle", "--hide-roi-rectangle", "Hide ROI rectangle", False, "Hides the ROI rectangle overlay if ROI is used."),
    ("hide_exclude_zones", "--hide-exclude-zones", "Hide exclude-zone rectangles", False, "Hides excluded-area rectangles on the preview/output video."),
]


class RoiExcludeDrawingWindow(tk.Toplevel):
    """
    Small drawing helper for rectangular ROI and exclude zones.

    The detector uses x,y,w,h rectangles. This window lets the user draw them
    with the mouse on a video frame and writes the pixel coordinates back to
    the main GUI fields.
    """

    MAX_DISPLAY_WIDTH = 1050
    MAX_DISPLAY_HEIGHT = 680

    def __init__(
        self,
        parent: tk.Tk,
        input_video: str,
        frame_index_var: tk.StringVar,
        roi_var: tk.StringVar,
        exclude_zones_var: tk.StringVar,
    ) -> None:
        super().__init__(parent)
        self.title("Draw ROI / exclude zones")
        self.geometry("1180x820")
        self.minsize(900, 650)

        self.input_video = input_video
        self.frame_index_var = frame_index_var
        self.roi_var = roi_var
        self.exclude_zones_var = exclude_zones_var

        self.mode_var = tk.StringVar(value="exclude")
        self.status_var = tk.StringVar(value="Draw rectangles with left mouse button.")
        self.frame_info_var = tk.StringVar(value="")

        self.frame_bgr = None
        self.frame_width = 0
        self.frame_height = 0
        self.scale = 1.0
        self.display_width = 0
        self.display_height = 0
        self.photo = None

        self.drag_start = None
        self.temp_rect_id = None

        self._build_widgets()
        self._load_frame()

    def _build_widgets(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Mode:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(top, text="ROI", variable=self.mode_var, value="roi").pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Exclude zone", variable=self.mode_var, value="exclude").pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(top, text="Frame:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.frame_index_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Reload frame", command=self._load_frame).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(top, text="Clear ROI", command=self._clear_roi).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Clear exclude zones", command=self._clear_exclude_zones).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Undo last exclude", command=self._undo_last_exclude_zone).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=3)

        info = ttk.Frame(self, padding=(8, 0, 8, 4))
        info.pack(fill=tk.X)
        ttk.Label(info, textvariable=self.frame_info_var).pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.status_var).pack(side=tk.RIGHT)

        canvas_frame = ttk.Frame(self, padding=8)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, background="#222222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        bottom = ttk.LabelFrame(self, text="Current values", padding=8)
        bottom.pack(fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(bottom, text="ROI").grid(row=0, column=0, sticky="nw", padx=4, pady=3)
        ttk.Entry(bottom, textvariable=self.roi_var).grid(row=0, column=1, sticky="ew", padx=4, pady=3)

        ttk.Label(bottom, text="Exclude zones").grid(row=1, column=0, sticky="nw", padx=4, pady=3)
        self.exclude_preview = tk.Text(bottom, height=4, width=60)
        self.exclude_preview.grid(row=1, column=1, sticky="ew", padx=4, pady=3)

        bottom.columnconfigure(1, weight=1)

        self.exclude_zones_var.trace_add("write", lambda *_: self._sync_exclude_preview())
        self._sync_exclude_preview()

    def _load_frame(self) -> None:
        input_path = Path(self.input_video)
        if not input_path.exists():
            messagebox.showerror("Draw ROI / exclude zones", f"Input video not found:\n{input_path}")
            self.destroy()
            return

        try:
            frame_index = int(self.frame_index_var.get().strip() or "0")
        except ValueError:
            messagebox.showerror("Draw ROI / exclude zones", "Frame index must be an integer.")
            return

        frame_index = max(0, frame_index)
        self.frame_index_var.set(str(frame_index))

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            messagebox.showerror("Draw ROI / exclude zones", f"Could not open video:\n{input_path}")
            self.destroy()
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            frame_index = min(frame_index, max(0, frame_count - 1))
            self.frame_index_var.set(str(frame_index))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            messagebox.showerror("Draw ROI / exclude zones", f"Could not read frame {frame_index}.")
            return

        self.frame_bgr = frame
        self.frame_height, self.frame_width = frame.shape[:2]

        self.scale = min(
            self.MAX_DISPLAY_WIDTH / max(1, self.frame_width),
            self.MAX_DISPLAY_HEIGHT / max(1, self.frame_height),
            1.0,
        )
        self.display_width = max(1, int(round(self.frame_width * self.scale)))
        self.display_height = max(1, int(round(self.frame_height * self.scale)))

        self.frame_info_var.set(
            f"Frame {frame_index} | original: {self.frame_width} x {self.frame_height} px | display scale: {self.scale:.3f}"
        )
        self._redraw_canvas()

    def _redraw_canvas(self) -> None:
        if self.frame_bgr is None:
            return

        display_bgr = cv2.resize(
            self.frame_bgr,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR,
        )
        display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

        ok, png_buffer = cv2.imencode(".png", display_rgb)
        if not ok:
            messagebox.showerror("Draw ROI / exclude zones", "Could not render preview frame.")
            return

        data = base64.b64encode(png_buffer).decode("ascii")
        self.photo = tk.PhotoImage(data=data)

        self.canvas.delete("all")
        self.canvas.configure(width=self.display_width, height=self.display_height, scrollregion=(0, 0, self.display_width, self.display_height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self._draw_existing_rectangles()

    def _draw_existing_rectangles(self) -> None:
        roi = self.roi_var.get().strip()
        if roi:
            rect = self._parse_rect_safely(roi)
            if rect:
                self._draw_rect_on_canvas(rect, outline="#00ff00", width=2, text="ROI")

        for idx, zone in enumerate(self._get_exclude_zones(), start=1):
            rect = self._parse_rect_safely(zone)
            if rect:
                self._draw_rect_on_canvas(rect, outline="#ff3333", width=2, text=f"EX {idx}")

    def _draw_rect_on_canvas(self, rect, outline: str, width: int, text: str) -> None:
        x, y, w, h = rect
        x1 = x * self.scale
        y1 = y * self.scale
        x2 = (x + w) * self.scale
        y2 = (y + h) * self.scale

        self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, width=width)
        self.canvas.create_text(x1 + 4, y1 + 4, anchor=tk.NW, fill=outline, text=text)

    def _on_mouse_down(self, event) -> None:
        if self.frame_bgr is None:
            return

        x = self._clamp(event.x, 0, self.display_width - 1)
        y = self._clamp(event.y, 0, self.display_height - 1)
        self.drag_start = (x, y)

        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

        color = "#00ff00" if self.mode_var.get() == "roi" else "#ff3333"
        self.temp_rect_id = self.canvas.create_rectangle(x, y, x, y, outline=color, width=2, dash=(4, 2))

    def _on_mouse_drag(self, event) -> None:
        if self.drag_start is None or self.temp_rect_id is None:
            return

        x0, y0 = self.drag_start
        x1 = self._clamp(event.x, 0, self.display_width - 1)
        y1 = self._clamp(event.y, 0, self.display_height - 1)
        self.canvas.coords(self.temp_rect_id, x0, y0, x1, y1)

        rect = self._display_coords_to_original_rect(x0, y0, x1, y1)
        self.status_var.set(f"Drawing: {rect[0]},{rect[1]},{rect[2]},{rect[3]}")

    def _on_mouse_up(self, event) -> None:
        if self.drag_start is None:
            return

        x0, y0 = self.drag_start
        x1 = self._clamp(event.x, 0, self.display_width - 1)
        y1 = self._clamp(event.y, 0, self.display_height - 1)
        self.drag_start = None

        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

        rect = self._display_coords_to_original_rect(x0, y0, x1, y1)
        x, y, w, h = rect

        if w < 2 or h < 2:
            self.status_var.set("Rectangle too small. Try again.")
            self._redraw_canvas()
            return

        rect_text = f"{x},{y},{w},{h}"

        if self.mode_var.get() == "roi":
            self.roi_var.set(rect_text)
            self.status_var.set(f"ROI set: {rect_text}")
        else:
            zones = self._get_exclude_zones()
            zones.append(rect_text)
            self.exclude_zones_var.set("\n".join(zones))
            self.status_var.set(f"Exclude zone added: {rect_text}")

        self._redraw_canvas()

    def _display_coords_to_original_rect(self, x0: int, y0: int, x1: int, y1: int):
        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1)
        bottom = max(y0, y1)

        ox1 = int(round(left / self.scale))
        oy1 = int(round(top / self.scale))
        ox2 = int(round(right / self.scale))
        oy2 = int(round(bottom / self.scale))

        ox1 = self._clamp(ox1, 0, self.frame_width - 1)
        oy1 = self._clamp(oy1, 0, self.frame_height - 1)
        ox2 = self._clamp(ox2, 0, self.frame_width)
        oy2 = self._clamp(oy2, 0, self.frame_height)

        return ox1, oy1, max(1, ox2 - ox1), max(1, oy2 - oy1)

    def _parse_rect_safely(self, value: str):
        try:
            parts = [int(p.strip()) for p in value.split(",")]
            if len(parts) != 4:
                return None
            return parts[0], parts[1], parts[2], parts[3]
        except Exception:
            return None

    def _get_exclude_zones(self) -> List[str]:
        raw = self.exclude_zones_var.get().strip()
        if not raw:
            return []
        raw = raw.replace(";", "\n")
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def _sync_exclude_preview(self) -> None:
        if not hasattr(self, "exclude_preview"):
            return
        current = self.exclude_preview.get("1.0", tk.END).strip()
        wanted = self.exclude_zones_var.get().strip()
        if current == wanted:
            return
        self.exclude_preview.delete("1.0", tk.END)
        self.exclude_preview.insert("1.0", wanted)

    def _clear_roi(self) -> None:
        self.roi_var.set("")
        self.status_var.set("ROI cleared.")
        self._redraw_canvas()

    def _clear_exclude_zones(self) -> None:
        self.exclude_zones_var.set("")
        self.status_var.set("Exclude zones cleared.")
        self._redraw_canvas()

    def _undo_last_exclude_zone(self) -> None:
        zones = self._get_exclude_zones()
        if not zones:
            self.status_var.set("No exclude zones to remove.")
            return
        removed = zones.pop()
        self.exclude_zones_var.set("\n".join(zones))
        self.status_var.set(f"Removed exclude zone: {removed}")
        self._redraw_canvas()

    @staticmethod
    def _clamp(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))


class ThermalDetectorGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1360x860")
        self.minsize(1220, 760)

        self.process: Optional[subprocess.Popen[str]] = None
        self.reader_thread: Optional[threading.Thread] = None

        self.path_vars: Dict[str, tk.StringVar] = {}
        self.num_vars: Dict[str, tk.StringVar] = {}
        self.bool_vars: Dict[str, tk.BooleanVar] = {}

        self._create_variables()
        self._create_widgets()
        self._refresh_command_preview()

    def _create_variables(self) -> None:
        self.path_vars["script"] = tk.StringVar(value=DETECTOR_MODULE)
        self.path_vars["input"] = tk.StringVar(value="")
        self.path_vars["output"] = tk.StringVar(value="thermal_blob_valid_tracks.mp4")
        self.path_vars["csv"] = tk.StringVar(value="thermal_blob_track_points.csv")
        self.path_vars["summary_csv"] = tk.StringVar(value="thermal_blob_track_summary.csv")
        self.path_vars["roi"] = tk.StringVar(value="")
        self.path_vars["exclude_zones"] = tk.StringVar(value="")
        self.path_vars["draw_frame"] = tk.StringVar(value="0")

        for key, _flag, _typ, default, _label, _explanation in NUMERIC_PARAMS:
            self.num_vars[key] = tk.StringVar(value=default)

        for key, _flag, _label, default, _explanation in BOOLEAN_PARAMS:
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
        meta = {key: (flag, typ, default, label, explanation) for key, flag, typ, default, label, explanation in NUMERIC_PARAMS}

        ttk.Label(parent, text="Parameter").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(parent, text="Value").grid(row=0, column=1, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(parent, text="Type").grid(row=0, column=2, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(parent, text="Explanation").grid(row=0, column=3, sticky="w", padx=4, pady=(0, 6))

        for row, key in enumerate(keys, start=1):
            _flag, typ, _default, label, explanation = meta[key]
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", padx=4, pady=4)
            entry = ttk.Entry(parent, textvariable=self.num_vars[key], width=14)
            entry.grid(row=row, column=1, sticky="nw", padx=4, pady=4)
            ttk.Label(parent, text=typ).grid(row=row, column=2, sticky="nw", padx=4, pady=4)

            explanation_label = ttk.Label(
                parent,
                text=explanation,
                wraplength=360,
                justify=tk.LEFT,
            )
            explanation_label.grid(row=row, column=3, sticky="nw", padx=8, pady=4)

        parent.columnconfigure(3, weight=1)

    def _build_mask_tab(self, parent: ttk.Frame) -> None:
        drawing_frame = ttk.LabelFrame(parent, text="Draw rectangles from video frame", padding=8)
        drawing_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            drawing_frame,
            text="Use this to draw ROI and exclude zones with the mouse instead of typing pixel coordinates.",
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 6))

        ttk.Label(drawing_frame, text="Frame index").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(drawing_frame, textvariable=self.path_vars["draw_frame"], width=10).grid(row=1, column=1, sticky="w", padx=4, pady=3)
        ttk.Button(drawing_frame, text="Draw ROI / exclude zones", command=self._open_roi_exclude_drawing_window).grid(row=1, column=2, sticky="w", padx=8, pady=3)

        ttk.Label(
            drawing_frame,
            text="Tip: use frame 0 first. If it is too dark/empty, choose another frame where the background is clear.",
            justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(4, 0))

        drawing_frame.columnconfigure(3, weight=1)

        ttk.Label(parent, text="ROI, optional, format: x,y,w,h").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.path_vars["roi"]).pack(fill=tk.X, pady=(2, 10))

        text = (
            "Exclude zones, optional.\n"
            "Use one rectangle per line or separate with semicolon.\n"
            "Format: x,y,w,h\n\n"
            "You can still type/edit values manually here."
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

    def _open_roi_exclude_drawing_window(self) -> None:
        input_video = self.path_vars["input"].get().strip()
        if not input_video:
            messagebox.showerror(APP_TITLE, "Choose an input video first.")
            return

        RoiExcludeDrawingWindow(
            parent=self,
            input_video=input_video,
            frame_index_var=self.path_vars["draw_frame"],
            roi_var=self.path_vars["roi"],
            exclude_zones_var=self.path_vars["exclude_zones"],
        )

    def _build_flags_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Option").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(parent, text="Explanation").grid(row=0, column=1, sticky="w", padx=8, pady=(0, 6))

        for row, (key, _flag, label, _default, explanation) in enumerate(BOOLEAN_PARAMS, start=1):
            cb = ttk.Checkbutton(parent, text=label, variable=self.bool_vars[key])
            cb.grid(row=row, column=0, sticky="nw", padx=4, pady=4)

            explanation_label = ttk.Label(
                parent,
                text=explanation,
                wraplength=430,
                justify=tk.LEFT,
            )
            explanation_label.grid(row=row, column=1, sticky="nw", padx=8, pady=4)

        parent.columnconfigure(1, weight=1)

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
        script_or_module = self.path_vars["script"].get().strip()
        input_video = self.path_vars["input"].get().strip()

        if not script_or_module:
            raise ValueError("Detector script/module is required.")
        if not input_video:
            raise ValueError("Input video is required.")
        if not Path(input_video).exists():
            raise ValueError(f"Input video not found:\n{input_video}")

        script_path = Path(script_or_module)
        if script_path.exists():
            cmd: List[str] = [sys.executable, "-u", str(script_path), "--input", input_video]
        elif script_or_module == DETECTOR_MODULE or "." in script_or_module:
            cmd = [sys.executable, "-u", "-m", script_or_module, "--input", input_video]
        else:
            raise ValueError(f"Detector script not found:\n{script_or_module}")

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

        meta = {key: (flag, typ) for key, flag, typ, _default, _label, _explanation in NUMERIC_PARAMS}
        for key, var in self.num_vars.items():
            value = var.get().strip()
            if value == "":
                continue
            flag, _typ = meta[key]
            cmd += [flag, value]

        for key, flag, _label, _default, _explanation in BOOLEAN_PARAMS:
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
        meta = {key: (typ, label) for key, _flag, typ, _default, label, _explanation in NUMERIC_PARAMS}

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

        script_or_module = self.path_vars["script"].get().strip()
        script_path = Path(script_or_module)
        cwd = str(script_path.parent) if script_path.exists() else str(Path.cwd())
        env = os.environ.copy()
        source_dir = str(Path(__file__).resolve().parent)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            paths = existing_pythonpath.split(os.pathsep)
            if source_dir not in paths:
                env["PYTHONPATH"] = os.pathsep.join([source_dir, existing_pythonpath])
        else:
            env["PYTHONPATH"] = source_dir

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
                env=env,
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
