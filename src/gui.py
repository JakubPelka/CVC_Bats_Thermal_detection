#!/usr/bin/env python3
"""
gui.py

Tkinter launcher for thermal_blob_detector.

Purpose:
- Change detector parameters from a small GUI.
- Run the detector script without editing Python code.
- Save/load parameter presets as JSON.
- Keep the detector itself separate and stable.

Run after editable install:
    thermal-blob-detector-gui

Or directly from the repository:
    PYTHONPATH=src python -m gui

UI note:
The Run/Stop buttons are placed in a persistent bottom action bar so they remain
visible on smaller screens and when desktop scaling is enabled. The main area uses
a draggable split: parameter explanations get more width by default and the log
panel can be resized with the vertical divider.
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
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import cv2


APP_TITLE = "Thermal Bat Blob Detector - Parameter GUI"
DETECTOR_MODULE = "thermal_blob_detector"


def _absolute_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _default_counting_config_path() -> str:
    return _absolute_path("outputs/counting_config_drawn.json")


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
    ("background_recalibrate_interval", "--background-recalibrate-interval", "int", "0", "Recalibrate interval, 0 = off", "Rebuild the thermal background every N processed frames. Leave at 0 to preserve the static-background behaviour."),
    ("background_recalibrate_frames", "--background-recalibrate-frames", "int", "200", "Recalibration frames", "Number of future sampled frames used for each periodic background update."),
    ("background_recalibrate_stride", "--background-recalibrate-stride", "int", "10", "Recalibration stride", "Frame step between samples in each periodic background window."),
    ("background_recalibrate_blend", "--background-recalibrate-blend", "float", "1.0", "Recalibration blend", "Update strength: 1.0 replaces the background; smaller values such as 0.2 adapt gently to slow drift."),

    ("trail_length", "--trail-length", "int", "0", "Trail length, 0 = full track", "Length of drawn trail. 0 draws full history; small values draw only a moving tail."),

    ("activity_bin_seconds", "--activity-bin-seconds", "float", "60.0", "Activity bin seconds", "Length of each time bucket in activity_by_time.csv. Smaller values give more detailed time series; larger values smooth the summary."),
    ("line_crossing_epsilon", "--line-crossing-epsilon", "float", "1.0", "Line crossing epsilon", "Pixel tolerance around a counting line. Points inside this band are treated as being on the line, which reduces false double counts from jitter near the boundary."),
    ("min_frames_between_same_line_crossing", "--min-frames-between-same-line-crossing", "int", "3", "Line crossing debounce", "Minimum frame gap before the same track can trigger the same line again. Higher values suppress rapid back-and-forth jitter near a line."),
    ("aoi_boundary_debounce_frames", "--aoi-boundary-debounce-frames", "int", "3", "AOI boundary debounce", "Minimum frame gap before entry/exit state changes are accepted near an AOI edge. Higher values reduce flicker when a centroid sits on the border."),
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
    ("count_all_tracks", "--count-all-tracks", "Count all tracks for diagnostics", False, "Counts invalid/static/noisy tracks too. Leave unchecked for final valid-track counting."),
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
        self._set_safe_window_size()

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

    def _set_safe_window_size(self) -> None:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(1180, max(860, screen_w - 80))
        height = min(820, max(600, screen_h - 120))
        self.geometry(f"{width}x{height}")
        self.minsize(760, 520)

    def _build_widgets(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

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
        info.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(info, textvariable=self.frame_info_var).pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.status_var).pack(side=tk.RIGHT)

        bottom = ttk.LabelFrame(self, text="Current values", padding=8)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(bottom, text="ROI").grid(row=0, column=0, sticky="nw", padx=4, pady=3)
        ttk.Entry(bottom, textvariable=self.roi_var).grid(row=0, column=1, sticky="ew", padx=4, pady=3)

        ttk.Label(bottom, text="Exclude zones").grid(row=1, column=0, sticky="nw", padx=4, pady=3)
        self.exclude_preview = tk.Text(bottom, height=4, width=60)
        self.exclude_preview.grid(row=1, column=1, sticky="ew", padx=4, pady=3)

        bottom.columnconfigure(1, weight=1)

        canvas_frame = ttk.Frame(self, padding=8)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, background="#222222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

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


class CountingConfigDrawingWindow(tk.Toplevel):
    """Click-based editor for counting lines and polygon AOIs."""

    MAX_DISPLAY_WIDTH = 1050
    MAX_DISPLAY_HEIGHT = 680

    def __init__(
        self,
        parent: tk.Tk,
        input_video: str,
        frame_index_var: tk.StringVar,
        counting_config_var: tk.StringVar,
        count_lines_var: tk.StringVar,
        count_aois_var: tk.StringVar,
    ) -> None:
        super().__init__(parent)
        self.title("Draw counting lines / AOIs")
        self.geometry(f"{min(1180, max(860, self.winfo_screenwidth() - 80))}x{min(820, max(600, self.winfo_screenheight() - 120))}")
        self.minsize(760, 520)

        self.input_video = input_video
        self.frame_index_var = frame_index_var
        self.counting_config_var = counting_config_var
        self.count_lines_var = count_lines_var
        self.count_aois_var = count_aois_var

        self.mode_var = tk.StringVar(value="line")
        self.status_var = tk.StringVar(value="Line: click A then B. Polyline/AOI: click points, Enter to finish.")
        self.frame_info_var = tk.StringVar(value="")
        self.lines: List[dict] = []
        self.aois: List[dict] = []
        self.current_points: List[Tuple[float, float]] = []

        self.frame_bgr = None
        self.frame_width = 0
        self.frame_height = 0
        self.scale = 1.0
        self.display_width = 0
        self.display_height = 0
        self.photo = None

        self._build_widgets()
        self._load_existing_config()
        self._load_frame()

    def _build_widgets(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Tool:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(top, text="Line", variable=self.mode_var, value="line", command=self._reset_points).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Polyline", variable=self.mode_var, value="polyline", command=self._reset_points).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="AOI polygon", variable=self.mode_var, value="polygon", command=self._reset_points).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(top, text="Frame:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.frame_index_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Reload frame", command=self._load_frame).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(top, text="Undo point", command=self._undo_point).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Remove last object", command=self._remove_last).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Clear all", command=self._clear_all).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Save JSON", command=self._save_json).pack(side=tk.RIGHT, padx=3)
        ttk.Button(top, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=3)

        info = ttk.Frame(self, padding=(8, 0, 8, 4))
        info.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(info, textvariable=self.frame_info_var).pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.status_var).pack(side=tk.RIGHT)

        canvas_frame = ttk.Frame(self, padding=8)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="#222222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.bind("<Return>", lambda _event: self._finish_multi_point())
        self.bind("<BackSpace>", lambda _event: self._undo_point())

        bottom = ttk.LabelFrame(self, text="Current counting objects", padding=8)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))
        self.summary_text = tk.Text(bottom, height=4, width=80)
        self.summary_text.pack(fill=tk.X)
        self._sync_summary()

    def _load_frame(self) -> None:
        input_path = Path(self.input_video)
        if not input_path.exists():
            messagebox.showerror("Draw counting config", f"Input video not found:\n{input_path}")
            self.destroy()
            return
        try:
            frame_index = max(0, int(self.frame_index_var.get().strip() or "0"))
        except ValueError:
            messagebox.showerror("Draw counting config", "Frame index must be an integer.")
            return
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            messagebox.showerror("Draw counting config", f"Could not open video:\n{input_path}")
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
            messagebox.showerror("Draw counting config", f"Could not read frame {frame_index}.")
            return
        self.frame_bgr = frame
        self.frame_height, self.frame_width = frame.shape[:2]
        self.scale = min(self.MAX_DISPLAY_WIDTH / max(1, self.frame_width), self.MAX_DISPLAY_HEIGHT / max(1, self.frame_height), 1.0)
        self.display_width = max(1, int(round(self.frame_width * self.scale)))
        self.display_height = max(1, int(round(self.frame_height * self.scale)))
        self.frame_info_var.set(f"Frame {frame_index} | original: {self.frame_width} x {self.frame_height} px | display scale: {self.scale:.3f}")
        self._redraw_canvas()

    def _redraw_canvas(self) -> None:
        if self.frame_bgr is None:
            return
        display_bgr = cv2.resize(self.frame_bgr, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR)
        display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        ok, png_buffer = cv2.imencode(".png", display_rgb)
        if not ok:
            messagebox.showerror("Draw counting config", "Could not render preview frame.")
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(png_buffer).decode("ascii"))
        self.canvas.delete("all")
        self.canvas.configure(width=self.display_width, height=self.display_height, scrollregion=(0, 0, self.display_width, self.display_height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self._draw_objects()

    def _draw_objects(self) -> None:
        for line in self.lines:
            pts = [self._img_to_display(point) for point in line["pts"]]
            for idx in range(1, len(pts)):
                self.canvas.create_line(*pts[idx - 1], *pts[idx], fill="#00d7ff", width=3, arrow=tk.LAST if idx == len(pts) - 1 else None)
            self.canvas.create_text(*pts[0], anchor=tk.SW, fill="#00d7ff", text="A")
            self.canvas.create_text(*pts[-1], anchor=tk.NW, fill="#00d7ff", text=f"B {line['name']}")
        for aoi in self.aois:
            pts = [self._img_to_display(point) for point in aoi["coordinates"]]
            flat = [coord for point in pts for coord in point]
            self.canvas.create_polygon(*flat, outline="#ffaa00", fill="", width=2)
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.canvas.create_text(cx, cy, fill="#ffaa00", text=aoi["name"])
        if self.current_points:
            pts = [self._img_to_display(point) for point in self.current_points]
            for idx, point in enumerate(pts):
                self.canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill="yellow", outline="")
                if idx > 0:
                    self.canvas.create_line(*pts[idx - 1], *point, fill="yellow", width=2)

    def _on_click(self, event) -> None:
        if self.frame_bgr is None:
            return
        x = max(0, min(self.display_width - 1, event.x)) / self.scale
        y = max(0, min(self.display_height - 1, event.y)) / self.scale
        self.current_points.append((x, y))
        mode = self.mode_var.get()
        if mode == "line" and len(self.current_points) == 2:
            self._add_line(self.current_points)
            self.current_points = []
        self._redraw_canvas()

    def _finish_multi_point(self) -> None:
        mode = self.mode_var.get()
        if mode == "polyline" and len(self.current_points) >= 2:
            self._add_line(self.current_points)
            self.current_points = []
        elif mode == "polygon" and len(self.current_points) >= 3:
            self._add_polygon(self.current_points)
            self.current_points = []
        else:
            self.status_var.set("Need 2 points for a line/polyline or 3 points for AOI polygon.")
            return
        self._redraw_canvas()

    def _add_line(self, points: List[Tuple[float, float]]) -> None:
        name = simpledialog.askstring("Line name", "Name for this counting line:", parent=self)
        if not name:
            return
        line_id = self._slug(name, f"line_{len(self.lines) + 1}")
        self.lines.append({
            "id": line_id,
            "name": name.strip(),
            "p1": [round(points[0][0], 3), round(points[0][1], 3)],
            "p2": [round(points[-1][0], 3), round(points[-1][1], 3)],
            "pts": [[round(x, 3), round(y, 3)] for x, y in points],
            "direction_labels": {"positive": "right_to_left", "negative": "left_to_right"},
            "enabled": True,
        })
        self.status_var.set(f"Line added: {name}")
        self._sync_summary()

    def _add_polygon(self, points: List[Tuple[float, float]]) -> None:
        name = simpledialog.askstring("AOI name", "Name for this AOI / ROI:", parent=self)
        if not name:
            return
        aoi_id = self._slug(name, f"aoi_{len(self.aois) + 1}")
        self.aois.append({
            "id": aoi_id,
            "name": name.strip(),
            "type": "polygon",
            "coordinates": [[round(x, 3), round(y, 3)] for x, y in points],
            "enabled": True,
        })
        self.status_var.set(f"AOI added: {name}")
        self._sync_summary()

    def _save_json(self) -> None:
        target = self.counting_config_var.get().strip()
        if not target:
            target = _default_counting_config_path()
        path = Path(target).expanduser().resolve()
        self.counting_config_var.set(str(path))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"lines": self.lines, "aois": self.aois}
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception as exc:
            messagebox.showerror("Save counting config", str(exc))
            return
        self.count_lines_var.set("")
        self.count_aois_var.set("")
        self.status_var.set(f"Saved: {path}")
        messagebox.showinfo("Save counting config", f"Saved counting config:\n{path}")

    def _load_existing_config(self) -> None:
        target = self.counting_config_var.get().strip()
        if not target or not Path(target).exists():
            return
        try:
            data = json.loads(Path(target).read_text(encoding="utf-8"))
        except Exception:
            return
        self.lines = []
        for idx, item in enumerate(data.get("lines", []), start=1):
            pts = item.get("pts") or [item.get("p1"), item.get("p2")]
            if pts and len(pts) >= 2:
                self.lines.append({**item, "id": item.get("id", f"line_{idx}"), "name": item.get("name", f"Line {idx}"), "pts": pts})
        self.aois = []
        for idx, item in enumerate(data.get("aois", data.get("zones", [])), start=1):
            coords = item.get("coordinates", item.get("pts", []))
            if item.get("type", "polygon") == "polygon" and len(coords) >= 3:
                self.aois.append({**item, "id": item.get("id", f"aoi_{idx}"), "name": item.get("name", f"AOI {idx}"), "type": "polygon", "coordinates": coords})
        self._sync_summary()

    def _sync_summary(self) -> None:
        if not hasattr(self, "summary_text"):
            return
        rows = [f"Lines: {len(self.lines)}", f"AOIs: {len(self.aois)}"]
        rows.extend(f"LINE {line['name']}: {len(line['pts'])} pts" for line in self.lines)
        rows.extend(f"AOI {aoi['name']}: {len(aoi['coordinates'])} pts" for aoi in self.aois)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(rows))

    def _reset_points(self) -> None:
        self.current_points = []
        mode = self.mode_var.get()
        if mode == "line":
            self.status_var.set("Line: click A then B.")
        elif mode == "polyline":
            self.status_var.set("Polyline: click points, press Enter to finish.")
        else:
            self.status_var.set("AOI polygon: click boundary points, press Enter to finish.")
        self._redraw_canvas()

    def _undo_point(self) -> None:
        if self.current_points:
            self.current_points.pop()
            self._redraw_canvas()

    def _remove_last(self) -> None:
        if self.aois:
            self.aois.pop()
        elif self.lines:
            self.lines.pop()
        self._sync_summary()
        self._redraw_canvas()

    def _clear_all(self) -> None:
        self.lines.clear()
        self.aois.clear()
        self.current_points.clear()
        self._sync_summary()
        self._redraw_canvas()

    def _img_to_display(self, point) -> Tuple[float, float]:
        return float(point[0]) * self.scale, float(point[1]) * self.scale

    @staticmethod
    def _slug(value: str, fallback: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip()).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug or fallback


class ScrollableFrame(ttk.Frame):
    """A reusable vertically scrollable frame for dense parameter tabs."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if not self.winfo_ismapped():
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event) -> None:
        if not self.winfo_ismapped():
            return
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")


class ThermalDetectorGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self._set_initial_window_size()

        self.process: Optional[subprocess.Popen[str]] = None
        self.reader_thread: Optional[threading.Thread] = None

        self.path_vars: Dict[str, tk.StringVar] = {}
        self.num_vars: Dict[str, tk.StringVar] = {}
        self.bool_vars: Dict[str, tk.BooleanVar] = {}

        self._create_variables()
        self._create_widgets()
        self._refresh_command_preview()

    def _set_initial_window_size(self) -> None:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(1360, max(980, screen_w - 80))
        height = min(860, max(620, screen_h - 120))
        self.geometry(f"{width}x{height}")
        self.minsize(900, 560)

    def _create_variables(self) -> None:
        self.path_vars["script"] = tk.StringVar(value=DETECTOR_MODULE)
        self.path_vars["input"] = tk.StringVar(value="")
        self.path_vars["output"] = tk.StringVar(value="outputs/thermal_blob_valid_tracks.mp4")
        self.path_vars["csv"] = tk.StringVar(value="outputs/thermal_blob_track_points.csv")
        self.path_vars["summary_csv"] = tk.StringVar(value="outputs/thermal_blob_track_summary.csv")
        self.path_vars["crossings_csv"] = tk.StringVar(value="outputs/crossings.csv")
        self.path_vars["aoi_events_csv"] = tk.StringVar(value="outputs/aoi_events.csv")
        self.path_vars["activity_csv"] = tk.StringVar(value="outputs/activity_by_time.csv")
        self.path_vars["run_summary_json"] = tk.StringVar(value="outputs/run_summary.json")
        self.path_vars["counting_config"] = tk.StringVar(value="")
        self.path_vars["count_lines"] = tk.StringVar(value="")
        self.path_vars["count_aois"] = tk.StringVar(value="")
        self.path_vars["roi"] = tk.StringVar(value="")
        self.path_vars["exclude_zones"] = tk.StringVar(value="")
        self.path_vars["draw_frame"] = tk.StringVar(value="0")

        for key, _flag, _typ, default, _label, _explanation in NUMERIC_PARAMS:
            self.num_vars[key] = tk.StringVar(value=default)

        for key, _flag, _label, default, _explanation in BOOLEAN_PARAMS:
            self.bool_vars[key] = tk.BooleanVar(value=default)
        self.bool_vars["save_annotated_video"] = tk.BooleanVar(value=True)

        for var in list(self.path_vars.values()) + list(self.num_vars.values()):
            var.trace_add("write", lambda *_: self._refresh_command_preview())
        for var in self.bool_vars.values():
            var.trace_add("write", lambda *_: self._refresh_command_preview())

    def _create_widgets(self) -> None:
        # Persistent action bar: this is intentionally packed first at the bottom.
        # It keeps Run/Stop accessible on small screens and with desktop scaling.
        action_bar = ttk.Frame(self, padding=(8, 6, 8, 8))
        action_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_buttons(action_bar)

        root = ttk.Frame(self, padding=(8, 8, 8, 0))
        root.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_file_section(root)

        # Resizable horizontal split. Parameters get more space by default,
        # while the log panel remains useful but no longer dominates the UI.
        middle = tk.PanedWindow(
            root,
            orient=tk.HORIZONTAL,
            sashwidth=8,
            sashrelief=tk.RAISED,
            bd=0,
        )
        middle.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

        left = ttk.Frame(middle)
        right = ttk.Frame(middle)

        screen_w = self.winfo_screenwidth()
        total_width = max(1040, min(1600, screen_w - 80))
        log_width = min(int(total_width * 0.30), max(300, int(screen_w * 0.22)))
        parameter_width = total_width - log_width

        middle.add(left, minsize=760, width=parameter_width, stretch="always")
        middle.add(right, minsize=280, width=log_width, stretch="never")

        self._build_parameter_section(left)
        self._build_command_and_log_section(right)

    def _build_file_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Files", padding=6)
        frame.pack(fill=tk.X)

        self._path_row(frame, "Detector script/module", "script", self._browse_script, row=0)
        self._path_row(frame, "Input video", "input", self._browse_input, row=1)
        run_options = ttk.Frame(frame)
        run_options.grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(4, 2))
        ttk.Checkbutton(run_options, text="Preview window", variable=self.bool_vars["show"]).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(run_options, text="Save annotated video", variable=self.bool_vars["save_annotated_video"]).pack(side=tk.LEFT)
        self._path_row(frame, "Output video", "output", self._browse_output, row=3)
        self._path_row(frame, "Track points CSV", "csv", self._browse_csv, row=4)
        self._path_row(frame, "Track summary CSV", "summary_csv", self._browse_summary_csv, row=5)
        self._path_row(frame, "Crossings CSV", "crossings_csv", self._browse_crossings_csv, row=6)
        self._path_row(frame, "AOI events CSV", "aoi_events_csv", self._browse_aoi_events_csv, row=7)
        self._path_row(frame, "Activity CSV", "activity_csv", self._browse_activity_csv, row=8)
        self._path_row(frame, "Run summary JSON", "run_summary_json", self._browse_run_summary_json, row=9)

        frame.columnconfigure(1, weight=1)

    def _path_row(self, parent: ttk.Frame, label: str, key: str, command, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
        entry = ttk.Entry(parent, textvariable=self.path_vars[key])
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, padx=4, pady=3)

    def _build_parameter_section(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_defs = [
            ("Detection", [
                "max_frames", "threshold", "motion_threshold", "min_area", "max_area",
                "min_width", "min_height", "max_width", "max_height", "morph_open", "morph_dilate",
            ]),
            ("Tracking", ["max_link_distance", "max_gap_frames", "min_track_lifetime", "trail_length"]),
            ("Track filters", [
                "min_track_displacement", "min_track_path_length", "min_mean_speed", "max_mean_speed",
                "min_directionality", "max_detections_per_frame",
            ]),
            ("Background", [
                "background_frames", "background_stride", "background_percentile",
                "background_recalibrate_interval", "background_recalibrate_frames",
                "background_recalibrate_stride", "background_recalibrate_blend",
            ]),
        ]

        for title, keys in tab_defs:
            scroll = ScrollableFrame(notebook, padding=0)
            notebook.add(scroll, text=title)
            self._params_grid(scroll.inner, keys)

        tab_masks = ttk.Frame(notebook, padding=8)
        notebook.add(tab_masks, text="ROI / exclude")
        self._build_mask_tab(tab_masks)

        tab_counting = ttk.Frame(notebook, padding=8)
        notebook.add(tab_counting, text="Counting / Statistics")
        self._build_counting_tab(tab_counting)

        tab_flags_scroll = ScrollableFrame(notebook, padding=0)
        notebook.add(tab_flags_scroll, text="Flags")
        self._build_flags_tab(tab_flags_scroll.inner)

    def _params_grid(self, parent: ttk.Frame, keys: List[str]) -> None:
        meta = {key: (flag, typ, default, label, explanation) for key, flag, typ, default, label, explanation in NUMERIC_PARAMS}

        ttk.Label(parent, text="Parameter").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(parent, text="Value").grid(row=0, column=1, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(parent, text="Explanation").grid(row=0, column=2, sticky="w", padx=4, pady=(0, 6))

        for row, key in enumerate(keys, start=1):
            _flag, typ, _default, label, explanation = meta[key]
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", padx=4, pady=4)
            entry = ttk.Entry(parent, textvariable=self.num_vars[key], width=10)
            entry.grid(row=row, column=1, sticky="nw", padx=4, pady=4)
            explanation_label = ttk.Label(parent, text=f"{typ}. {explanation}", wraplength=760, justify=tk.LEFT)
            explanation_label.grid(row=row, column=2, sticky="nw", padx=8, pady=4)

        parent.columnconfigure(0, minsize=150)
        parent.columnconfigure(1, minsize=90)
        parent.columnconfigure(2, weight=1, minsize=650)

    def _build_mask_tab(self, parent: ttk.Frame) -> None:
        drawing_frame = ttk.LabelFrame(parent, text="Draw rectangles from video frame", padding=8)
        drawing_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(drawing_frame, text="Frame index").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(drawing_frame, textvariable=self.path_vars["draw_frame"], width=10).grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Button(drawing_frame, text="Draw ROI / exclude zones", command=self._open_roi_exclude_drawing_window).grid(row=0, column=2, sticky="w", padx=8, pady=3)

        drawing_frame.columnconfigure(3, weight=1)

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


    def _open_counting_drawing_window(self) -> None:
        input_video = self.path_vars["input"].get().strip()
        if not input_video:
            messagebox.showerror(APP_TITLE, "Choose an input video first.")
            return
        if not self.path_vars["counting_config"].get().strip():
            self.path_vars["counting_config"].set(_default_counting_config_path())
        CountingConfigDrawingWindow(
            parent=self,
            input_video=input_video,
            frame_index_var=self.path_vars["draw_frame"],
            counting_config_var=self.path_vars["counting_config"],
            count_lines_var=self.path_vars["count_lines"],
            count_aois_var=self.path_vars["count_aois"],
        )

    def _browse_counting_config_save(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save counting config JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.path_vars["counting_config"].set(path)

    def _build_counting_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Counting config JSON, optional").pack(anchor="w")
        config_row = ttk.Frame(parent)
        config_row.pack(fill=tk.X, pady=(2, 6))
        ttk.Entry(config_row, textvariable=self.path_vars["counting_config"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(config_row, text="Browse", command=self._browse_counting_config).pack(side=tk.LEFT, padx=(4, 0))

        draw_row = ttk.LabelFrame(parent, text="Draw counting geometry from video frame", padding=8)
        draw_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(draw_row, text="Frame index").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(draw_row, textvariable=self.path_vars["draw_frame"], width=10).grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Button(draw_row, text="Draw lines / AOIs", command=self._open_counting_drawing_window).grid(row=0, column=2, sticky="w", padx=8, pady=3)
        ttk.Button(draw_row, text="Save as...", command=self._browse_counting_config_save).grid(row=0, column=3, sticky="w", padx=4, pady=3)
        draw_row.columnconfigure(4, weight=1)

        ttk.Checkbutton(parent, text="Count all tracks for diagnostics", variable=self.bool_vars["count_all_tracks"]).pack(anchor="w", pady=(0, 10))

        numeric_frame = ttk.LabelFrame(parent, text="Counting thresholds", padding=6)
        numeric_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(numeric_frame, text="Parameter").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(numeric_frame, text="Value").grid(row=0, column=1, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(numeric_frame, text="Explanation").grid(row=0, column=2, sticky="w", padx=8, pady=(0, 6))
        meta = {key: (typ, explanation) for key, _flag, typ, _default, _label, explanation in NUMERIC_PARAMS}
        for row, key in enumerate(["activity_bin_seconds", "line_crossing_epsilon", "min_frames_between_same_line_crossing", "aoi_boundary_debounce_frames"]):
            row += 1
            label = next(item[4] for item in NUMERIC_PARAMS if item[0] == key)
            typ, explanation = meta[key]
            ttk.Label(numeric_frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(numeric_frame, textvariable=self.num_vars[key], width=10).grid(row=row, column=1, sticky="w", padx=4, pady=3)
            ttk.Label(numeric_frame, text=f"{typ}. {explanation}", wraplength=720, justify=tk.LEFT).grid(row=row, column=2, sticky="w", padx=8, pady=3)
        numeric_frame.columnconfigure(2, weight=1)

    def _build_flags_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Option").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))
        ttk.Label(parent, text="Explanation").grid(row=0, column=1, sticky="w", padx=8, pady=(0, 6))

        row = 1
        for key, _flag, label, _default, explanation in BOOLEAN_PARAMS:
            if key == "show":
                continue
            cb = ttk.Checkbutton(parent, text=label, variable=self.bool_vars[key])
            cb.grid(row=row, column=0, sticky="nw", padx=4, pady=4)
            explanation_label = ttk.Label(parent, text=explanation, wraplength=520, justify=tk.LEFT)
            explanation_label.grid(row=row, column=1, sticky="nw", padx=8, pady=4)
            row += 1

        parent.columnconfigure(1, weight=1, minsize=480)

    def _build_command_and_log_section(self, parent: ttk.Frame) -> None:
        command_frame = ttk.LabelFrame(parent, text="Generated command", padding=6)
        command_frame.pack(fill=tk.X)

        self.command_text = tk.Text(command_frame, height=4, wrap=tk.WORD)
        self.command_text.pack(fill=tk.X, expand=False)

        ttk.Button(command_frame, text="Copy command", command=self._copy_command).pack(anchor="e", pady=(4, 0))

        log_frame = ttk.LabelFrame(parent, text="Run log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.log_text = ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_buttons(self, parent: ttk.Frame) -> None:
        primary = ttk.Frame(parent)
        primary.pack(fill=tk.X)

        self.run_button = ttk.Button(primary, text="Run detector", command=self._run_detector)
        self.run_button.pack(side=tk.LEFT, padx=(0, 6))

        self.stop_button = ttk.Button(primary, text="Stop", command=self._stop_detector, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=6)

        ttk.Label(primary, text="Run/Stop stay visible here.").pack(side=tk.LEFT, padx=12)

        secondary = ttk.Frame(parent)
        secondary.pack(fill=tk.X, pady=(6, 0))

        ttk.Button(secondary, text="Clear log", command=self._clear_log).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(secondary, text="Save preset JSON", command=self._save_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(secondary, text="Load preset JSON", command=self._load_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(secondary, text="Open output folder", command=self._open_output_folder).pack(side=tk.LEFT, padx=6)

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

    def _browse_crossings_csv(self) -> None:
        self._browse_save_path("Select crossings CSV", "crossings_csv", ".csv", [("CSV", "*.csv"), ("All files", "*.*")])

    def _browse_aoi_events_csv(self) -> None:
        self._browse_save_path("Select AOI events CSV", "aoi_events_csv", ".csv", [("CSV", "*.csv"), ("All files", "*.*")])

    def _browse_activity_csv(self) -> None:
        self._browse_save_path("Select activity CSV", "activity_csv", ".csv", [("CSV", "*.csv"), ("All files", "*.*")])

    def _browse_run_summary_json(self) -> None:
        self._browse_save_path("Select run summary JSON", "run_summary_json", ".json", [("JSON", "*.json"), ("All files", "*.*")])

    def _browse_counting_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select counting config JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.path_vars["counting_config"].set(path)

    def _browse_save_path(self, title: str, key: str, extension: str, filetypes) -> None:
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=extension,
            filetypes=filetypes,
        )
        if path:
            self.path_vars[key].set(path)

    def _suggest_outputs_from_input(self, input_path: Path) -> None:
        stem = input_path.stem
        output_dir = Path("outputs")
        self.path_vars["output"].set(str(output_dir / f"{stem}_valid_tracks.mp4"))
        self.path_vars["csv"].set(str(output_dir / f"{stem}_track_points.csv"))
        self.path_vars["summary_csv"].set(str(output_dir / f"{stem}_track_summary.csv"))
        self.path_vars["crossings_csv"].set(str(output_dir / f"{stem}_crossings.csv"))
        self.path_vars["aoi_events_csv"].set(str(output_dir / f"{stem}_aoi_events.csv"))
        self.path_vars["activity_csv"].set(str(output_dir / f"{stem}_activity_by_time.csv"))
        self.path_vars["run_summary_json"].set(str(output_dir / f"{stem}_run_summary.json"))

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
            raise ValueError(f"Detector script/module not found:\n{script_or_module}")

        output = self.path_vars["output"].get().strip()
        csv_path = self.path_vars["csv"].get().strip()
        summary_csv = self.path_vars["summary_csv"].get().strip()
        crossings_csv = self.path_vars["crossings_csv"].get().strip()
        aoi_events_csv = self.path_vars["aoi_events_csv"].get().strip()
        activity_csv = self.path_vars["activity_csv"].get().strip()
        run_summary_json = self.path_vars["run_summary_json"].get().strip()

        if self.bool_vars["save_annotated_video"].get():
            if output:
                cmd += ["--output", output]
        else:
            cmd += ["--output", ""]
        if csv_path:
            cmd += ["--csv", csv_path]
        if summary_csv:
            cmd += ["--summary-csv", summary_csv]
        if crossings_csv:
            cmd += ["--crossings-csv", crossings_csv]
        if aoi_events_csv:
            cmd += ["--aoi-events-csv", aoi_events_csv]
        if activity_csv:
            cmd += ["--activity-csv", activity_csv]
        if run_summary_json:
            cmd += ["--run-summary-json", run_summary_json]
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

        counting_config = self.path_vars["counting_config"].get().strip()
        if counting_config:
            counting_config = _absolute_path(counting_config)
            self.path_vars["counting_config"].set(counting_config)
            cmd += ["--counting-config", counting_config]
        if self.bool_vars.get("count_all_tracks") and self.bool_vars["count_all_tracks"].get():
            cmd.append("--count-all-tracks")

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

    def _parse_multiline_values(self, key: str) -> List[str]:
        raw = self.path_vars[key].get().strip()
        if not raw:
            return []
        normalized = raw.replace(";", "\n")
        return [line.strip() for line in normalized.splitlines() if line.strip()]

    def _validate_count_line(self, value: str) -> None:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) not in (6, 8):
            raise ValueError(f"Counting line must use id,name,x1,y1,x2,y2[,positive_label,negative_label]. Got: {value}")
        try:
            [float(p) for p in parts[2:6]]
        except ValueError as exc:
            raise ValueError(f"Counting line coordinates must be numeric. Got: {value}") from exc

    def _validate_count_aoi(self, value: str) -> None:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 6:
            raise ValueError(f"Counting AOI must use id,name,x,y,w,h. Got: {value}")
        try:
            [float(p) for p in parts[2:6]]
        except ValueError as exc:
            raise ValueError(f"Counting AOI coordinates must be numeric. Got: {value}") from exc

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
