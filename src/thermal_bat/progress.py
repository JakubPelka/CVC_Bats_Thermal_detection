"""Terminal progress formatting shared by the pipeline and tests."""

import time
from typing import Optional


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def build_progress_text(frame_idx: int, total_frames: int, start_time: float,
                        current_time: Optional[float] = None) -> str:
    elapsed = max(0.0, (time.time() if current_time is None else current_time) - start_time)
    processed = max(1, frame_idx + 1)
    fps = processed / elapsed if elapsed > 0 else 0.0
    if total_frames > 0:
        percent = min(100.0, 100.0 * processed / total_frames)
        eta = max(0, total_frames - processed) / fps if fps > 0 else 0.0
        return f"Frame {processed} / {total_frames} ({percent:.1f}%) | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | {fps:.1f} fps"
    return f"Frame {processed} | elapsed {format_duration(elapsed)} | {fps:.1f} fps"
