"""Single-video and batch processing orchestration."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import cv2

from batch_processing import build_output_paths, collect_input_videos, safe_stem, write_batch_summary
from counting_stats import (
    CountingConfig, analyze_tracks, aoi_from_cli, line_from_cli, load_counting_config,
    write_activity_csv, write_aoi_events_csv, write_counting_config_json,
    write_crossings_csv, write_run_summary_json,
    write_track_summary_csv as write_counting_track_summary_csv,
)
from event_clips import build_clip_windows, export_event_clips, merge_clip_windows, write_clip_manifest
from .config import Rect, ThermalBlobConfig
from .detector import ThermalBlobDetector
from .exports import write_track_points_csv
from .live_counting import LiveCounting
from .progress import build_progress_text
from .validation import is_valid_flying_track
from .visualization import OverlayRenderer, draw_event_clip_overlay


def resolve_frame_range(frame_count: int, start_frame: int, end_frame: int,
                        max_frames: int = 0) -> tuple[int, int, int]:
    """Return inclusive source bounds and number of frames to process."""
    if start_frame < 0 or end_frame < 0 or max_frames < 0:
        raise ValueError("Frame range values must be non-negative")
    start = start_frame
    video_end = frame_count - 1 if frame_count > 0 else end_frame
    end = video_end if end_frame == 0 else end_frame
    if frame_count > 0:
        start = min(start, video_end)
        end = min(end, video_end)
    if end < start:
        raise ValueError(f"End frame ({end_frame}) must be zero or >= start frame ({start_frame})")
    count = end - start + 1
    if max_frames > 0:
        count = min(count, max_frames)
        end = start + count - 1
    return start, end, count


def parse_rect(value: Optional[str], argument_name: str = "rectangle") -> Optional[Rect]:
    if not value:
        return None
    parts = [int(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"{argument_name} must use format x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def parse_rect_list(values: Optional[List[str]], argument_name: str = "rectangle") -> List[Rect]:
    if not values:
        return []

    rects: List[Rect] = []
    for value in values:
        rect = parse_rect(value, argument_name)
        if rect is not None:
            rects.append(rect)
    return rects


def process_single_video(
    input_path: Path,
    args: argparse.Namespace,
    output_paths: Optional[Dict[str, object]] = None,
) -> dict:
    """Run detection, statistics, and optional event clips for one video."""
    run_started = time.time()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_paths is None:
        output_path = Path(args.output) if args.output else None
        csv_path = Path(args.csv) if args.csv else None
        summary_csv_path = Path(args.summary_csv) if args.summary_csv else None
        crossings_csv_path = Path(args.crossings_csv) if args.crossings_csv else None
        aoi_events_csv_path = Path(args.aoi_events_csv) if args.aoi_events_csv else None
        activity_csv_path = Path(args.activity_csv) if args.activity_csv else None
        run_summary_json_path = Path(args.run_summary_json) if args.run_summary_json else None
        counting_config_out_path = Path(args.counting_config_out) if args.counting_config_out else None
        default_clip_dir = Path(args.batch_output_dir) / safe_stem(input_path) / f"{safe_stem(input_path)}_event_clips"
        event_clips_dir = Path(args.event_clips_dir) if args.event_clips_dir else default_clip_dir
    else:
        output_path = output_paths.get("annotated_video")
        csv_path = output_paths.get("track_points_csv")
        summary_csv_path = output_paths.get("track_summary_csv")
        crossings_csv_path = output_paths.get("crossings_csv")
        aoi_events_csv_path = output_paths.get("aoi_events_csv")
        activity_csv_path = output_paths.get("activity_csv")
        run_summary_json_path = output_paths.get("run_summary_json")
        counting_config_out_path = output_paths.get("counting_config_out")
        event_clips_dir = output_paths["event_clips_dir"]

    cfg = ThermalBlobConfig(
        threshold=args.threshold,
        motion_threshold=args.motion_threshold,
        use_motion_gate=args.motion_gate,
        min_area=args.min_area,
        max_area=args.max_area,
        morph_open=args.morph_open,
        morph_dilate=args.morph_dilate,
        max_link_distance=args.max_link_distance,
        max_gap_frames=args.max_gap_frames,
        min_track_lifetime=args.min_track_lifetime,
        use_prediction=not args.no_prediction,
        draw_valid_only=not args.draw_all_tracks,
        min_track_displacement=args.min_track_displacement,
        min_track_path_length=args.min_track_path_length,
        min_mean_speed=args.min_mean_speed,
        max_mean_speed=args.max_mean_speed,
        min_directionality=args.min_directionality,
        min_track_max_blob_area=args.min_track_max_blob_area,
        min_track_mean_blob_area=args.min_track_mean_blob_area,
        max_detections_per_frame=args.max_detections_per_frame,
        background_frames=args.background_frames,
        background_stride=args.background_stride,
        background_percentile=args.background_percentile,
        background_recalibrate_interval=args.background_recalibrate_interval,
        background_recalibrate_frames=args.background_recalibrate_frames,
        background_recalibrate_stride=args.background_recalibrate_stride,
        background_recalibrate_blend=args.background_recalibrate_blend,
        roi=parse_rect(args.roi, "ROI"),
        exclude_zones=parse_rect_list(args.exclude_zone, "exclude zone"),
        draw_inactive_tracks=not args.hide_inactive_tracks,
        trail_length=args.trail_length,
        annotation_style=args.annotation_style,
        track_line_thickness=max(0, args.track_line_thickness),
        bbox_thickness=max(1, args.bbox_thickness),
        bbox_padding=max(0, args.bbox_padding),
        current_point_radius=max(1, args.current_point_radius),
        show_track_id=args.show_track_id,
        draw_roi=not args.hide_roi_rectangle,
        draw_exclude_zones=not args.hide_exclude_zones,
    )

    detector = ThermalBlobDetector(cfg)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analysis_start_frame, analysis_end_frame, progress_total = resolve_frame_range(
        frame_count, args.start_frame, args.end_frame, args.max_frames
    )

    print(f"Input: {input_path}")
    print(f"Video: {width}x{height}, fps={fps:.3f}, frames={frame_count}")
    print("Building background model...")
    if analysis_start_frame == 0:
        detector.build_background(cap)
    else:
        detector.background = detector.build_background_window(
            cap, analysis_start_frame, cfg.background_frames, cfg.background_stride
        )
        detector.previous_gray = None
    print("Background ready.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, analysis_start_frame)
    print(
        f"Analysis frame range: {analysis_start_frame}-{analysis_end_frame} "
        f"({progress_total} frames)"
    )

    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

    counting_cfg = build_counting_config_from_args(args)
    has_live_counting = bool(counting_cfg.lines or counting_cfg.aois)
    live_counter = None
    if has_live_counting:
        live_counter = LiveCounting(
            cfg=counting_cfg,
            fps=fps,
            crossings_csv_path=crossings_csv_path,
            aoi_events_csv_path=aoi_events_csv_path,
            is_countable_track=lambda track: is_valid_flying_track(track, cfg),
        )
    needs_visual = writer is not None or args.show
    overlay_renderer = OverlayRenderer() if needs_visual else None
    processed_frames = 0
    analysis_start_time = time.time()
    progress_checkpoint_time = analysis_start_time
    progress_checkpoint_frames = 0
    stage_totals = {name: 0.0 for name in ("detect", "track", "count", "overlay", "write", "show")}
    final_progress_text = ""

    try:
        frame_idx = analysis_start_frame
        while True:
            if frame_idx > analysis_end_frame:
                break
            if (
                cfg.background_recalibrate_interval > 0
                and processed_frames > 0
                and processed_frames % cfg.background_recalibrate_interval == 0
            ):
                detector.recalibrate_background(cap, frame_idx)
                print(f"Background recalibrated at frame {frame_idx}.")

            ok, frame = cap.read()
            if not ok:
                break

            stage_started = time.perf_counter()
            detections, _diff_u8, _mask = detector.detect(frame, frame_idx)
            stage_totals["detect"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            updated_tracks = detector.update_tracks(detections)
            stage_totals["track"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            if live_counter is not None:
                live_counter.update(
                    updated_tracks, frame_idx,
                    refresh_aoi_occupancy=needs_visual,
                    occupancy_tracks=list(detector.tracks.values()) if needs_visual and counting_cfg.aois else None,
                )
            stage_totals["count"] += time.perf_counter() - stage_started
            final_progress_text = (
                build_progress_text(processed_frames, progress_total, analysis_start_time)
                + f" | source frame {frame_idx}"
            )
            if needs_visual:
                stage_started = time.perf_counter()
                debug_frame = overlay_renderer.render(
                    frame, detections, detector, frame_idx,
                    counting_cfg if has_live_counting else None,
                    live_counter, final_progress_text,
                )
                stage_totals["overlay"] += time.perf_counter() - stage_started
                if writer is not None:
                    stage_started = time.perf_counter()
                    writer.write(debug_frame)
                    stage_totals["write"] += time.perf_counter() - stage_started
                if args.show:
                    stage_started = time.perf_counter()
                    cv2.imshow("Thermal blob detector MVP - valid tracks", debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    stage_totals["show"] += time.perf_counter() - stage_started
                    if key in (27, ord("q")):
                        print("Stopped by user.")
                        break

            processed_frames += 1
            if processed_frames % 100 == 0:
                checkpoint_now = time.time()
                checkpoint_elapsed = checkpoint_now - progress_checkpoint_time
                checkpoint_count = processed_frames - progress_checkpoint_frames
                recent_fps = checkpoint_count / checkpoint_elapsed if checkpoint_elapsed > 0 else 0.0
                print(f"{final_progress_text} | recent {recent_fps:.1f} fps")
                per_frame_ms = {
                    name: 1000.0 * elapsed / max(1, checkpoint_count)
                    for name, elapsed in stage_totals.items()
                }
                print(
                    "  stages ms/frame "
                    + " ".join(f"{name}={value:.2f}" for name, value in per_frame_ms.items())
                    + f" | detections={len(detections)} active={detector.active_track_count} "
                    + f"valid={detector.valid_track_count} total={len(detector.tracks)}"
                )
                progress_checkpoint_time = checkpoint_now
                progress_checkpoint_frames = processed_frames
                for name in stage_totals:
                    stage_totals[name] = 0.0
            if processed_frames >= progress_total:
                break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if live_counter is not None:
            live_counter.close()
        if args.show:
            cv2.destroyAllWindows()

    if processed_frames > 0:
        final_progress_text = (
            build_progress_text(processed_frames - 1, progress_total, analysis_start_time)
            + f" | source frame {analysis_start_frame + processed_frames - 1}"
        )
        print(final_progress_text)

    if csv_path:
        write_track_points_csv(csv_path, detector.tracks, fps, cfg)
        print(f"CSV written: {csv_path}")

    counting_results = analyze_tracks(
        tracks=detector.tracks.values(),
        fps=fps,
        cfg=counting_cfg,
        is_valid_track=lambda track: is_valid_flying_track(track, cfg),
        input_video=str(input_path),
        frame_count_processed=processed_frames,
        parameter_preset=args.parameter_preset,
        notes=f"Skipped noisy detection frames: {detector.skipped_detection_frames}",
    )

    if summary_csv_path:
        write_counting_track_summary_csv(summary_csv_path, counting_results.track_summaries)
        print(f"Enhanced track summary CSV written: {summary_csv_path}")

    if crossings_csv_path:
        write_crossings_csv(crossings_csv_path, counting_results.crossings)
        print(f"Crossings CSV written: {crossings_csv_path}")

    if aoi_events_csv_path:
        write_aoi_events_csv(aoi_events_csv_path, counting_results.aoi_events)
        print(f"AOI events CSV written: {aoi_events_csv_path}")

    if activity_csv_path:
        write_activity_csv(activity_csv_path, counting_results.activity_rows)
        print(f"Activity CSV written: {activity_csv_path}")

    if run_summary_json_path:
        write_run_summary_json(run_summary_json_path, counting_results.run_summary)
        print(f"Run summary JSON written: {run_summary_json_path}")

    if counting_config_out_path:
        write_counting_config_json(counting_config_out_path, counting_cfg)
        print(f"Counting config JSON written: {counting_config_out_path}")

    if args.event_clips:
        print("Event clips enabled.")
        print(f"Building event clip windows from {args.event_clip_trigger}...")
        clip_total_frames = analysis_end_frame + 1
        raw_windows = build_clip_windows(
            tracks=detector.tracks.values(), crossings=counting_results.crossings,
            aoi_events=counting_results.aoi_events, trigger=args.event_clip_trigger,
            pre_frames=args.event_clip_pre_frames, post_frames=args.event_clip_post_frames,
            total_frames=clip_total_frames, min_track_lifetime=cfg.min_track_lifetime,
            is_valid_track=lambda track: is_valid_flying_track(track, cfg),
        )
        for window in raw_windows:
            window.start_frame = max(analysis_start_frame, window.start_frame)
            window.end_frame = min(analysis_end_frame, window.end_frame)
        windows = merge_clip_windows(raw_windows, args.event_clip_merge_gap_frames)
        print(f"Raw windows: {len(raw_windows)}")
        print(f"Merged windows: {len(windows)}")
        if not windows:
            print("No event clip windows found; no clips were written.")
        else:
            manifest = export_event_clips(
                input_path, event_clips_dir, windows, fps, width, height,
                args.event_clip_fourcc,
                lambda frame, frame_idx, window, clip_idx, clip_count: draw_event_clip_overlay(
                    frame, frame_idx, window, detector.tracks, counting_cfg, cfg, clip_idx, clip_count
                ),
                {track.track_id for track in detector.valid_tracks()},
            )
            write_clip_manifest(event_clips_dir, manifest)
            print(f"Event clip manifest written: {event_clips_dir / 'event_clips_manifest.csv'}")

    if output_path:
        print(f"Debug video written: {output_path}")

    all_tracks = list(detector.tracks.values())
    confirmed = detector.confirmed_tracks()
    valid = detector.valid_tracks()
    print("Done.")
    print(f"Frames processed: {processed_frames}")
    print(f"Skipped noisy detection frames: {detector.skipped_detection_frames}")
    print(f"All tracks: {len(all_tracks)}")
    print(f"Confirmed tracks >= {cfg.min_track_lifetime} detections: {len(confirmed)}")
    print(f"Valid flying tracks: {len(valid)}")
    print(f"Line crossings: {len(counting_results.crossings)}")
    print(f"AOI events: {len(counting_results.aoi_events)}")
    return {
        "run_summary_json": str(run_summary_json_path) if run_summary_json_path else "",
        "track_points_csv": str(csv_path) if csv_path else "",
        "track_summary_csv": str(summary_csv_path) if summary_csv_path else "",
        "crossings_csv": str(crossings_csv_path) if crossings_csv_path else "",
        "aoi_events_csv": str(aoi_events_csv_path) if aoi_events_csv_path else "",
        "activity_csv": str(activity_csv_path) if activity_csv_path else "",
        "event_clips_dir": str(event_clips_dir) if args.event_clips else "",
        "processed_frames": processed_frames,
        "valid_tracks": len(valid),
        "line_crossings": len(counting_results.crossings),
        "aoi_events": len(counting_results.aoi_events),
        "event_clip_count": len(manifest) if args.event_clips and windows else 0,
        "elapsed_seconds": round(time.time() - run_started, 3),
    }


def process_video(args: argparse.Namespace) -> dict:
    """Backward-compatible single-input entry point."""
    if not args.input:
        raise ValueError("process_video requires --input")
    return process_single_video(Path(args.input), args)


def process_batch(args: argparse.Namespace) -> List[dict]:
    input_paths = collect_input_videos(args)
    if not input_paths:
        raise ValueError("No matching input videos found")
    if args.batch_output_dir:
        batch_output_dir = Path(args.batch_output_dir)
    elif args.input_dir:
        batch_output_dir = Path(args.input_dir).resolve() / "outputs"
    else:
        batch_output_dir = input_paths[0].resolve().parent / "outputs"
    print(f"Batch output folder: {batch_output_dir}")
    print(f"Found {len(input_paths)} input videos.")
    if args.event_clips_dir and len(input_paths) > 1:
        print("WARNING: --event-clips-dir is shared in batch mode; clip filenames may collide between videos.")
    rows: List[dict] = []
    for index, input_path in enumerate(input_paths, start=1):
        print(f"[{index}/{len(input_paths)}] Processing: {input_path}")
        paths = build_output_paths(input_path, batch_output_dir, args)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        summary_path = paths.get("run_summary_json")
        if args.skip_existing and summary_path and summary_path.exists():
            print(f"Skipping existing result: {summary_path}")
            rows.append({
                "input_path": str(input_path), "status": "skipped", "error": "",
                "output_dir": str(paths["output_dir"]), "run_summary_json": str(summary_path),
            })
            continue
        try:
            result = process_single_video(input_path, args, paths)
            rows.append({
                "input_path": str(input_path), "status": "ok", "error": "",
                "output_dir": str(paths["output_dir"]), **result,
            })
            print(f"Done: {input_path}")
        except Exception as exc:
            if not args.continue_on_error:
                write_batch_summary(batch_output_dir, rows + [{
                    "input_path": str(input_path), "status": "error", "error": str(exc),
                    "output_dir": str(paths["output_dir"]),
                }])
                raise
            print(f"ERROR processing {input_path}: {exc}")
            rows.append({
                "input_path": str(input_path), "status": "error", "error": str(exc),
                "output_dir": str(paths["output_dir"]),
            })
    write_batch_summary(batch_output_dir, rows)
    print(f"Batch summary written: {batch_output_dir / 'batch_summary.csv'}")
    return rows


def build_counting_config_from_args(args: argparse.Namespace) -> CountingConfig:
    counting_cfg = CountingConfig()

    if args.counting_config:
        counting_cfg = load_counting_config(Path(args.counting_config))

    for value in args.count_line or []:
        counting_cfg.lines.append(line_from_cli(value))
    for value in args.count_aoi or []:
        counting_cfg.aois.append(aoi_from_cli(value))

    if args.count_all_tracks:
        counting_cfg.count_valid_tracks_only = False
    if args.activity_bin_seconds is not None:
        counting_cfg.activity_bin_seconds = args.activity_bin_seconds
    if args.line_crossing_epsilon is not None:
        counting_cfg.line_crossing_epsilon = args.line_crossing_epsilon
    if args.min_frames_between_same_line_crossing is not None:
        counting_cfg.min_frames_between_same_line_crossing = args.min_frames_between_same_line_crossing
    if args.aoi_boundary_debounce_frames is not None:
        counting_cfg.aoi_boundary_debounce_frames = args.aoi_boundary_debounce_frames
    return counting_cfg
