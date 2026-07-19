"""Command-line interface for thermal video processing."""

import argparse

from .config import TRACK_COLOR_PALETTE
from .pipeline import process_batch, process_video


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and track bright moving blobs in tripod thermal video. "
            "Draws valid flight-like tracks by default and can export track-based line/AOI counting statistics."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", default="", help="Input thermal video file")
    input_group.add_argument("--inputs", nargs="+", default=None, help="Explicit list of input video files")
    input_group.add_argument("--input-dir", default="", help="Folder containing input videos")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --input-dir")
    parser.add_argument("--video-extensions", default=".mp4,.avi,.mov,.mkv", help="Comma-separated extensions for folder scans")
    parser.add_argument(
        "--batch-output-dir", default="",
        help="Root output directory for batch results; empty creates an outputs folder beside the input recordings",
    )
    parser.add_argument(
        "--output-per-input-folder", action="store_true",
        help="Put each recording's results in its own subfolder; by default all prefixed results share one output folder",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing after a video fails")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos whose derived run summary already exists")
    parser.add_argument("--output", default="thermal_blob_valid_tracks.mp4", help="Output debug video path")
    parser.add_argument("--csv", default="thermal_blob_track_points.csv", help="Output CSV path for per-detection track points")
    parser.add_argument("--summary-csv", default="thermal_blob_track_summary.csv", help="Enhanced output CSV path for per-track summary")
    parser.add_argument("--crossings-csv", default="crossings.csv", help="Output CSV path for line crossing events")
    parser.add_argument("--aoi-events-csv", default="aoi_events.csv", help="Output CSV path for AOI entry/exit events")
    parser.add_argument("--activity-csv", default="activity_by_time.csv", help="Output CSV path for activity time bins")
    parser.add_argument("--run-summary-json", default="run_summary.json", help="Output JSON path for compact run summary")
    parser.add_argument("--counting-config-out", default="", help="Optional path to save the effective counting config JSON")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame to analyze (inclusive)")
    parser.add_argument("--end-frame", type=int, default=0, help="Last source frame to analyze (inclusive); 0 uses video end")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional processing limit for quick tests")
    parser.add_argument("--event-clips", action="store_true", help="Export annotated activity clips after analysis")
    parser.add_argument("--event-clips-dir", default="", help="Optional event clip directory override")
    parser.add_argument("--event-clip-pre-frames", type=int, default=100, help="Frames included before each activity window")
    parser.add_argument("--event-clip-post-frames", type=int, default=100, help="Frames included after each activity window")
    parser.add_argument("--event-clip-merge-gap-frames", type=int, default=100, help="Maximum gap between windows to merge")
    parser.add_argument(
        "--event-clip-trigger", default="valid_tracks",
        choices=("valid_tracks", "all_tracks", "crossings", "aois", "all_events"),
        help="Activity source used to create event clip windows",
    )
    parser.add_argument("--event-clip-fourcc", default="mp4v", help="FourCC codec for event clip video output")

    parser.add_argument("--threshold", type=float, default=30.0, help="Brightness difference threshold above background")
    parser.add_argument(
        "--motion-gate", action=argparse.BooleanOptionalAction, default=True,
        help="Require both bright-above-background and frame-to-frame motion",
    )
    parser.add_argument("--motion-threshold", type=float, default=25.0, help="Frame-to-frame motion threshold")

    parser.add_argument("--min-area", type=int, default=3, help="Minimum blob area in pixels")
    parser.add_argument("--max-area", type=int, default=1200, help="Maximum blob area in pixels")
    # Deprecated compatibility options. Blob shape is no longer filtered by its
    # bounding-box dimensions; area-based track filters are more robust for
    # small, streaked thermal targets.
    parser.add_argument("--min-width", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-height", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-width", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-height", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--morph-open", type=int, default=1, help="Morphological opening kernel size; 0 disables")
    parser.add_argument("--morph-dilate", type=int, default=1, help="Dilation kernel size; 0 disables")

    parser.add_argument("--max-link-distance", type=float, default=90.0, help="Maximum centroid/predicted-centroid distance for track linking")
    parser.add_argument("--max-gap-frames", type=int, default=4, help="How many frames a track may be missing before closing")
    parser.add_argument("--min-track-lifetime", type=int, default=3, help="Minimum detections before a track can be valid")
    parser.add_argument("--no-prediction", action="store_true", help="Disable simple velocity prediction during track linking")

    parser.add_argument("--draw-all-tracks", action="store_true", help="Draw all tracks, including invalid/static/noisy ones")
    parser.add_argument("--min-track-displacement", type=float, default=12.0, help="Minimum start-to-end displacement for valid flying track")
    parser.add_argument("--min-track-path-length", type=float, default=18.0, help="Minimum total path length for valid flying track")
    parser.add_argument("--min-mean-speed", type=float, default=0.8, help="Minimum mean speed in pixels/frame for valid flying track")
    parser.add_argument("--max-mean-speed", type=float, default=120.0, help="Maximum mean speed in pixels/frame for valid flying track")
    parser.add_argument("--min-directionality", type=float, default=0.15, help="Minimum net_displacement/path_length ratio for valid flying track")
    parser.add_argument(
        "--min-track-max-blob-area", type=int, default=14,
        help="Require at least one detection in a valid track to reach this area; 0 disables",
    )
    parser.add_argument(
        "--min-track-mean-blob-area", type=float, default=8.0,
        help="Minimum mean detection area across a valid track; 0 disables",
    )
    parser.add_argument("--max-detections-per-frame", type=int, default=40, help="Skip frame detections above this count; 0 disables")

    parser.add_argument("--background-frames", type=int, default=200, help="Number of sampled frames for background model")
    parser.add_argument("--background-stride", type=int, default=10, help="Frame step between background samples")
    parser.add_argument("--background-percentile", type=float, default=50.0, help="Background percentile; 50=median")
    parser.add_argument("--background-recalibrate-interval", type=int, default=1000, help="Rebuild background every N processed frames; 0 disables")
    parser.add_argument("--background-recalibrate-frames", type=int, default=200, help="Number of sampled frames for periodic background recalibration")
    parser.add_argument("--background-recalibrate-stride", type=int, default=10, help="Frame step between periodic background samples")
    parser.add_argument("--background-recalibrate-blend", type=float, default=0.5, help="New background blend strength from 0.0 to 1.0")

    parser.add_argument("--roi", default=None, help="Optional rectangular ROI as x,y,w,h")
    parser.add_argument(
        "--exclude-zone",
        action="append",
        default=None,
        help="Optional exclusion rectangle x,y,w,h. Can be repeated for fixed camera artefact zones.",
    )
    parser.add_argument("--hide-inactive-tracks", action="store_true", help="Do not draw inactive tracks")
    parser.add_argument("--trail-length", type=int, default=0, help="Recent points drawn per track. 0 = full track history")
    parser.add_argument(
        "--annotation-style", choices=("trail", "thin-trail", "bbox", "bbox-trail", "dot", "minimal"),
        default="bbox-trail", help="Track annotation style used in preview, output video, and event clips",
    )
    parser.add_argument(
        "--track-color-mode", choices=("random", "fixed"), default="random",
        help="Use deterministic per-track colors or one fixed annotation color",
    )
    parser.add_argument(
        "--track-fixed-color", choices=tuple(TRACK_COLOR_PALETTE), default="cyan",
        help="Fixed annotation color used when --track-color-mode=fixed",
    )
    parser.add_argument("--track-line-thickness", type=int, default=1, help="Track trail line thickness; 0 disables the trail")
    parser.add_argument("--bbox-thickness", type=int, default=1, help="Current detection bounding-box thickness")
    parser.add_argument("--bbox-padding", type=int, default=4, help="Padding around current detection bounding boxes")
    parser.add_argument("--current-point-radius", type=int, default=3, help="Current centroid marker radius")
    parser.add_argument(
        "--show-track-id", action=argparse.BooleanOptionalAction, default=True,
        help="Show the track ID beside current annotations",
    )
    parser.add_argument(
        "--verification-mode", action=argparse.BooleanOptionalAction, default=False,
        help="Render event clips as side-by-side views of the same frame",
    )
    parser.add_argument(
        "--verification-left-style", choices=("trail", "thin-trail", "bbox", "bbox-trail", "dot", "minimal", "raw"),
        default="bbox-trail", help="Annotation style for the left verification view",
    )
    parser.add_argument(
        "--verification-right-style", choices=("trail", "thin-trail", "bbox", "bbox-trail", "dot", "minimal", "raw"),
        default="raw", help="Annotation style for the right verification view; raw shows the unannotated source frame",
    )
    parser.add_argument("--hide-roi-rectangle", action="store_true", help="Do not draw ROI rectangle")
    parser.add_argument("--hide-exclude-zones", action="store_true", help="Do not draw exclusion-zone rectangles")

    parser.add_argument("--counting-config", default="", help="Optional JSON file with counting lines, AOIs and counting settings")
    parser.add_argument(
        "--count-line",
        action="append",
        default=None,
        help="Counting line as id,name,x1,y1,x2,y2[,positive_label,negative_label]. Can be repeated.",
    )
    parser.add_argument(
        "--count-aoi",
        action="append",
        default=None,
        help="Rectangular counting AOI as id,name,x,y,w,h. Can be repeated.",
    )
    parser.add_argument("--count-all-tracks", action="store_true", help="Diagnostic counting mode: count all tracks, not only valid flying tracks")
    parser.add_argument("--activity-bin-seconds", type=float, default=None, help="Activity statistics bin size in seconds")
    parser.add_argument("--line-crossing-epsilon", type=float, default=None, help="Pixel-side tolerance for line crossing tests")
    parser.add_argument("--min-frames-between-same-line-crossing", type=int, default=None, help="Debounce repeated crossings of the same line by one track")
    parser.add_argument("--aoi-boundary-debounce-frames", type=int, default=None, help="Debounce repeated AOI entry/exit events near a boundary")
    parser.add_argument("--parameter-preset", default="custom", help="Preset label written to run_summary.json")

    parser.add_argument("--fourcc", default="mp4v", help="Output video codec fourcc, e.g. mp4v or XVID")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.input:
        process_video(args)
    else:
        process_batch(args)


if __name__ == "__main__":
    main()
