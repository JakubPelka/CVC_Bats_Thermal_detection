# Changelog

All notable changes to this project will be documented in this file.

This project follows a pragmatic pre-release workflow. The current goal is to stabilise the first working release, `v0.1.0`, based on real thermal bat video tests.

## [Unreleased]

### Added for v0.1.1-alpha

- Optional periodic background recalibration with configurable sampling and blending.
- Analysis progress in the terminal and preview HUD with total frames, percentage,
  elapsed time, processing FPS and ETA.
- Flat-by-default, recording-prefixed batch and event-clip exports, with an
  optional per-input-folder switch.
- Recording filename and absolute recording time in annotated event-clip HUDs.
- Stable event-clip manifest ordering with date/time after `clip_id` and
  `track_ids` retained as the final CSV/JSON field.

### Planned

- Validate detection, tracking and counting on additional thermal bat recordings.
- Tune default parameter presets using more sample videos.
- Review generated outputs and ensure no large or sensitive files are committed.
- Improve documentation based on real test workflow.
- Prepare release notes for `v0.1.0`.

### Considered for `v0.1.1`

- Explore replacing or complementing the Tkinter GUI with a clearer step-by-step frontend.
- Possible guided workflow:
  1. choose input video,
  2. choose output folder,
  3. accept or adjust detection parameters,
  4. enable or disable extra features,
  5. draw ROI / exclude zones / counting geometry,
  6. run processing,
  7. review output files.
- Keep the detector and statistics logic independent from the GUI/frontend layer.

## [0.1.0] - 2026-06-17

### Added

- Initial dedicated thermal bat detection workflow.
- Bright thermal blob detection based on static background modelling and thresholding.
- Simple centroid tracking for small moving thermal objects.
- Valid-track filtering to reduce camera artefacts and noise.
- GUI-first workflow for parameter tuning and processing.
- Drawing support for detection ROI and exclude zones from video frames.
- Drawing support for counting geometry from video frames.
- Track-based line crossing counting.
- Track-based AOI event handling.
- Activity statistics over time.
- Enhanced track summary output.
- Run summary output as JSON.
- CSV exports for track points, track summaries, line crossings, AOI events and activity over time.
- Optional annotated video output.
- Loadable parameter presets.
- Example counting configuration.
- Basic synthetic tests for counting and statistics logic.
- Python packaging with console entry points:
  - `thermal-blob-detector`
  - `thermal-blob-detector-gui`

### Notes

- This is an experimental working prototype.
- The tool is currently kept as a dedicated thermal bat workflow and is not integrated into `ComputerVisionCounter_video`.
- Initial tests show that detection, counting and statistics work on the available sample data.
- More recordings are needed before default parameters can be treated as robust.
- Sample videos, annotated videos and generated output files should not be committed to the repository unless they are deliberately small, anonymised examples.
