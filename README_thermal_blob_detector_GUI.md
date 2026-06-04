# Thermal Bat Blob Detector - Tkinter GUI

This GUI is a small launcher for:

`thermal_blob_detector_mvp_v3_valid_tracks.py`

It does not replace the detector. It only lets you change parameters from a window and run the detector without editing code.

## Files

Place these files in the same folder:

```text
thermal_blob_detector_mvp_v3_valid_tracks.py
thermal_blob_detector_gui.py
```

## Requirements

```bash
pip install opencv-python numpy
```

Tkinter is included with most standard Python installations on Windows.

## Run

```bash
python thermal_blob_detector_gui.py
```

## Recommended workflow

1. Choose the detector script if it is not found automatically.
2. Choose input video.
3. Start with **Quick 500-frame test**.
4. Try presets:
   - **Good current defaults**
   - **Loose / rescue more bats**
   - **Strict / remove artefacts**
   - **Diagnostic all tracks**
5. Run detector.
6. Check output video and CSV.
7. Save a good parameter combination as JSON.

## Important parameters

### Detection

- `Brightness threshold`  
  Higher value = fewer weak blobs, lower value = more weak detections and more noise.

- `Max blob area`  
  Useful when close/bright bats become larger blobs. Current useful value from testing: `1200`.

- `Morph open / dilate`  
  Controls mask cleanup. Keep small for tiny thermal objects.

### Tracking

- `Max link distance`  
  Higher value helps reconnect fast/near bats between frames. Current useful value from testing: `90`.

- `Max gap frames`  
  How long a track can survive missing detections.

- `Min track lifetime`  
  Higher value removes flashes/noise, but can remove very fast bats visible only briefly.

### Track filters

These remove camera artefacts by checking whether a track behaves like a moving/flying object.

- `Min track displacement`
- `Min track path length`
- `Min mean speed`
- `Min directionality`
- `Max detections/frame`

If valid bats disappear, use the **Loose / rescue more bats** preset.

If camera artefacts remain, use the **Strict / remove artefacts** preset or add exclude zones.

## ROI and exclude zones

### ROI

Format:

```text
x,y,w,h
```

Example:

```text
100,50,800,500
```

### Exclude zones

Use one rectangle per line:

```text
100,50,80,80
600,20,50,120
```

These are useful for fixed camera artefacts, hot pixels, UI marks or permanently problematic corners.

## Notes

The GUI shows the generated command. You can copy it and run it manually if needed.

The detector still performs no line/AOI counting. This is intentional. The output is meant to remain a detector/tracker prototype before later CVC integration.


## Preview window

The OpenCV preview window is enabled by default in this updated GUI.

To disable it, go to the **Flags** tab and uncheck:

```text
Show OpenCV preview window
```

If the preview opens behind the Tkinter window, use Alt+Tab or click the OpenCV window. Press `q` or `Esc` in the preview window to stop preview processing.


## Parameter explanation column

This GUI version adds an **Explanation** column in the parameter tabs.

It describes:
- what each parameter controls,
- when to increase it,
- when to decrease it,
- which settings are mainly diagnostic.

The **Flags** tab also includes explanations for each checkbox.
