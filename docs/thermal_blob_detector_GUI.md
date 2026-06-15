# Thermal Bat Blob Detector GUI - drawn ROI and exclude zones

This GUI version adds a drawing helper for rectangular ROI and exclude zones.

It is still only a launcher for:

```text
thermal_blob_detector
```

The detector itself is unchanged.

## What is new

In the **ROI / exclude** tab there is now a button:

```text
Draw ROI / exclude zones
```

Workflow:

1. Choose input video.
2. Open **ROI / exclude** tab.
3. Choose frame index, for example `0`.
4. Click **Draw ROI / exclude zones**.
5. In the drawing window choose mode:
   - `ROI`
   - `Exclude zone`
6. Drag rectangles with the left mouse button.
7. The GUI writes `x,y,w,h` values automatically.

## Controls in drawing window

- **ROI mode**  
  Replaces the current ROI with the rectangle you draw.

- **Exclude zone mode**  
  Adds one exclude rectangle per drag.

- **Clear ROI**  
  Removes current ROI.

- **Clear exclude zones**  
  Removes all exclude rectangles.

- **Undo last exclude**  
  Removes the most recently added exclude rectangle.

- **Reload frame**  
  Loads another frame index from the same video.

## Notes

The detector currently supports rectangular ROI/exclude zones. This GUI follows that structure.

Polygon drawing can be added later, but it would require a detector-side change too, because the current CLI uses:

```text
--roi x,y,w,h
--exclude-zone x,y,w,h
```

## Run

```bash
PYTHONPATH=src python -m gui
```

The OpenCV preview window is enabled by default.
