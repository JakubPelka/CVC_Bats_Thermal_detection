# Thermal Bat Blob Detector GUI - drawn geometry

This GUI version adds drawing helpers for rectangular detection ROI/exclude zones and track-based counting lines/AOIs.

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

## Counting lines and AOIs

In the **Counting / Statistics** tab, use:

```text
Draw lines / AOIs
```

Workflow:

1. Choose input video.
2. Open **Counting / Statistics**.
3. Choose a frame index.
4. Click **Draw lines / AOIs**.
5. Choose `Line`, `Polyline`, or `AOI polygon`.
6. Click points on the video frame. For polylines and AOI polygons, press Enter to finish.
7. Give the object a name and click **Save JSON**.

The GUI saves the JSON path in **Counting config JSON**, so the run uses `--counting-config` automatically. Drawn lines count crossings by track movement; drawn AOI polygons count entry and exit events.

Rectangular detection ROI/exclude zones still use:

```text
--roi x,y,w,h
--exclude-zone x,y,w,h
```

## Run

```bash
PYTHONPATH=src python -m gui
```

The OpenCV preview window is enabled by default.
