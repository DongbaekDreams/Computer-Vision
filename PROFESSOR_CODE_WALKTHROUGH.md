# Computer Vision Project Walkthrough (Professor Q&A Focus)

This document explains the major parts of the codebase and the reasoning behind design choices, so you can confidently answer "how does it work?" questions.

## 1) High-Level System Overview

There are two main tracks in this repository:

1. `Project/` - a real-time pose dashboard (OpenCV + MediaPipe) for live camera processing, angle extraction, recording, and review.
2. `MentalHealthAdditional/video-analysis.ipynb` - offline analysis notebook for blink-rate and facial measurement statistics from recorded videos.

Core idea:
- **Live dashboard**: captures camera frames -> detects pose -> computes 2D or 3D joint angles -> smooths signals -> visualizes + records.
- **Notebook analysis**: loads face videos -> extracts face landmarks -> computes EAR/blink metrics and anthropometric-style measurements -> builds plots/tables.

---

## 2) Real-Time Dashboard (`Project/`)

## 2.1 Entry Point and Runtime Loop

Main file: `Project/main.py`

What it does:
- Loads configuration constants (`config.py`).
- Loads camera calibration + last setup (`camera_config.py`).
- Initializes MediaPipe pose landmarker.
- Opens cameras (single or multi-camera).
- Runs a per-frame loop:
  - read camera batch,
  - run pose detection,
  - compute angles (2D or triangulated 3D),
  - smooth data (One Euro filters),
  - update live/record buffers,
  - draw dashboard UI (video + polar + timeline + clip),
  - handle keyboard/mouse controls.

Why this structure matters:
- It keeps all real-time decisions in one loop, so timing, buffering, and UI stay synchronized.

---

## 2.2 Configuration Layer

File: `Project/config.py`

Role:
- Central source of constants for:
  - camera sizes and inference scaling,
  - confidence thresholds (`POSE_MIN_*`),
  - smoothing params (`SMOOTH_2D_*`, `SMOOTH_3D_*`),
  - UI colors/layout,
  - timeline/polar export behavior,
  - model asset path (`pose_landmarker_full.task`).

Professor-ready explanation:
- "I separated hyperparameters and styling into `config.py` so I can tune quality, performance, and UX without touching algorithmic code."

---

## 2.3 Pose Processing (2D Angles)

File: `Project/pose_processor.py`

Key responsibilities:
- Converts MediaPipe landmarks into pixel-space and normalized coordinates.
- Computes joint angles:
  - hips, knees, ankles, shoulders, elbows.
- Handles mirrored preview correctly by swapping left/right landmark indices.
- Provides `best_2d_angle_values(...)` fallback when one camera loses tracking.

Important design point:
- Angle values are clamped to biomechanical range `[0, 180]` and use visibility gating to avoid noisy/invalid geometry.

---

## 2.4 Multi-Camera Geometry (3D Angles)

File: `Project/triangulation.py`

Key responsibilities:
- Triangulates 3D landmarks from 2+ camera views using projection matrices.
- Supports two-view (`cv2.triangulatePoints`) and multi-view SVD formulation.
- Computes 3D joint angles from triangulated points.
- Falls back to best 2D angles if 3D quality is insufficient.

Professor-ready explanation:
- "3D is only used when extrinsics exist and enough confident landmarks are visible in at least two cameras; otherwise the system degrades gracefully to robust 2D."

---

## 2.5 Calibration and Camera Management

Files:
- `Project/camera_config.py`
- `Project/calibrate_cameras.py`
- `Project/calibrate_extrinsics_3d.py`

### `camera_config.py`
- Defines calibration dataclasses (`Intrinsics`, `Extrinsics`, `Calibration`, `LastCameraSetup`).
- Loads/saves JSON calibration files.
- Handles camera detection and stable opening logic.
- Includes `MultiCameraReader` with serialized USB reads to reduce Windows multi-webcam instability.

### `calibrate_cameras.py` (intrinsics workflow)
- Camera selection UI (choose cameras, primary, rotation).
- Records calibration videos from selected cameras.
- Detects chessboard corners from videos and solves intrinsics with `cv2.calibrateCamera`.
- Saves `camera_calibrations.json` + `last_camera_setup.json`.

### `calibrate_extrinsics_3d.py` (extrinsics workflow)
- Uses calibrated cameras + board snapshot to solve each camera pose (`solvePnP`).
- Writes camera extrinsics into shared world frame for triangulation.

Professor-ready explanation:
- "Intrinsics correct lens and projection properties per camera; extrinsics align all cameras into one coordinate system so triangulation is physically meaningful."

---

## 2.6 Temporal Smoothing

File: `Project/adaptive_filter.py`

Role:
- Implements One Euro filtering for landmarks in 2D and 3D.
- Reduces jitter while preserving responsiveness to fast motion.

Why this is useful:
- Raw landmark streams are noisy; filtered trajectories improve angle stability and visual readability.

---

## 2.7 State, Buffers, and Timeline

Files:
- `Project/state.py`
- `Project/ui/timeline.py`

### `state.py`
- Stores live and recorded deques:
  - timestamps,
  - angle series,
  - pose snapshots.

### `ui/timeline.py`
- Timeline controls (LIVE/REVIEW, REC, PLAY, CLEAR).
- Segment extraction for fixed windows.
- Mouse drag/click interactions for selecting review windows.

Professor-ready explanation:
- "I keep app-state in deques for efficient append/pop in streaming use, then the timeline slices those buffers into review windows for synchronized plots and clip playback."

---

## 2.8 Visualization + UI Components

Main files:
- `Project/visualization/skeleton.py` - overlays pose skeleton/joints on video.
- `Project/visualization/polar_plot.py` - angle trajectories in polar space; also PNG export.
- `Project/visualization/clip_preview.py` - compact pose animation clip.
- `Project/ui/drawing.py` - reusable panel/table/stat drawing utilities.
- `Project/ui/export_gallery.py` and `Project/ui/export_viewer.py` - browse saved polar exports.
- `Project/ui/stage_modal.py` - overlay modal layout for in-window dialogs.
- `Project/ui/console.py` - terminal status-line formatting.

Design rationale:
- UI and visualization modules are split from core logic to keep the pipeline testable and easier to reason about.

---

## 3) Data Flow You Can Explain in 20 Seconds

1. Camera frames are captured (single/multi).
2. MediaPipe Pose detects landmarks per frame.
3. 2D angles are computed immediately from landmark triplets.
4. If 2+ calibrated cameras with extrinsics are available, landmarks are triangulated -> 3D angles are computed.
5. 2D/3D landmark streams are smoothed with One Euro filters.
6. Data is pushed into live and optional recording buffers.
7. Dashboard renders video overlay, stats, timeline, clip, and polar plot.

---

## 4) Notebook Analysis (`MentalHealthAdditional/video-analysis.ipynb`)

This notebook has three major analysis blocks.

## 4.1 Blink Pipeline + Caching

What happens:
- Loads cached files if present:
  - `blink_analysis_data.npz`
  - `blink_analysis_stats.pkl`
- Otherwise computes from video folder:
  - reads MP4s,
  - runs MediaPipe Face Landmarker,
  - computes EAR (Eye Aspect Ratio) from eye landmark sets,
  - sweeps thresholds to estimate blink count/rate,
  - saves cache.

Key formula idea:
- EAR is a ratio of vertical eye opening to horizontal width; lower EAR indicates eye closure.

## 4.2 Single Time-Series and Group-Wise Blink Analysis

What happens:
- Plots threshold vs blink rate.
- Plots EAR over time.
- Computes sliding-window blink-rate trend.
- Reconstructs per-video boundaries from durations.
- Builds per-video summary table:
  - duration,
  - blink count,
  - blink rate,
  - mean EAR,
  - EAR std dev.
- Excludes very short videos (<5s) from group stats.

Why this matters:
- Separating by video avoids hidden bias from concatenated timelines.

## 4.3 Face Measurements Block

What happens:
- Uses MediaPipe face landmarks plus known scale references:
  - IPD-based scaling (known interpupillary distance),
  - iris-based scaling.
- Samples frames at stride and max-per-video limits.
- Computes face/eye/nose/mouth dimensions in:
  - normalized relative units (`x IPD`),
  - estimated centimeters via IPD/iris scaling.
- Aggregates measurements into summary DataFrames and visual outputs.

---

## 5) Expected Professor Questions (with Short Answers)

- **Q: Why choose MediaPipe Tasks API?**  
  A: It provides robust pretrained landmark detectors with practical real-time performance and easy Python integration.

- **Q: When do you use 2D vs 3D angles?**  
  A: 2D is always available; 3D is used only when at least two calibrated cameras with extrinsics and sufficient visibility are present.

- **Q: How do you keep multi-camera capture stable on Windows?**  
  A: Serialized USB reads, careful backend/mode selection, frame draining, and fallback setup logic reduce index and buffering issues.

- **Q: Why smooth landmarks instead of smoothing angles directly?**  
  A: Landmark-space smoothing preserves geometry consistency before angle computation, reducing jitter without distorting joint relationships.

- **Q: What are key failure modes?**  
  A: Occlusion/low visibility, missing extrinsics, camera index drift, and low-quality calibration data.

- **Q: How is blink counting implemented?**  
  A: Threshold crossing with state machine (`closed` flag) on EAR time-series, then normalized by duration for blink rate.

- **Q: Why cache notebook outputs?**  
  A: Face landmark extraction across long videos is expensive; caching makes iterative analysis and plotting reproducible and faster.

---

## 6) Practical "Defense" Talking Points

- "The system is intentionally fault-tolerant: if 3D fails, it degrades to best available 2D rather than dropping output."
- "Calibration is split into intrinsics and extrinsics because they solve different geometry problems."
- "The app is designed as a streaming pipeline with explicit state buffers, so live view and review mode can share one source of truth."
- "Notebook analyses are separated from live dashboard logic to keep exploratory statistics independent from real-time constraints."

---

## 7) Key Files to Know by Name

- `Project/main.py`
- `Project/config.py`
- `Project/pose_processor.py`
- `Project/triangulation.py`
- `Project/camera_config.py`
- `Project/calibrate_cameras.py`
- `Project/calibrate_extrinsics_3d.py`
- `Project/adaptive_filter.py`
- `Project/state.py`
- `Project/ui/timeline.py`
- `Project/visualization/polar_plot.py`
- `MentalHealthAdditional/video-analysis.ipynb`

