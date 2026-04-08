# Eyes on Motion: Capturing the Beauty of Human Movement

## First Page Content (for PDF)

### 1) Problem Being Solved
Human gait is highly individual, but high-quality motion capture is usually limited to expensive, specialized labs. While single-camera pose estimation is accessible, it suffers from depth ambiguity and occlusion, making it unreliable for detailed walking analysis. This project addresses that gap by using low-cost, everyday cameras to capture structured gait information with enough fidelity to support biomechanical interpretation, while also turning that data into expressive visual forms.

### 2) Approach Taken
The system uses two or more consumer cameras observing the same walking space from different viewpoints. Each camera estimates body landmarks independently, then multi-view calibration and triangulation are used to reconstruct 3D motion. From reconstructed landmarks, hip, knee, and ankle angles are computed over time. Those cyclical joint-angle signals are then transformed into "Gait Irises": circular, iris-like visualizations in which gait periodicity, symmetry, and variation are encoded as radius, color, and layered structure.

### 3) Results Achieved
The project produces a complete pipeline from multi-camera capture to real-time tracking and exportable visual artifacts. It supports live skeleton visualization, angle timelines, recording/review windows, and high-resolution polar exports. In demo usage, each walking sequence generates a distinct movement portrait, showing that affordable hardware can produce structured, analyzable gait data while also enabling personalized, aesthetically meaningful representations of human motion.

---

## How to Execute the Code

These instructions are intended to be followed exactly.
Tested workflow target: Windows + Python 3.11.

### 1. Prerequisites
- Python 3.11
- USB/web cameras (2 recommended for 3D)
- `pip` available in terminal

### 2. Open the project
From repository root, go to:
- `Project/`

### 3. Create and activate virtual environment
```powershell
cd Project
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies
```powershell
pip install --upgrade pip
pip install mediapipe opencv-python numpy
```

Dependency note:
- Required external packages for this submitted pipeline are exactly:
  - `mediapipe`
  - `opencv-python`
  - `numpy`
- Standard-library modules (for example `tkinter`, `argparse`, `pathlib`, `json`, `urllib`) are included with Python and do not need separate installation.
- Optional/vestigial scripts in `Vestigial/` reference extra packages (`onnxruntime`, `huggingface_hub`) but those are not required for the submitted main workflow.

### 5. Run camera/intrinsics setup
```powershell
python calibrate_cameras.py
```
What this does:
- Detects connected cameras
- Lets you select active cameras and primary camera
- Records/solves intrinsics
- Saves setup/calibration JSON used by runtime

### 6. Run 3D extrinsics calibration
```powershell
python calibrate_extrinsics_3d.py
```
What this does:
- Solves relative camera geometry for multi-view 3D triangulation

### 7. Run main dashboard
```powershell
python main.py
```

### 8. Runtime controls (core)
- `q` = quit
- `a` = toggle polar strip
- `g` = palette gallery
- `o` = export polar PNG
- `e` = export gallery/viewer
- `d` = dual-camera strip toggle (when 2+ cams)
- `b` = optional baseline override (meters)
- `m` = edit body profile

---

## Advanced Features and Recommended Setup Procedure

This section documents the "fancy" runtime features and the exact recommended setup order.

### A) Recommended setup order (important)
1. Run `python calibrate_cameras.py` (camera selection + intrinsics).
2. Run `python calibrate_extrinsics_3d.py` (multi-view camera geometry).
3. Run `python main.py`.
4. In `main.py`, set optional runtime tuning:
   - `b` to set/clear baseline override in meters.
   - `[` / `]` to shift primary camera skeleton overlay left/right.
   - `;` / `'` to shift secondary camera skeleton overlay left/right.
   - `m` to edit body profile (height and segment lengths).

### B) Baseline distance (`b`) and why to use it
- `b` sets a baseline override (meters) for metric scale alignment.
- This helps align output units with a physically measured camera spacing.
- If left blank, the system uses solved calibration scale from extrinsics.

### C) Body segment lengths (`m`)
- The body profile editor stores subject-specific measurements (height, thigh, shank, etc.).
- These values are resolved and displayed during runtime and are used for profile-aware interpretation/personalization.
- This supports subject-specific reporting and improves consistency of profile metadata across runs.

### D) Overlay alignment controls and why side-shift can appear
- During real use, the drawn skeleton may appear slightly left/right of the body in one camera view due to:
  - residual lens distortion differences,
  - calibration/index remap mismatch,
  - small viewpoint/model projection differences.
- The runtime controls `[` `]` `;` `'` apply display-only x-offset alignment so the overlay visually matches the person on screen.

### E) Why overlay side-shift is not a downstream processing issue
- The offset controls modify only the displayed 2D draw coordinates for the on-screen overlay.
- Core downstream computation (pose estimation, triangulation, angle calculation, polar generation) uses the underlying pose data, not these display offsets.
- Therefore, visual recentering for readability does not corrupt analytical processing.

---

## Folder Structure and Setup Details

```text
Project/
├─ main.py                      # Main dashboard app (live view + angles + exports)
├─ calibrate_cameras.py         # Camera selection + intrinsics workflow
├─ calibrate_extrinsics_3d.py   # Multi-camera extrinsics calibration workflow
├─ camera_config.py             # Camera I/O, calibration I/O, multi-camera reader
├─ triangulation.py             # 3D triangulation + 3D angle computation
├─ pose_processor.py            # 2D landmark processing + angle extraction
├─ body_profile.py              # Body segment profile load/save/resolve
├─ config.py                    # UI, model, smoothing, and behavior constants
├─ ui/                          # Dashboard UI modules (drawing, timeline, galleries)
├─ visualization/               # Polar plot, clip preview, skeleton rendering, 3D view
├─ models/                      # Pose model assets (.task)
├─ camera_calibrations.json     # Saved intrinsics/extrinsics
└─ last_camera_setup.json       # Last selected camera IDs/settings
```

### Required runtime files
- `camera_calibrations.json` and `last_camera_setup.json` are generated/updated by calibration scripts.
- Model asset files are expected under `Project/models/` (or fetched by code if absent, per current implementation).

---

## Operating System, Software, and Dependency Requirements

### OS
- Primary target: Windows 10/11

### Software
- Python 3.11
- Webcam drivers recognized by OpenCV

### Python packages
- `mediapipe`
- `opencv-python`
- `numpy`

### Hardware notes
- 2 cameras recommended for reliable 3D angles.
- Stable USB bandwidth matters (avoid overloaded hubs where possible).

---

## Submission Readiness Notes

- Code should be submitted in runnable state with calibration scripts and dashboard entrypoint.
- Evaluators will execute based on this document; avoid requiring undocumented manual steps.
- If model files are not bundled, ensure network access is available for first-run model fetch (as implemented), or bundle `Project/models` in submission.

---

## GitHub Link (separate file requirement)

Create a separate simple text/markdown file (for example: `github_link.txt` or `github_link.md`) containing only:

```text
https://github.com/<your-username>/<your-repo>
```
