# Multi-camera pose dashboard

The dashboard supports 1+ cameras. With two or more **calibrated** cameras (including extrinsics), joint angles are computed in 3D via triangulation for better accuracy.

## Workflow

### First-time setup

1. **Calibrate cameras**  
   From the project root (or from the `Project` directory):
   ```bash
   python -m Project.calibrate_cameras
   ```
   or, from inside `Project`:
   ```bash
   python calibrate_cameras.py
   ```

2. In `calibrate_cameras.py`:
   - Select which connected cameras to use (number keys to toggle, P to set primary, Enter to confirm).
   - For each selected camera that is not yet calibrated: record a clip with **Space** (start/stop) while moving the chessboard; intrinsics are solved from the saved videos (see on-screen instructions).
   - The script saves calibrations to `Project/camera_calibrations.json` and the last-used camera selection to `Project/last_camera_setup.json`.

3. **3D extrinsics** (only if you want triangulation with 2+ cameras): after every camera has intrinsics, run:
   ```bash
   python calibrate_extrinsics_3d.py
   ```
   or `python calibrate_extrinsics_3d.py --cameras 1,5`. Place the board so at least two cameras show “board OK”, then press **Space** to capture. Uses the same chessboard constants as `calibrate_cameras.py`.

### Normal use

1. **Run the dashboard**  
   From the project root or from `Project`:
   ```bash
   python -m Project.main
   ```
   or:
   ```bash
   cd Project && python main.py
   ```

2. On startup, if any **calibrated** cameras are connected:
   - A camera selection screen appears: you see which cameras are connected and calibrated.
   - Toggle cameras with number keys (0–9), set primary with **P**, confirm with **Enter**.
   - Your choice is saved as the default for next time.

3. If no calibration exists (first run) or no calibrated cameras are connected:
   - The app falls back to a single camera (index 0) and uses 2D angles only.

### Recalibration

- **Intrinsics** (lens/distortion): Re-run `calibrate_cameras.py` and calibrate the camera again; or delete that camera’s entry from `camera_calibrations.json` and run calibration for that camera.
- **Extrinsics only** (cameras moved): Run `calibrate_extrinsics_3d.py` again with the same rig (or `--cameras ...`); capture a new board snapshot with **Space**.

## Files

- `camera_config.py` – Load/save calibrations and last setup; detect connected cameras; projection matrix helpers.
- `calibrate_cameras.py` – Camera selection, intrinsics from recorded chessboard video, and saving last-used cameras.
- `calibrate_extrinsics_3d.py` – Optional multi-camera extrinsics (chessboard snapshot) for 3D triangulation.
- `triangulation.py` – Multi-view triangulation and 3D joint angle computation.
- `camera_calibrations.json` – Stored per-camera intrinsics and optional extrinsics (written by the calibration scripts).
- `last_camera_setup.json` – Last selected camera IDs and primary camera (written by the calibration script and by main after you confirm selection).

## Behaviour summary

| Cameras          | Calibration              | Behaviour                          |
|------------------|--------------------------|------------------------------------|
| 1                | Any                      | 2D angles from that camera        |
| 2+               | Intrinsics only          | 2D angles from primary camera    |
| 2+               | Intrinsics + extrinsics  | 3D triangulation → 3D joint angles |

The video background and 2D skeleton overlay always use the **primary** camera’s feed.
