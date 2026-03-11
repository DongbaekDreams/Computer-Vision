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

2. In the calibration script:
   - Select which connected cameras to use (number keys to toggle, P to set primary, Enter to confirm).
   - For each selected camera that is not yet calibrated:
     - Point the camera at the chessboard and press **c** to capture calibration frames while moving the board (different angles and distances). Press **d** when done (at least 15 frames with the board detected).
   - Optionally compute **extrinsics** (camera positions in a common world frame): place the chessboard so at least two cameras see it, then press **Space** to capture a snapshot. This is required for 3D triangulation.
   - The script saves calibrations to `Project/camera_calibrations.json` and the last-used camera selection to `Project/last_camera_setup.json`.

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
- **Extrinsics only** (cameras moved): Re-run the calibration script, skip re-doing intrinsics for cameras that are already calibrated, and run the extrinsics step again (chessboard visible in 2+ cameras, press Space). Save the setup as usual.

## Files

- `camera_config.py` – Load/save calibrations and last setup; detect connected cameras; projection matrix helpers.
- `calibrate_cameras.py` – Interactive calibration (intrinsics from chessboard video, optional extrinsics) and saving last-used cameras.
- `triangulation.py` – Multi-view triangulation and 3D joint angle computation.
- `camera_calibrations.json` – Stored per-camera intrinsics and optional extrinsics (created by the calibration script).
- `last_camera_setup.json` – Last selected camera IDs and primary camera (written by the calibration script and by main after you confirm selection).

## Behaviour summary

| Cameras          | Calibration              | Behaviour                          |
|------------------|--------------------------|------------------------------------|
| 1                | Any                      | 2D angles from that camera        |
| 2+               | Intrinsics only          | 2D angles from primary camera    |
| 2+               | Intrinsics + extrinsics  | 3D triangulation → 3D joint angles |

The video background and 2D skeleton overlay always use the **primary** camera’s feed.
