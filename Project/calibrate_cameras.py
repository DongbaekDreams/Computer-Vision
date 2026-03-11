"""
Camera calibration and setup for multi-camera pose dashboard.

Run this script to:
1. Detect connected cameras and select which to use
2. Calibrate intrinsics per camera from a short chessboard video
3. Optionally compute extrinsics (camera poses in a common world frame)
4. Save calibrations and last-used camera selection for main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure Project directory is on path for imports
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from camera_config import (
    CALIBRATIONS_PATH,
    LAST_SETUP_PATH,
    Calibration,
    CameraInfo,
    Extrinsics,
    Intrinsics,
    LastCameraSetup,
    detect_connected_cameras,
    load_calibrations,
    load_last_setup,
    save_calibrations,
    save_last_setup,
)

# Chessboard: inner corners (not squares). Adjust to match your printed pattern.
CHESSBOARD_INNER_CORNERS = (9, 6)  # (cols, rows)
CHESSBOARD_SQUARE_SIZE_MM = 25.0  # used only for 3D object points scale (arbitrary units)
CALIB_PREVIEW_W = 640
CALIB_PREVIEW_H = 480
MIN_CALIB_FRAMES = 15
WINDOW_CALIB = "Calibration - press 'c' to capture frame, 'd' when done"
WINDOW_SELECT = "Camera selection - number keys toggle, Enter to confirm"
WINDOW_EXTRINSICS = "Extrinsics - place board in view, press Space to capture"


def _make_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """3D points of chessboard corners in board frame (Z=0)."""
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def _capture_intrinsics_for_camera(
    cam_index: int,
    cam_label: str,
    board_size: tuple[int, int],
    square_size: float,
) -> Intrinsics | None:
    """
    Open camera, show live feed. User moves chessboard; press 'c' to capture
    a frame, 'd' when done. Run calibrateCamera and return Intrinsics or None.
    """
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend)
    if not cap.isOpened():
        print(f"  Could not open camera {cam_index}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    objp = _make_object_points(board_size[0], board_size[1], square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    cv2.namedWindow(WINDOW_CALIB, cv2.WINDOW_NORMAL)
    print(f"  Camera {cam_index} ({cam_label}): move chessboard, press 'c' to capture, 'd' when done (need {MIN_CALIB_FRAMES}+ frames).")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        display = frame.copy()
        if ret:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            cv2.drawChessboardCorners(display, board_size, corners_refined, ret)
            status = f"Board found | captured: {len(objpoints)} (press 'c' to add, 'd' when done)"
        else:
            status = "No board detected"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(WINDOW_CALIB, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            break
        if key == ord("c") and ret:
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print(f"    Captured frame {len(objpoints)}")

    cap.release()
    cv2.destroyWindow(WINDOW_CALIB)
    if len(objpoints) < MIN_CALIB_FRAMES:
        print(f"  Not enough frames (got {len(objpoints)}, need {MIN_CALIB_FRAMES}). Skipping camera {cam_index}.")
        return None
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        mean_error += float(np.linalg.norm(imgpoints[i] - imgpts2))
    mean_error /= len(objpoints)
    print(f"  Camera {cam_index}: reprojection error {mean_error:.4f} px")
    return Intrinsics(K=K, dist=dist, image_size=(w, h), reproj_error=mean_error)


def _compute_extrinsics(
    caps: list[tuple[int, cv2.VideoCapture]],
    calibrations: dict[str, Calibration],
    board_size: tuple[int, int],
    square_size: float,
) -> None:
    """
    Capture one frame from each camera with the board visible. Use first camera's
    board pose as world frame; compute each camera's R, t in that world.
    """
    objp = _make_object_points(board_size[0], board_size[1], square_size)
    # Collect (camera_id, frame, gray, corners) for each cam that sees the board
    cv2.namedWindow(WINDOW_EXTRINSICS, cv2.WINDOW_NORMAL)
    print("Place the chessboard in the shared capture volume so all selected cameras can see it.")
    print("Press Space to capture one synchronized snapshot from all cameras.")
    captured: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    while True:
        frames: list[tuple[str, np.ndarray, np.ndarray] | None] = []
        for cam_index, cap in caps:
            cam_id = str(cam_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                frames.append(None)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cal = calibrations.get(cam_id)
            if cal is None:
                frames.append(None)
                continue
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            if ret:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                frames.append((cam_id, frame, corners_refined))
            else:
                frames.append(None)
        # Show first camera's view
        if frames and frames[0] is not None:
            _, frame, _ = frames[0]
            disp = cv2.resize(frame, (CALIB_PREVIEW_W, CALIB_PREVIEW_H))
            cv2.putText(disp, "Space = capture snapshot for extrinsics", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(WINDOW_EXTRINSICS, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            valid = [f for f in frames if f is not None]
            if len(valid) < 2:
                print("Need at least 2 cameras to see the board. Move board and try again.")
                continue
            captured = []
            for cam_id, frame, corners_refined in valid:
                cal = calibrations[cam_id]
                K = cal.intrinsics.K
                dist = cal.intrinsics.dist
                ret, rvec, tvec = cv2.solvePnP(objp, corners_refined, K, dist)
                if not ret:
                    continue
                R, _ = cv2.Rodrigues(rvec)
                captured.append((cam_id, R, tvec.ravel(), frame.shape[1], frame.shape[0]))
            if len(captured) < 2:
                continue
            # World frame = board frame as seen by first camera in list
            cam0_id, R0, t0, w0, h0 = captured[0]
            # R0, t0: from board to camera0. So board origin in camera0 is -R0.T @ t0.
            # World = board. Camera0 pose in world: R_world_to_cam0 = R0, t_world_to_cam0 = t0 (board points in cam0 = R0 @ world + t0).
            # We want: world point P_w. In cam0: P_c0 = R0 @ P_w + t0 => P_w = R0.T @ (P_c0 - t0).
            # So extrinsics for cam0 in world: we store R, t such that P_cam = R @ P_w + t. So R = R0, t = t0 for cam0.
            for cam_id, R_cam, t_cam, _w, _h in captured:
                cal = calibrations[cam_id]
                cal.extrinsics = Extrinsics(R=R_cam, t=t_cam)
            print("Extrinsics computed and saved.")
            break
        if key == ord("q"):
            break
    cv2.destroyWindow(WINDOW_EXTRINSICS)


def _open_caps_for_indices(indices: list[int]) -> list[tuple[int, cv2.VideoCapture]]:
    """Open VideoCapture for each index. Returns list of (index, cap)."""
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    out: list[tuple[int, cv2.VideoCapture]] = []
    for i in indices:
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            out.append((i, cap))
        else:
            cap.release()
    return out


def run_camera_selection(
    cameras: list[CameraInfo],
    last_setup: LastCameraSetup | None,
) -> tuple[list[int], int]:
    """
    Show preview of each camera; user toggles selection with number keys,
    Enter to confirm. Returns (list of selected indices, primary index).
    """
    selected: set[int] = set()
    if last_setup and last_setup.selected_camera_ids:
        for cid in last_setup.selected_camera_ids:
            try:
                selected.add(int(cid))
            except ValueError:
                pass
    if not selected and cameras:
        selected.add(cameras[0].index)
    primary = cameras[0].index if cameras else 0
    if last_setup and last_setup.primary_camera_id:
        try:
            primary = int(last_setup.primary_camera_id)
        except ValueError:
            pass

    caps: list[tuple[int, cv2.VideoCapture]] = []
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    for info in cameras:
        cap = cv2.VideoCapture(info.index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            caps.append((info.index, cap))

    cv2.namedWindow(WINDOW_SELECT, cv2.WINDOW_NORMAL)
    print("Number keys 0-9: toggle camera. P: set primary. Enter: confirm.")
    while True:
        for idx, cap in caps:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            disp = frame.copy()
            sel = "SELECTED" if idx in selected else "off"
            prim = " [PRIMARY]" if idx == primary else ""
            cv2.putText(disp, f"Cam {idx} {sel}{prim}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, f"Key {idx} toggle | P=primary | Enter=done", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(f"Cam {idx}", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:  # Enter
            break
        if key == ord("p") and selected:
            primary = min(selected)
            print(f"Primary set to camera {primary}")
        if ord("0") <= key <= ord("9"):
            k = key - ord("0")
            if k in [c.index for c in cameras]:
                if k in selected:
                    selected.discard(k)
                else:
                    selected.add(k)
                if primary not in selected and selected:
                    primary = min(selected)
                print(f"Selected: {sorted(selected)}, primary: {primary}")
        if key == ord("q"):
            selected = set()
            break
    for _, cap in caps:
        cap.release()
    for info in cameras:
        cv2.destroyWindow(f"Cam {info.index}")
    cv2.destroyWindow(WINDOW_SELECT)
    return sorted(selected), primary


def main() -> None:
    print("Multi-camera calibration and setup")
    print("----------------------------------")
    cameras = detect_connected_cameras()
    if not cameras:
        print("No cameras detected.")
        return
    print(f"Found {len(cameras)} camera(s): indices {[c.index for c in cameras]}")

    last_setup = load_last_setup()
    selected_indices, primary_index = run_camera_selection(cameras, last_setup)
    if not selected_indices:
        print("No cameras selected. Exiting.")
        return

    calibrations = load_calibrations()
    board_size = CHESSBOARD_INNER_CORNERS
    square_size = CHESSBOARD_SQUARE_SIZE_MM

    # Intrinsic calibration for any selected camera that doesn't have calibration yet
    for idx in selected_indices:
        cam_id = str(idx)
        if cam_id in calibrations:
            print(f"Camera {idx} already calibrated. Skip (delete {CALIBRATIONS_PATH} entry to redo).")
            continue
        label = next((c.label or f"Cam{idx}" for c in cameras if c.index == idx), f"Cam{idx}")
        intr = _capture_intrinsics_for_camera(idx, label, board_size, square_size)
        if intr is not None:
            calibrations[cam_id] = Calibration(camera_id=cam_id, intrinsics=intr, extrinsics=None, metadata={"label": label})
            save_calibrations(calibrations)

    # Reload in case we added new entries
    calibrations = load_calibrations()
    selected_calibrated = [str(i) for i in selected_indices if str(i) in calibrations]
    if not selected_calibrated:
        print("No cameras were successfully calibrated.")
        return

    # Extrinsics: open all selected calibrated cams, one snapshot with board visible
    caps = _open_caps_for_indices([int(cid) for cid in selected_calibrated])
    if len(caps) >= 2:
        do_ext = input("Compute extrinsics (camera positions in world)? [y/N]: ").strip().lower()
        if do_ext == "y":
            _compute_extrinsics(caps, calibrations, board_size, square_size)
            save_calibrations(calibrations)
    for _, cap in caps:
        cap.release()

    # Save last setup for main.py
    setup = LastCameraSetup(
        selected_camera_ids=selected_calibrated,
        primary_camera_id=str(primary_index),
        use_triangulation=True,
    )
    save_last_setup(setup)
    print(f"Setup saved: cameras {selected_calibrated}, primary {primary_index}")
    print("Run main.py to use these cameras.")


if __name__ == "__main__":
    main()
