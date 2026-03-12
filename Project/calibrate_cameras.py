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
    ROTATION_OPTIONS,
    Calibration,
    CameraInfo,
    Extrinsics,
    Intrinsics,
    LastCameraSetup,
    apply_rotation,
    detect_connected_cameras,
    is_url_source,
    load_calibrations,
    load_last_setup,
    open_camera,
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

# Detect whether OpenCV highgui is available (namedWindow/imshow).
try:
    _test_win = "_cv2_test_window_"
    cv2.namedWindow(_test_win, cv2.WINDOW_NORMAL)
    cv2.destroyWindow(_test_win)
    HAVE_GUI = True
except cv2.error:
    HAVE_GUI = False


def _ask_url_dialog() -> str:
    """Pop up a GUI dialog asking for a camera URL. Returns the URL or empty string."""
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    url = simpledialog.askstring(
        "Add IP Camera",
        "Enter the middle part of your camera URL.\n\n"
        "It will be wrapped as:\n"
        "  http://<your input>/video\n\n"
        "Example: type  192.168.1.5:8080\n"
        "to get  http://192.168.1.5:8080/video\n\n"
        "Or paste a full URL (http:// or rtsp://) to use as-is.",
        parent=root,
        initialvalue="192.168.",
    )
    root.destroy()
    raw = (url or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("rtsp://"):
        return raw
    if not raw.endswith("/video"):
        raw = f"http://{raw}/video"
    else:
        raw = f"http://{raw}"
    return raw


def _show_error_dialog(msg: str) -> None:
    """Pop up a GUI error message box."""
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showerror("Camera Error", msg, parent=root)
    root.destroy()


def _make_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """3D points of chessboard corners in board frame (Z=0)."""
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def _capture_intrinsics_for_camera(
    cam_id: str,
    cam_label: str,
    board_size: tuple[int, int],
    square_size: float,
) -> Intrinsics | None:
    """
    Open camera (by index or URL), show live feed. User moves chessboard;
    press 'c' to capture a frame, 'd' when done.
    Run calibrateCamera and return Intrinsics or None.
    """
    cap = open_camera(cam_id)
    if not cap.isOpened():
        print(f"  Could not open camera {cam_id}")
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    objp = _make_object_points(board_size[0], board_size[1], square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    if HAVE_GUI:
        cv2.namedWindow(WINDOW_CALIB, cv2.WINDOW_NORMAL)
        print(
            f"  Camera {cam_id} ({cam_label}): move chessboard, "
            f"press 'c' to capture, 'd' when done (need {MIN_CALIB_FRAMES}+ frames)."
        )
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            display = frame.copy()
            if ret:
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                cv2.drawChessboardCorners(display, board_size, corners_refined, ret)
                status = (
                    f"Board found | captured: {len(objpoints)} "
                    "(press 'c' to add, 'd' when done)"
                )
            else:
                status = "No board detected"
            cv2.putText(
                display,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
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
    else:
        print(
            f"  Camera {cam_id} ({cam_label}): OpenCV GUI not available; "
            f"auto-capturing frames when chessboard is detected "
            f"(need at least {MIN_CALIB_FRAMES} valid frames)."
        )
        max_frames = 600
        frames_seen = 0
        while frames_seen < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            if ret:
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                print(f"    Captured frame {len(objpoints)}")
                if len(objpoints) >= max(MIN_CALIB_FRAMES, 30):
                    break
            frames_seen += 1
        cap.release()

    if len(objpoints) < MIN_CALIB_FRAMES:
        print(f"  Not enough frames (got {len(objpoints)}, need {MIN_CALIB_FRAMES}). Skipping camera {cam_id}.")
        return None
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        mean_error += float(np.linalg.norm(imgpoints[i] - imgpts2))
    mean_error /= len(objpoints)
    print(f"  Camera {cam_id}: reprojection error {mean_error:.4f} px")
    return Intrinsics(K=K, dist=dist, image_size=(w, h), reproj_error=mean_error)


def _compute_extrinsics(
    caps: list[tuple[str, cv2.VideoCapture]],
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
    if HAVE_GUI:
        cv2.namedWindow(WINDOW_EXTRINSICS, cv2.WINDOW_NORMAL)
        print("Place the chessboard in the shared capture volume so all selected cameras can see it.")
        print("Press Space to capture one synchronized snapshot from all cameras.")
        captured: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        while True:
            frames: list[tuple[str, np.ndarray, np.ndarray] | None] = []
            for cam_id, cap in caps:
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
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                    )
                    frames.append((cam_id, frame, corners_refined))
                else:
                    frames.append(None)
            # Show first camera's view
            if frames and frames[0] is not None:
                _, frame, _ = frames[0]
                disp = cv2.resize(frame, (CALIB_PREVIEW_W, CALIB_PREVIEW_H))
                cv2.putText(
                    disp,
                    "Space = capture snapshot for extrinsics",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
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
    else:
        print("OpenCV GUI not available; capturing frames automatically for extrinsics.")
        print("Place the chessboard where at least two cameras can see it and hold still.")
        max_iters = 600
        for _ in range(max_iters):
            frames: list[tuple[str, np.ndarray, np.ndarray] | None] = []
            for cam_id, cap in caps:
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
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                    )
                    frames.append((cam_id, frame, corners_refined))
                else:
                    frames.append(None)
            valid = [f for f in frames if f is not None]
            if len(valid) < 2:
                continue
            captured: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
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
            for cam_id, R_cam, t_cam, _w, _h in captured:
                cal = calibrations[cam_id]
                cal.extrinsics = Extrinsics(R=R_cam, t=t_cam)
            print("Extrinsics computed and saved (headless mode).")
            break


def _open_caps_for_ids(camera_ids: list[str]) -> list[tuple[str, cv2.VideoCapture]]:
    """Open VideoCapture for each camera_id (index or URL). Returns list of (cam_id, cap)."""
    out: list[tuple[str, cv2.VideoCapture]] = []
    for cid in camera_ids:
        cap = open_camera(cid)
        if cap.isOpened():
            out.append((cid, cap))
        else:
            cap.release()
    return out


def _short_label(cam_id: str) -> str:
    """Readable short label for a camera ID."""
    if is_url_source(cam_id):
        return cam_id.split("//")[-1][:30]
    return f"Cam {cam_id}"


def run_camera_selection(
    cameras: list[CameraInfo],
    last_setup: LastCameraSetup | None,
) -> tuple[list[str], str, dict[str, int]]:
    """
    GUI camera selection with live previews.  Supports both local index
    cameras and IP/URL cameras.

    Controls shown on screen:
      0-9   toggle a local camera
      I     add an IP / URL camera
      R     cycle rotation for the focused (last-toggled or primary) camera
      P     cycle primary among selected cameras
      Enter confirm and proceed

    Returns (selected_camera_ids, primary_camera_id, camera_rotations).
    """
    all_ids: list[str] = [str(c.index) for c in cameras]
    if last_setup:
        for cid in last_setup.selected_camera_ids:
            if cid not in all_ids:
                all_ids.append(cid)

    selected: set[str] = set()
    if last_setup and last_setup.selected_camera_ids:
        for cid in last_setup.selected_camera_ids:
            selected.add(cid)
    if not selected and all_ids:
        selected.add(all_ids[0])

    primary = all_ids[0] if all_ids else "0"
    if last_setup and last_setup.primary_camera_id:
        primary = last_setup.primary_camera_id

    rotations: dict[str, int] = {}
    if last_setup and last_setup.camera_rotations:
        rotations = dict(last_setup.camera_rotations)

    # Which camera the R key applies to (last toggled, or primary)
    focused_cam = primary

    caps: list[tuple[str, cv2.VideoCapture]] = []
    for cid in all_ids:
        cap = open_camera(cid, 640, 480)
        if cap.isOpened():
            caps.append((cid, cap))

    print("Camera selection:")
    print("  0-9  toggle local camera")
    print("  I    add IP/URL camera")
    print("  R    rotate focused camera (cycles 0/90/180/270)")
    print("  P    cycle primary")
    print("  Enter  confirm")
    while True:
        for cid, cap in caps:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            rot = rotations.get(cid, 0)
            frame = apply_rotation(frame, rot)
            disp = frame.copy()
            sel = "SELECTED" if cid in selected else "off"
            prim = " [PRIMARY]" if cid == primary else ""
            focus = " *" if cid == focused_cam else ""
            rot_txt = f" rot={rot}" if rot else ""
            label = _short_label(cid)
            cv2.putText(disp, f"{label} {sel}{prim}{rot_txt}{focus}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, "0-9=toggle | I=IP | R=rotate | P=primary | Enter=done",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
            win_name = f"Cam: {label}"
            cv2.imshow(win_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:  # Enter
            break
        if key in (ord("i"), ord("I")):
            url = _ask_url_dialog()
            if url:
                test_cap = open_camera(url, 640, 480)
                if test_cap.isOpened():
                    ok, frame = test_cap.read()
                    if ok and frame is not None and frame.mean() > 2.0:
                        all_ids.append(url)
                        selected.add(url)
                        caps.append((url, test_cap))
                        print(f"  Added and selected: {url}")
                    else:
                        test_cap.release()
                        _show_error_dialog(f"Connected but got no usable frames from:\n{url}")
                else:
                    test_cap.release()
                    _show_error_dialog(f"Could not open camera at:\n{url}\n\nCheck the URL format, e.g.:\n  http://192.168.1.5:8080/video")
        if key in (ord("r"), ord("R")):
            cur = rotations.get(focused_cam, 0)
            idx = ROTATION_OPTIONS.index(cur) if cur in ROTATION_OPTIONS else 0
            new_rot = ROTATION_OPTIONS[(idx + 1) % len(ROTATION_OPTIONS)]
            rotations[focused_cam] = new_rot
            print(f"  {_short_label(focused_cam)} rotation: {new_rot} deg")
        if key in (ord("p"), ord("P")) and selected:
            sel_list = sorted(selected)
            try:
                idx = sel_list.index(primary)
                primary = sel_list[(idx + 1) % len(sel_list)]
            except ValueError:
                primary = sel_list[0]
            focused_cam = primary
            print(f"  Primary: {primary}")
        if ord("0") <= key <= ord("9"):
            k = str(key - ord("0"))
            if k in all_ids:
                if k in selected:
                    selected.discard(k)
                else:
                    selected.add(k)
                focused_cam = k
                if primary not in selected and selected:
                    primary = sorted(selected)[0]
                print(f"  Selected: {sorted(selected)}, primary: {primary}")
        if key == ord("q"):
            selected = set()
            break

    for cid, cap in caps:
        cap.release()
        cv2.destroyWindow(f"Cam: {_short_label(cid)}")

    return sorted(selected), primary, rotations


def main() -> None:
    print("Multi-camera calibration and setup")
    print("----------------------------------")
    cameras = detect_connected_cameras()
    if not cameras:
        print("No local cameras detected (you can still add IP cameras).")
    else:
        print(f"Found {len(cameras)} local camera(s): indices {[c.index for c in cameras]}")

    last_setup = load_last_setup()
    selected_ids, primary_id, cam_rotations = run_camera_selection(cameras, last_setup)
    if not selected_ids:
        print("No cameras selected. Exiting.")
        return

    calibrations = load_calibrations()
    board_size = CHESSBOARD_INNER_CORNERS
    square_size = CHESSBOARD_SQUARE_SIZE_MM

    for cam_id in selected_ids:
        if cam_id in calibrations:
            print(f"Camera {cam_id} already calibrated. Skip (delete {CALIBRATIONS_PATH} entry to redo).")
            continue
        label = _short_label(cam_id)
        intr = _capture_intrinsics_for_camera(cam_id, label, board_size, square_size)
        if intr is not None:
            calibrations[cam_id] = Calibration(
                camera_id=cam_id, intrinsics=intr, extrinsics=None,
                metadata={"label": label},
            )
            save_calibrations(calibrations)

    calibrations = load_calibrations()
    selected_calibrated = [cid for cid in selected_ids if cid in calibrations]
    if not selected_calibrated:
        print("No cameras were successfully calibrated.")
        return

    caps = _open_caps_for_ids(selected_calibrated)
    if len(caps) >= 2:
        do_ext = input("Compute extrinsics (camera positions in world)? [y/N]: ").strip().lower()
        if do_ext == "y":
            _compute_extrinsics(caps, calibrations, board_size, square_size)
            save_calibrations(calibrations)
    for _, cap in caps:
        cap.release()

    setup = LastCameraSetup(
        selected_camera_ids=selected_calibrated,
        primary_camera_id=primary_id,
        use_triangulation=True,
        camera_rotations=cam_rotations,
    )
    save_last_setup(setup)
    print(f"Setup saved: cameras {selected_calibrated}, primary {primary_id}")
    print("Run main.py to use these cameras.")


if __name__ == "__main__":
    main()
