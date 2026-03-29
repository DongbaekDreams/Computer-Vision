"""
Multi-camera extrinsic (3D) calibration for the pose dashboard.

Run after calibrate_cameras.py has stored intrinsics for each camera.
Places all cameras in a common world frame (chessboard) so main.py can triangulate.

Usage:
  python calibrate_extrinsics_3d.py
  python calibrate_extrinsics_3d.py --cameras 1,5

Default camera set: intersection of last_camera_setup.json selected IDs with
cameras that have intrinsics in camera_calibrations.json (need at least two).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from calibrate_cameras import (
    CALIB_CAPTURE_H_DUAL,
    CALIB_CAPTURE_W_DUAL,
    CHESSBOARD_INNER_CORNERS,
    CHESSBOARD_SQUARE_SIZE_MM,
    HAVE_GUI,
    _chessboard_sizes_to_try,
    _find_chessboard_corners_robust,
    _letterbox_bgr,
    _make_object_points,
)
from camera_config import (
    Calibration,
    Extrinsics,
    MultiCameraReader,
    load_calibrations,
    load_last_setup,
    save_calibrations,
)

WINDOW_EXTRINSICS = "Extrinsics - place board in view, press Space to capture"
EXTRINSICS_TILE_W = 480
EXTRINSICS_TILE_H = 360


def _detect_board_best(
    gray: np.ndarray, preferred: tuple[int, int]
) -> tuple[bool, tuple[int, int] | None, np.ndarray | None]:
    """
    Same strategy as intrinsics-from-video: downscaled detect + subpixel on full-res,
    try CHESSBOARD_INNER_CORNERS then fallback sizes (SB + CLAHE inside robust).
    """
    for sz in _chessboard_sizes_to_try(preferred):
        ok, corners = _find_chessboard_corners_robust(gray, sz)
        if ok and corners is not None:
            return True, sz, corners
    return False, None, None


def _build_extrinsics_mosaic(
    frame_infos: list[
        tuple[str, np.ndarray | None, np.ndarray | None, bool, tuple[int, int] | None]
    ],
) -> np.ndarray:
    """Side-by-side live view for extrinsics; every camera gets a tile (even if no board)."""
    tiles: list[np.ndarray] = []
    gap = 8
    for cam_id, frame, corners, sees_board, det_size in frame_infos:
        if frame is None:
            tile = np.zeros((EXTRINSICS_TILE_H, EXTRINSICS_TILE_W, 3), dtype=np.uint8)
            cv2.putText(
                tile,
                f"Cam {cam_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                "no signal",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (100, 100, 240),
                2,
                cv2.LINE_AA,
            )
        else:
            vis = frame.copy()
            if sees_board and corners is not None and det_size is not None:
                cv2.drawChessboardCorners(vis, det_size, corners, True)
            tile = _letterbox_bgr(vis, EXTRINSICS_TILE_W, EXTRINSICS_TILE_H)
            st = "board OK" if sees_board else "no board"
            col = (80, 220, 100) if sees_board else (120, 120, 255)
            cv2.putText(
                tile,
                f"Cam {cam_id} — {st}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                col,
                2,
                cv2.LINE_AA,
            )
        tiles.append(tile)

    if not tiles:
        return np.zeros((400, 640, 3), dtype=np.uint8)

    row = tiles[0]
    for t in tiles[1:]:
        g = np.zeros((row.shape[0], gap, 3), dtype=np.uint8)
        g[:] = (30, 34, 42)
        row = np.hstack([row, g, t])

    banner_h = 88
    out = np.zeros((row.shape[0] + banner_h, row.shape[1], 3), dtype=np.uint8)
    out[:] = (14, 18, 26)
    out[banner_h:, :] = row
    cv2.putText(
        out,
        "Extrinsics — Space = capture | Q = skip",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (242, 242, 242),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        'Place the board so 2+ cameras show "board OK", then press Space.',
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (188, 194, 206),
        1,
        cv2.LINE_AA,
    )
    return out


def _batch_to_frame_infos(
    batch: dict[str, np.ndarray],
    camera_ids: list[str],
    calibrations: dict[str, Calibration],
    preferred_board: tuple[int, int],
) -> list[tuple[str, np.ndarray | None, np.ndarray | None, bool, tuple[int, int] | None]]:
    out: list[tuple[str, np.ndarray | None, np.ndarray | None, bool, tuple[int, int] | None]] = []
    for cam_id in camera_ids:
        frame = batch.get(cam_id)
        if frame is None:
            out.append((cam_id, None, None, False, None))
            continue
        if calibrations.get(cam_id) is None:
            out.append((cam_id, frame, None, False, None))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, sz, corners = _detect_board_best(gray, preferred_board)
        out.append((cam_id, frame, corners, ok, sz))
    return out


def _compute_extrinsics(
    reader: MultiCameraReader,
    camera_ids: list[str],
    calibrations: dict[str, Calibration],
    preferred_board: tuple[int, int],
    square_size: float,
) -> None:
    """
    Capture one frame from each camera with the board visible. Use first camera's
    board pose as world frame; compute each camera's R, t in that world.
    """
    if HAVE_GUI:
        cv2.namedWindow(WINDOW_EXTRINSICS, cv2.WINDOW_NORMAL)
        print("Place the chessboard in the shared capture volume so all selected cameras can see it.")
        print("Press Space to capture one synchronized snapshot from all cameras.")
        print(
            "Detection uses the same multi-size search as intrinsics calibration; "
            "if the board never turns green, set CHESSBOARD_INNER_CORNERS in calibrate_cameras.py."
        )
        while True:
            batch = reader.read_batch()
            frame_infos = _batch_to_frame_infos(
                batch, camera_ids, calibrations, preferred_board
            )

            mosaic = _build_extrinsics_mosaic(frame_infos)
            mw, mh = mosaic.shape[1], mosaic.shape[0]
            max_disp_w, max_disp_h = 1920, 1080
            scale = min(max_disp_w / float(mw), max_disp_h / float(mh), 1.0)
            if scale < 1.0:
                dw = max(1, int(round(mw * scale)))
                dh = max(1, int(round(mh * scale)))
                display = cv2.resize(mosaic, (dw, dh), interpolation=cv2.INTER_AREA)
            else:
                display = mosaic
            cv2.imshow(WINDOW_EXTRINSICS, display)
            cv2.resizeWindow(WINDOW_EXTRINSICS, display.shape[1], display.shape[0])

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                valid_snap: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int]]] = []
                for cam_id in camera_ids:
                    frame = batch.get(cam_id)
                    if frame is None or calibrations.get(cam_id) is None:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ok_cb, sz, corners_refined = _detect_board_best(gray, preferred_board)
                    if ok_cb and corners_refined is not None and sz is not None:
                        valid_snap.append((cam_id, frame, corners_refined, sz))
                if len(valid_snap) < 2:
                    print("Need at least 2 cameras to see the board. Move board and try again.")
                    continue
                sizes = {t[3] for t in valid_snap}
                if len(sizes) != 1:
                    print(
                        "All cameras must see the same physical board (same inner corner grid). "
                        f"Detected grids (cols, rows): {sizes}"
                    )
                    continue
                sz_one = valid_snap[0][3]
                objp = _make_object_points(sz_one[0], sz_one[1], square_size)
                captured = []
                for cam_id, frame, corners_refined, _sz in valid_snap:
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
                print(f"Extrinsics computed (board inner corners {sz_one[0]}x{sz_one[1]}).")
                break
            if key == ord("q"):
                break
        cv2.destroyWindow(WINDOW_EXTRINSICS)
    else:
        print("OpenCV GUI not available; capturing frames automatically for extrinsics.")
        print("Place the chessboard where at least two cameras can see it and hold still.")
        max_iters = 600
        for _ in range(max_iters):
            batch = reader.read_batch()
            row: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int]]] = []
            for cam_id in camera_ids:
                frame = batch.get(cam_id)
                if frame is None or calibrations.get(cam_id) is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok_cb, sz, corners_refined = _detect_board_best(gray, preferred_board)
                if ok_cb and corners_refined is not None and sz is not None:
                    row.append((cam_id, frame, corners_refined, sz))
            if len(row) < 2:
                continue
            sizes = {t[3] for t in row}
            if len(sizes) != 1:
                continue
            sz_one = row[0][3]
            objp = _make_object_points(sz_one[0], sz_one[1], square_size)
            captured: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            for cam_id, frame, corners_refined, _sz in row:
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
            print(f"Extrinsics computed and saved (headless mode), board {sz_one[0]}x{sz_one[1]}.")
            break


def _parse_camera_list(s: str) -> list[str]:
    parts = [p.strip() for p in s.replace(" ", "").split(",") if p.strip()]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrinsic (3D) calibration for multi-camera triangulation.")
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='Comma-separated camera indices, e.g. "1,5". Default: from last_camera_setup.json.',
    )
    args = parser.parse_args()

    calibrations = load_calibrations()
    if not calibrations:
        print("No camera_calibrations.json entries. Run calibrate_cameras.py first.")
        return

    if args.cameras:
        candidate_ids = _parse_camera_list(args.cameras)
    else:
        last_setup = load_last_setup()
        if last_setup is None or not last_setup.selected_camera_ids:
            print(
                "No --cameras given and no last_camera_setup.json rig. "
                'Use: python calibrate_extrinsics_3d.py --cameras 1,5'
            )
            return
        candidate_ids = list(last_setup.selected_camera_ids)

    ids_with_intrinsics = [cid for cid in candidate_ids if cid in calibrations]
    if len(ids_with_intrinsics) < 2:
        print(
            f"Need at least two cameras with intrinsics in calibrations. "
            f"Got {ids_with_intrinsics} from {candidate_ids}. "
            "Run calibrate_cameras.py for each index, or fix --cameras / last_camera_setup.json."
        )
        return

    board_size = CHESSBOARD_INNER_CORNERS
    square_size = CHESSBOARD_SQUARE_SIZE_MM
    ex_w, ex_h = CALIB_CAPTURE_W_DUAL, CALIB_CAPTURE_H_DUAL

    print(f"3D extrinsics calibration for cameras: {ids_with_intrinsics} at {ex_w}x{ex_h}")
    try:
        reader = MultiCameraReader.from_camera_ids(
            list(ids_with_intrinsics),
            ex_w,
            ex_h,
            read_order=list(ids_with_intrinsics),
        )
    except RuntimeError as e:
        print(e)
        return

    try:
        _compute_extrinsics(reader, ids_with_intrinsics, calibrations, board_size, square_size)
        save_calibrations(calibrations)
        print("Wrote camera_calibrations.json (extrinsics updated where captured).")
    finally:
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
