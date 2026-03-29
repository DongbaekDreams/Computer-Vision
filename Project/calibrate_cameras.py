"""
Camera calibration and setup for multi-camera pose dashboard.

Run this script to:
1. Detect connected cameras and select which to use
2. Record synchronized calibration clips from all selected cameras, then solve intrinsics from those videos
3. Save calibrations and last-used camera selection for main.py

For multi-camera 3D triangulation, run calibrate_extrinsics_3d.py after intrinsics exist.
"""

from __future__ import annotations

import math
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
    Intrinsics,
    LastCameraSetup,
    ThreadedCapture,
    apply_rotation,
    detect_connected_cameras,
    is_inactive_virtual_cam_frame,
    load_calibrations,
    INTRINSICS_OPEN_BACKENDS,
    load_last_setup,
    open_camera,
    save_calibrations,
    save_last_setup,
)

# Chessboard: INNER corner counts (where black meets white), not square counts.
# A board often sold as "9x6 squares" is (8,5) inner corners; OpenCV needs inner counts.
CHESSBOARD_INNER_CORNERS = (9, 6)  # (cols, rows); use (7, 7) for a standard 8x8 chessboard
CHESSBOARD_SQUARE_SIZE_MM = 25.0  # used only for 3D object points scale (arbitrary units)
# If your print uses different grid, detection still tries these (unique) after the default:
_CHESSBOARD_FALLBACK_SIZES = (
    (8, 5),
    (7, 7),  # standard 8x8 chessboard → 7x7 inner corners
    (10, 7),
    (9, 7),
    (8, 6),
    (7, 5),
    (6, 4),
)
# Selection / intrinsics preview cells are 4:3; frames are letterboxed inside (no stretch).
SELECT_TILE_W = 640
SELECT_TILE_H = 480
SELECT_GAP = 12
CALIB_TILE_MAX_W = 960
CALIB_TILE_MAX_H = 720
# Intrinsics mosaic only: downscale before letterbox (recording uses full-res from ThreadedCapture).
CALIB_INTRINSICS_PREVIEW_MAX_SRC = 640
MIN_CALIB_FRAMES = 15
# Intrinsics: record all cameras at once, then calibrate from saved videos (see _run_intrinsics_record_session).
# Tag ~real USB rate so playback is not sped up; detection uses every decoded frame regardless.
CALIB_RECORD_FPS = 12.0
CALIB_RECORD_MAX_S = 120.0
CALIB_WARMUP_READS = 12
CALIB_VIDEO_DETECT_MAX_SIDE = 720
CALIB_VIDEO_TALLY_FRAME_STRIDE = 3
CALIB_VIDEO_COLLECT_STRIDE = 2
CALIB_VIDEO_MAX_BOARD_FRAMES = 100
CALIB_VIDEO_PROGRESS_EVERY_FRAMES = 200
# Request 4:3 from the driver (common for USB webcams); letterbox still corrects mismatch.
CALIB_CAPTURE_W = 1280
CALIB_CAPTURE_H = 960
CALIB_OPEN_STAGGER_S = 0.25
# Multi-cam: 720p target avoids many drivers falling to ~800x600 YUY2 under USB load.
# Raise to 1920x1080 if you have separate root ports / bandwidth.
CALIB_CAPTURE_W_DUAL = 1280
CALIB_CAPTURE_H_DUAL = 720
CALIB_OPEN_STAGGER_DUAL_S = 0.85
SELECT_PREVIEW_W = 512
SELECT_PREVIEW_H = 384
SELECT_OPEN_STAGGER_S = 0.55
WINDOW_CALIB = "Intrinsics - Space record/stop | Q quit"
WINDOW_SELECT = "Camera setup - choose cameras, primary, and rotation"
_CALIB_RECORD_DIR = _PROJECT_DIR / "calib_recordings"

# Detect whether OpenCV highgui is available (namedWindow/imshow).
try:
    _test_win = "_cv2_test_window_"
    cv2.namedWindow(_test_win, cv2.WINDOW_NORMAL)
    cv2.destroyWindow(_test_win)
    HAVE_GUI = True
except cv2.error:
    HAVE_GUI = False


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


def _drain_capture(cap: cv2.VideoCapture, n: int = CALIB_WARMUP_READS) -> None:
    """Discard queued frames so USB cameras return a live image (fixes black 2nd cam)."""
    for _ in range(n):
        cap.grab()


def _wait_intrinsics_live_frame(tc: ThreadedCapture, timeout_s: float = 3.0) -> bool:
    """True once the background reader has delivered a non-black real frame."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame, _ = tc.latest()
        if (
            frame is not None
            and frame.size > 0
            and float(frame.mean()) > 2.0
            and not is_inactive_virtual_cam_frame(frame)
        ):
            return True
        time.sleep(0.04)
    return False


# Successful intrinsics open profile: (prefer_mjpeg, requested_width, requested_height).
IntrinsicsOpenProfile = tuple[bool, int, int]


def _intrinsics_attempt_order(
    req_w: int,
    req_h: int,
    align_with: IntrinsicsOpenProfile | None,
    *,
    dual_rig: bool = False,
) -> list[tuple[bool, int, int]]:
    """
    Ordered (prefer_mjpeg, width, height) attempts.
    If align_with is set (from the first camera), those tries run first so all cams
    match resolution/codec when the hardware supports it.
    For the first camera on a dual rig asking for ~1080p, try 720p first so both
    USB streams are more likely to negotiate the same mode.
    """
    ordered: list[tuple[bool, int, int]] = []
    if dual_rig and align_with is None and max(req_w, req_h) >= 1080:
        for prefer_mjpeg in (True, False):
            ordered.append((prefer_mjpeg, 1280, 720))
    if align_with is not None:
        pm, aw, ah = align_with
        ordered.append((pm, aw, ah))
        ordered.append((not pm, aw, ah))
    for prefer_mjpeg in (True, False):
        ordered.append((prefer_mjpeg, req_w, req_h))
    # Prefer HD-ish modes before 800x600 VGA-class fallbacks.
    for fw, fh in ((1280, 720), (960, 720), (960, 540), (800, 600), (640, 480)):
        if (fw, fh) == (req_w, req_h):
            continue
        for prefer_mjpeg in (True, False):
            ordered.append((prefer_mjpeg, fw, fh))
    seen: set[tuple[bool, int, int]] = set()
    out: list[tuple[bool, int, int]] = []
    for t in ordered:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _open_intrinsics_threaded_capture(
    cam_id: str,
    req_w: int,
    req_h: int,
    *,
    align_with: IntrinsicsOpenProfile | None = None,
    dual_rig: bool = False,
) -> tuple[ThreadedCapture | None, IntrinsicsOpenProfile | None]:
    """
    Try modes until live video. Returns (capture, profile) where profile is the
    winning (prefer_mjpeg, requested_w, requested_h) for aligning other cameras.
    """
    attempts = _intrinsics_attempt_order(req_w, req_h, align_with, dual_rig=dual_rig)
    for prefer_mjpeg, w, h in attempts:
        cap = open_camera(
            cam_id,
            w,
            h,
            prefer_mjpeg=prefer_mjpeg,
            backends=INTRINSICS_OPEN_BACKENDS,
        )
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _drain_capture(cap, n=5)
        ok_probe, probe = cap.read()
        probe_ok = (
            ok_probe
            and probe is not None
            and probe.size > 0
            and float(probe.mean()) > 2.0
            and not is_inactive_virtual_cam_frame(probe)
        )
        tc = ThreadedCapture(cap)
        wait_s = 0.5 if probe_ok else 1.0
        if _wait_intrinsics_live_frame(tc, timeout_s=wait_s):
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = "MJPEG" if prefer_mjpeg else "driver default"
            print(
                f"  Camera {cam_id}: preview OK — {aw}x{ah} actual "
                f"(asked {w}x{h}, {codec})."
            )
            return tc, (prefer_mjpeg, w, h)
        tc.release()
    print(
        f"  Camera {cam_id}: no live frames after trying MJPEG/uncompressed and fallbacks."
    )
    return None, None


def _intrinsics_preview_downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink camera frames for intrinsics mosaic + overlay only (not for VideoWriter)."""
    if frame is None or frame.size == 0:
        return frame
    fh, fw = frame.shape[:2]
    m = max(fh, fw)
    if m <= max_side:
        return frame
    sc = max_side / float(m)
    nw = max(1, int(round(fw * sc)))
    nh = max(1, int(round(fh * sc)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _letterbox_bgr(
    frame: np.ndarray,
    out_w: int,
    out_h: int,
    bg_color: tuple[int, int, int] = (20, 24, 32),
) -> np.ndarray:
    """
    Uniformly scale frame to fit inside out_w x out_h, pad with bg_color.
    Preserves source aspect ratio (no non-uniform stretch).
    """
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out[:] = bg_color
    if frame is None or frame.size == 0:
        return out
    fh, fw = frame.shape[:2]
    if fh < 1 or fw < 1:
        return out
    scale = min(out_w / float(fw), out_h / float(fh))
    nw = max(1, int(round(fw * scale)))
    nh = max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _chessboard_sizes_to_try(preferred: tuple[int, int]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for t in (preferred,) + _CHESSBOARD_FALLBACK_SIZES:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _gray_scaled_for_detect(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    """Downscale for chessboard search; return (small_gray, sx, sy) to map corners to full-res coords."""
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray, 1.0, 1.0
    sc = max_side / float(m)
    nw = max(1, int(round(w * sc)))
    nh = max(1, int(round(h * sc)))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, w / float(nw), h / float(nh)


def _find_chessboard_corners_robust(
    gray: np.ndarray,
    board_size: tuple[int, int],
    *,
    detect_max_side: int = CALIB_VIDEO_DETECT_MAX_SIDE,
) -> tuple[bool, np.ndarray | None]:
    """
    Detect on downscaled gray (fast), refine corners on full-res gray.
    Running SB + CLAHE on 1080p every frame can take tens of minutes per video.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    work, sx, sy = _gray_scaled_for_detect(gray, detect_max_side)
    n_expected = board_size[0] * board_size[1]
    gh, gw = gray.shape[:2]
    win = min(11, max(3, min(gw, gh) // 40 * 2 + 1))

    def _subpix_full(corners_small: np.ndarray | None) -> tuple[bool, np.ndarray | None]:
        if corners_small is None or corners_small.size == 0:
            return False, None
        c = np.ascontiguousarray(corners_small.reshape(-1, 1, 2), dtype=np.float32)
        if len(c) != n_expected:
            return False, None
        c[:, 0, 0] *= np.float32(sx)
        c[:, 0, 1] *= np.float32(sy)
        try:
            refined = cv2.cornerSubPix(
                gray,
                c,
                (win, win),
                (-1, -1),
                criteria,
            )
        except cv2.error:
            return False, None
        if refined is None or len(refined) != n_expected:
            return False, None
        return True, refined

    flag_sets = [
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS,
    ]
    grays = [work]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grays.append(clahe.apply(work))

    for g in grays:
        for flags in flag_sets:
            ret, corners = cv2.findChessboardCorners(g, board_size, flags)
            if not ret or corners is None:
                continue
            ok, refined = _subpix_full(corners)
            if ok and refined is not None:
                return True, refined

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = int(
            getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0)
            | getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
            | getattr(cv2, "CALIB_CB_ACCURACY", 0)
        )
        try:
            ret, corners = cv2.findChessboardCornersSB(work, board_size, sb_flags)
            if ret and corners is not None:
                ok, refined = _subpix_full(corners)
                if ok and refined is not None:
                    return True, refined
        except cv2.error:
            pass
    return False, None


def _find_chessboard_corners_preview_fast(
    gray: np.ndarray, board_size: tuple[int, int]
) -> tuple[bool, np.ndarray | None]:
    """Lightweight corner find for video tally only (FAST_CHECK + small images)."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not ret:
        return False, None
    return True, corners


def _open_video_capture(path_str: str) -> cv2.VideoCapture | None:
    _ffmpeg = getattr(cv2, "CAP_FFMPEG", 1900)
    cap = cv2.VideoCapture(path_str, _ffmpeg)
    if not cap.isOpened():
        cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return None
    return cap


def _tally_chessboard_sizes_in_video(
    cap: cv2.VideoCapture,
    candidates: list[tuple[int, int]],
    prefer: tuple[int, int],
) -> dict[tuple[int, int], int]:
    """Count frames where each inner-corner grid matches (fast check on downscaled gray)."""
    tally: dict[tuple[int, int], int] = {c: 0 for c in candidates}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if fi % CALIB_VIDEO_TALLY_FRAME_STRIDE != 0:
            fi += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small, _, _ = _gray_scaled_for_detect(gray, CALIB_VIDEO_DETECT_MAX_SIDE)
        hits = [c for c in candidates if _find_chessboard_corners_preview_fast(small, c)[0]]
        if len(hits) == 1:
            tally[hits[0]] += 1
        elif len(hits) > 1 and prefer in hits:
            tally[prefer] += 1
        elif len(hits) > 1:
            tally[hits[0]] += 1
        fi += 1
    return tally


def _calibrate_intrinsics_from_video_file(
    video_path: Path,
    board_size: tuple[int, int],
    square_size: float,
) -> Intrinsics | None:
    """
    Pick the best-matching inner-corner count (many prints are mis-labeled vs OpenCV),
    then collect corners and calibrate.
    """
    path_str = str(video_path)
    candidates = _chessboard_sizes_to_try(board_size)

    cap_tally = _open_video_capture(path_str)
    if cap_tally is None:
        print(f"  Could not open recording: {video_path}")
        return None
    print(f"  Scanning {video_path.name} for board size (subsampled, fast)...")
    tally = _tally_chessboard_sizes_in_video(cap_tally, candidates, board_size)
    cap_tally.release()

    best_size = max(tally, key=lambda k: tally[k])
    best_count = tally[best_size]
    min_tally_hits = max(
        6,
        (MIN_CALIB_FRAMES + CALIB_VIDEO_TALLY_FRAME_STRIDE - 1)
        // CALIB_VIDEO_TALLY_FRAME_STRIDE,
    )
    if best_count < min_tally_hits:
        print(
            f"  Not enough chessboard frames in video (best pattern {best_size[0]}x{best_size[1]} "
            f"matched {best_count}x on subsampled frames, need {min_tally_hits}). File: {video_path.name}"
        )
        print(f"  Tried inner-corner grids (cols x rows): {tally}")
        print(
            "  OpenCV counts INNER corners (intersections), not squares. "
            "Example: 9x6 printed squares often means (8,5) inner corners."
        )
        return None

    if best_size != board_size:
        print(
            f"  Using detected board {best_size[0]}x{best_size[1]} inner corners "
            f"({best_count} frames); set CHESSBOARD_INNER_CORNERS = ({best_size[0]}, {best_size[1]}) "
            "in calibrate_cameras.py and use the same in calibrate_extrinsics_3d.py."
        )

    cap = _open_video_capture(path_str)
    if cap is None:
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ok, test = cap.read()
        if not ok or test is None:
            cap.release()
            return None
        h, w = test.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    objp = _make_object_points(best_size[0], best_size[1], square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    n_read = 0
    first_mean: float | None = None
    print(
        f"  Collecting board poses from {video_path.name} "
        f"(every {CALIB_VIDEO_COLLECT_STRIDE} frames, max {CALIB_VIDEO_MAX_BOARD_FRAMES})..."
    )

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        n_read += 1
        if CALIB_VIDEO_PROGRESS_EVERY_FRAMES > 0 and n_read % CALIB_VIDEO_PROGRESS_EVERY_FRAMES == 0:
            print(f"    ... decoded {n_read} frames, {len(objpoints)} board hits so far")
        if (n_read - 1) % CALIB_VIDEO_COLLECT_STRIDE != 0:
            continue
        if first_mean is None and frame.size > 0:
            first_mean = float(frame.mean())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = _find_chessboard_corners_robust(gray, best_size)
        if ret and corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            if len(objpoints) >= CALIB_VIDEO_MAX_BOARD_FRAMES:
                break

    cap.release()

    if len(objpoints) < MIN_CALIB_FRAMES:
        print(
            f"  Refined pass: only {len(objpoints)} frames (need {MIN_CALIB_FRAMES}). File: {video_path.name}"
        )
        print(
            f"  Debug: decoded {n_read} frames; first-frame mean BGR={first_mean}; "
            "Try slower motion, sharper focus, and even light on the board."
        )
        return None

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        mean_error += float(np.linalg.norm(imgpoints[i] - imgpts2))
    mean_error /= len(objpoints)
    print(
        f"  From video {video_path.name}: board {best_size[0]}x{best_size[1]} inner corners, "
        f"{len(objpoints)} frames, reprojection error {mean_error:.4f} px"
    )
    return Intrinsics(K=K, dist=dist, image_size=(w, h), reproj_error=mean_error)


def _safe_filename_part(cam_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in cam_id)[:80]


def _build_intrinsics_mosaic(
    frames_by_id: list[tuple[str, np.ndarray]],
    recording: bool,
    elapsed_s: float,
) -> np.ndarray:
    """Side-by-side preview: each camera letterboxed to a 4:3 cell (chessboard is detected after recording)."""
    if not frames_by_id:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    tiles: list[np.ndarray] = []
    for cam_id, frame in frames_by_id:
        disp = _letterbox_bgr(frame, CALIB_TILE_MAX_W, CALIB_TILE_MAX_H)
        tile = disp.copy()
        cv2.putText(
            tile,
            f"Cam {cam_id}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    gap = 8
    if len(tiles) == 1:
        mosaic = tiles[0]
    else:
        mosaic = tiles[0]
        for t in tiles[1:]:
            gap_canvas = np.zeros((mosaic.shape[0], gap, 3), dtype=np.uint8)
            gap_canvas[:] = (30, 34, 42)
            mosaic = np.hstack([mosaic, gap_canvas, t])

    banner_h = 100
    out = np.zeros((mosaic.shape[0] + banner_h, mosaic.shape[1], 3), dtype=np.uint8)
    out[:] = (14, 18, 26)
    out[banner_h:, :] = mosaic
    rec_txt = "REC" if recording else "PREVIEW"
    cv2.putText(
        out,
        f"Intrinsics calibration | {rec_txt} | {elapsed_s:0.1f}s",
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (242, 242, 242),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "Chessboard detection runs on saved videos after you stop recording.",
        (16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (188, 194, 206),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "SPACE start/stop recording (all cams) | Q quit",
        (16, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (239, 170, 84),
        1,
        cv2.LINE_AA,
    )
    return out


def _run_intrinsics_record_session(
    cam_ids: list[str],
    rotations: dict[str, int],
    board_size: tuple[int, int],
    square_size: float,
) -> dict[str, Intrinsics | None]:
    """
    Open all cameras together, show full mosaic preview, record synchronized videos,
    then run calibrateCamera from each file.
    """
    out: dict[str, Intrinsics | None] = {cid: None for cid in cam_ids}
    if not cam_ids:
        return out

    if not HAVE_GUI:
        print("  OpenCV GUI not available; cannot run intrinsics recording session.")
        return out

    n_cams = len(cam_ids)
    if n_cams >= 2:
        cap_w, cap_h = CALIB_CAPTURE_W_DUAL, CALIB_CAPTURE_H_DUAL
        open_stagger = CALIB_OPEN_STAGGER_DUAL_S
    else:
        cap_w, cap_h = CALIB_CAPTURE_W, CALIB_CAPTURE_H
        open_stagger = CALIB_OPEN_STAGGER_S

    print(
        "  Opening cameras for intrinsics (short wait per device; failed modes are skipped quickly)..."
    )
    threaded_caps: dict[str, ThreadedCapture] = {}
    leader_profile: IntrinsicsOpenProfile | None = None
    for i, cid in enumerate(cam_ids):
        if i > 0:
            time.sleep(open_stagger)
        align = leader_profile if n_cams >= 2 else None
        tc, prof = _open_intrinsics_threaded_capture(
            cid, cap_w, cap_h, align_with=align, dual_rig=(n_cams >= 2)
        )
        if tc is None:
            for t in threaded_caps.values():
                t.release()
            return out
        threaded_caps[cid] = tc
        if i == 0 and prof is not None:
            leader_profile = prof

    # Brief settle after staggered opens (per-cam live check already ran).
    time.sleep(0.2 + 0.15 * max(0, n_cams - 1))

    if n_cams >= 2:
        live_shapes: dict[str, tuple[int, int]] = {}
        for cid in cam_ids:
            fr, _ = threaded_caps[cid].latest()
            if fr is not None and fr.size > 0:
                live_shapes[cid] = (fr.shape[1], fr.shape[0])
        uniq = set(live_shapes.values())
        if len(uniq) > 1:
            print(
                "  Warning: cameras are recording different frame sizes "
                f"{live_shapes}. Prefer separate USB root ports or lower "
                "CALIB_CAPTURE_W_DUAL / CALIB_CAPTURE_H_DUAL if bandwidth is the limit."
            )

    _CALIB_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    video_paths: dict[str, Path] = {
        cid: _CALIB_RECORD_DIR / f"intrinsics_{_safe_filename_part(cid)}_{stamp}.avi"
        for cid in cam_ids
    }
    writers: dict[str, cv2.VideoWriter] = {}
    recording = False
    record_start = 0.0
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    cv2.namedWindow(WINDOW_CALIB, cv2.WINDOW_NORMAL)

    print("  Cameras use background capture with serialized USB reads so both streams stay alive.")
    if n_cams >= 2:
        print(
            f"  Multi-cam: requesting {cap_w}x{cap_h} (MJPEG for bandwidth). "
            "If you still see ~800x600, use different USB root ports or raise CALIB_CAPTURE_W_DUAL/H_DUAL."
        )
    else:
        print(
            f"  Requested capture size {cap_w}x{cap_h} (4:3); "
            "preview tiles are letterboxed 4:3."
        )
    print("  SPACE = start/stop recording (all selected cameras). Move the board while recording.")
    print("  Q = abort. Videos save under:", _CALIB_RECORD_DIR)
    print(
        f"  Live preview is downscaled (max side {CALIB_INTRINSICS_PREVIEW_MAX_SRC}px); "
        "recorded files use full camera resolution."
    )

    finished = False
    try:
        while True:
            frames_order: list[tuple[str, np.ndarray]] = []
            for cid in cam_ids:
                frame, _ts = threaded_caps[cid].latest()
                if frame is None:
                    frame = np.zeros((cap_h, cap_w, 3), dtype=np.uint8)
                rot = rotations.get(cid, 0)
                if rot:
                    frame = apply_rotation(frame, rot)
                frames_order.append((cid, frame))

            frames_preview = [
                (
                    cid,
                    _intrinsics_preview_downscale(fr, CALIB_INTRINSICS_PREVIEW_MAX_SRC),
                )
                for cid, fr in frames_order
            ]

            elapsed = (time.time() - record_start) if recording else 0.0
            if recording and elapsed >= CALIB_RECORD_MAX_S:
                recording = False
                for w in writers.values():
                    w.release()
                writers.clear()
                print("  Recording stopped (max length).")

            mosaic = _build_intrinsics_mosaic(frames_preview, recording, elapsed)

            mw, mh = mosaic.shape[1], mosaic.shape[0]
            max_disp_w, max_disp_h = 1920, 1080
            disp_scale = min(max_disp_w / float(mw), max_disp_h / float(mh), 1.0)
            if disp_scale < 1.0:
                nd_w = max(1, int(round(mw * disp_scale)))
                nd_h = max(1, int(round(mh * disp_scale)))
                display = cv2.resize(mosaic, (nd_w, nd_h), interpolation=cv2.INTER_AREA)
            else:
                display = mosaic
            dw, dh = display.shape[1], display.shape[0]
            cv2.imshow(WINDOW_CALIB, display)
            cv2.resizeWindow(WINDOW_CALIB, dw, dh)

            if recording:
                for cid, frame in frames_order:
                    w = writers.get(cid)
                    if w is not None:
                        w.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                if recording:
                    recording = False
                    for w in writers.values():
                        w.release()
                    writers.clear()
                for p in video_paths.values():
                    if p.exists():
                        p.unlink(missing_ok=True)
                print("  Aborted intrinsics recording.")
                return out

            if key == 32:  # Space
                if not recording:
                    # Each camera may have different (w,h) after 90/270 rotation; writers must match.
                    for cid in cam_ids:
                        frame_c = next(f for c, f in frames_order if c == cid)
                        fh, fw = frame_c.shape[:2]
                        vw = cv2.VideoWriter(
                            str(video_paths[cid]),
                            fourcc,
                            CALIB_RECORD_FPS,
                            (fw, fh),
                        )
                        if not vw.isOpened():
                            print(f"  Could not open video writer for {cid}")
                            vw.release()
                            for w in writers.values():
                                w.release()
                            writers.clear()
                            return out
                        _vq = getattr(cv2, "VIDEOWRITER_PROP_QUALITY", None)
                        if _vq is not None:
                            try:
                                vw.set(_vq, 95)
                            except cv2.error:
                                pass
                        writers[cid] = vw
                    recording = True
                    record_start = time.time()
                    print("  Recording... (SPACE again to stop)")
                    for cid in cam_ids:
                        fc = next(f for c, f in frames_order if c == cid)
                        print(f"    writer {cid}: {fc.shape[1]}x{fc.shape[0]} px (must match rotated frame)")
                else:
                    recording = False
                    for w in writers.values():
                        w.release()
                    writers.clear()
                    print("  Recording saved. Processing videos...")
                    finished = True
                    break
    finally:
        for w in writers.values():
            w.release()
        writers.clear()
        for tc in threaded_caps.values():
            tc.release()
        threaded_caps.clear()
        cv2.destroyWindow(WINDOW_CALIB)

    if not finished:
        return out

    for cid in cam_ids:
        path = video_paths[cid]
        if not path.exists() or path.stat().st_size < 1000:
            print(f"  Missing or empty recording for camera {cid}.")
            continue
        intr = _calibrate_intrinsics_from_video_file(path, board_size, square_size)
        out[cid] = intr

    return out


def _short_label(cam_id: str) -> str:
    """Readable short label for a camera ID."""
    return f"Cam {cam_id}"


def _open_preview_capture(cam_id: str) -> ThreadedCapture | None:
    """Open a modest-resolution preview for selection UI (easier on USB with several cams)."""
    cap = open_camera(cam_id, SELECT_PREVIEW_W, SELECT_PREVIEW_H)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return ThreadedCapture(cap)


def _wait_for_usable_preview(tc: ThreadedCapture, timeout_s: float = 2.0) -> bool:
    """Wait for a real video frame (not black / virtual-cam idle splash)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame, _ = tc.latest()
        if (
            frame is not None
            and frame.size > 0
            and float(frame.mean()) > 2.0
            and not is_inactive_virtual_cam_frame(frame)
        ):
            return True
        time.sleep(0.05)
    frame, _ = tc.latest()
    return (
        frame is not None
        and frame.size > 0
        and float(frame.mean()) > 2.0
        and not is_inactive_virtual_cam_frame(frame)
    )


def _build_selection_tile(
    cam_id: str,
    tc: ThreadedCapture,
    selected: set[str],
    primary: str,
    focused_cam: str,
    rotations: dict[str, int],
) -> np.ndarray:
    """Render one preview tile for the selection mosaic."""
    frame, _ts = tc.latest()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rot = rotations.get(cam_id, 0)
    if rot:
        frame = apply_rotation(frame, rot)
    tile = _letterbox_bgr(frame, SELECT_TILE_W, SELECT_TILE_H)

    if cam_id == primary:
        border = (84, 208, 120)
    elif cam_id in selected:
        border = (239, 170, 84)
    else:
        border = (72, 76, 90)
    cv2.rectangle(tile, (0, 0), (SELECT_TILE_W - 1, SELECT_TILE_H - 1), border, 3)
    if cam_id == focused_cam:
        cv2.rectangle(tile, (6, 6), (SELECT_TILE_W - 7, SELECT_TILE_H - 7), (189, 135, 255), 2)

    label = _short_label(cam_id)
    status = "SELECTED" if cam_id in selected else "off"
    if cam_id == primary:
        status += " | PRIMARY"
    if rot:
        status += f" | rot={rot}"
    cv2.putText(
        tile,
        f"{label} ({cam_id})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        status,
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        border,
        2,
        cv2.LINE_AA,
    )
    return tile


def _render_selection_mosaic(
    ordered_ids: list[str],
    preview_caps: dict[str, ThreadedCapture],
    selected: set[str],
    primary: str,
    focused_cam: str,
    rotations: dict[str, int],
) -> np.ndarray:
    """Render the full camera selection window as a single image."""
    if not ordered_ids:
        canvas = np.zeros((220, 900, 3), dtype=np.uint8)
        cv2.putText(canvas, "No preview cameras available", (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Q to quit", (24, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return canvas

    cols = min(2, max(1, int(math.ceil(math.sqrt(len(ordered_ids))))))
    rows = int(math.ceil(len(ordered_ids) / float(cols)))
    header_h = 112
    canvas_w = cols * SELECT_TILE_W + (cols + 1) * SELECT_GAP
    canvas_h = header_h + rows * SELECT_TILE_H + (rows + 1) * SELECT_GAP
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (14, 18, 26)

    selected_label = ", ".join(sorted(selected)) if selected else "none"
    help_line = "0-9 toggle | P primary | R rotate | Enter confirm | Q cancel"
    cv2.putText(canvas, "Camera setup", (SELECT_GAP, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (242, 242, 242), 2, cv2.LINE_AA)
    cv2.putText(canvas, help_line, (SELECT_GAP, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (188, 194, 206), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Selected: {selected_label} | Primary: {primary if primary else 'none'} | Focus: {focused_cam if focused_cam else 'none'}",
        (SELECT_GAP, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (239, 170, 84),
        1,
        cv2.LINE_AA,
    )

    for idx, cam_id in enumerate(ordered_ids):
        row = idx // cols
        col = idx % cols
        x0 = SELECT_GAP + col * (SELECT_TILE_W + SELECT_GAP)
        y0 = header_h + SELECT_GAP + row * (SELECT_TILE_H + SELECT_GAP)
        tile = _build_selection_tile(cam_id, preview_caps[cam_id], selected, primary, focused_cam, rotations)
        canvas[y0:y0 + SELECT_TILE_H, x0:x0 + SELECT_TILE_W] = tile
    return canvas


def run_camera_selection(
    cameras: list[CameraInfo],
    last_setup: LastCameraSetup | None,
) -> tuple[list[str], str, dict[str, int]]:
    """
    GUI camera selection with live previews for local (USB) camera indices.

    Controls shown on screen:
      0-9   toggle a camera
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

    selected: set[str] = set(last_setup.selected_camera_ids) if last_setup and last_setup.selected_camera_ids else set()
    rotations = dict(last_setup.camera_rotations) if last_setup and last_setup.camera_rotations else {}

    preview_caps: dict[str, ThreadedCapture] = {}
    ordered_ids: list[str] = []
    for i, cid in enumerate(all_ids):
        if i > 0:
            time.sleep(SELECT_OPEN_STAGGER_S)
        tc = _open_preview_capture(cid)
        if tc is None:
            continue
        if not _wait_for_usable_preview(tc):
            tc.release()
            continue
        preview_caps[cid] = tc
        ordered_ids.append(cid)

    if not ordered_ids:
        print("No previewable cameras opened.")
        return [], "", rotations

    selected = {cid for cid in selected if cid in ordered_ids}
    if not selected:
        selected.add(ordered_ids[0])

    primary = ordered_ids[0]
    if last_setup and last_setup.primary_camera_id in ordered_ids:
        primary = last_setup.primary_camera_id
    elif primary not in selected and selected:
        primary = sorted(selected)[0]

    focused_cam = primary if primary in ordered_ids else ordered_ids[0]

    print("Step 1/2: select cameras.")
    print("  0-9  toggle camera")
    print("  R    rotate focused camera (cycles 0/90/180/270)")
    print("  P    cycle primary")
    print("  Enter  confirm and continue")
    print("  Q    cancel setup")

    if HAVE_GUI:
        cv2.namedWindow(WINDOW_SELECT, cv2.WINDOW_NORMAL)

    try:
        while True:
            if HAVE_GUI:
                mosaic = _render_selection_mosaic(
                    ordered_ids, preview_caps, selected, primary, focused_cam, rotations
                )
                mw, mh = mosaic.shape[1], mosaic.shape[0]
                max_disp_w, max_disp_h = 1920, 1080
                sel_scale = min(max_disp_w / float(mw), max_disp_h / float(mh), 1.0)
                if sel_scale < 1.0:
                    sw = max(1, int(round(mw * sel_scale)))
                    sh = max(1, int(round(mh * sel_scale)))
                    display_sel = cv2.resize(mosaic, (sw, sh), interpolation=cv2.INTER_AREA)
                else:
                    display_sel = mosaic
                dw, dh = display_sel.shape[1], display_sel.shape[0]
                cv2.imshow(WINDOW_SELECT, display_sel)
                cv2.resizeWindow(WINDOW_SELECT, dw, dh)

            key = cv2.waitKey(1) & 0xFF if HAVE_GUI else ord("q")
            if key in (13, 10):
                break
            if key in (ord("q"), ord("Q")):
                selected = set()
                break
            if key in (ord("r"), ord("R")) and focused_cam in ordered_ids:
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
                if k in ordered_ids:
                    if k in selected:
                        selected.discard(k)
                    else:
                        selected.add(k)
                    focused_cam = k
                    if primary not in selected and selected:
                        primary = sorted(selected)[0]
                    print(f"  Selected: {sorted(selected)}, primary: {primary}")
    finally:
        for tc in preview_caps.values():
            tc.release()
        if HAVE_GUI:
            cv2.destroyWindow(WINDOW_SELECT)

    return sorted(selected), primary, rotations


def main() -> None:
    print("Multi-camera calibration and setup")
    print("----------------------------------")
    print(
        "Camera indices (0, 1, 2, ...) are USB plug order, not a fixed device ID. "
        "New webcams or different ports mean old camera_calibrations.json / "
        "last_camera_setup.json may describe the wrong hardware — delete or "
        "overwrite those entries and calibrate again for the cameras you use now."
    )
    print()
    cameras = detect_connected_cameras()
    if not cameras:
        print("No local cameras detected. Connect USB webcams and try again.")
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
    baseline_m = last_setup.camera_baseline_m if last_setup else None

    print("Step 2/2: calibrate intrinsics (record all cameras together, then auto-calibrate from video).")

    for cam_id in selected_ids:
        if cam_id in calibrations:
            print(
                f"Camera {cam_id} already has intrinsics — skipping capture "
                f"(delete its entry in {CALIBRATIONS_PATH} to redo)."
            )

    need_intrinsics = [cid for cid in selected_ids if cid not in calibrations]
    if need_intrinsics:
        results = _run_intrinsics_record_session(
            need_intrinsics, cam_rotations, board_size, square_size
        )
        for cam_id, intr in results.items():
            if intr is None:
                continue
            label = _short_label(cam_id)
            calibrations[cam_id] = Calibration(
                camera_id=cam_id,
                intrinsics=intr,
                extrinsics=None,
                metadata={"label": label},
            )
            save_calibrations(calibrations)

    calibrations = load_calibrations()
    selected_calibrated = [cid for cid in selected_ids if cid in calibrations]
    if not selected_calibrated:
        print("No cameras were successfully calibrated.")
        return

    # Save the full selected rig (0+2 etc.), not only cameras that finished intrinsics this run,
    # so main.py can open the same indices after partial sessions.
    setup = LastCameraSetup(
        selected_camera_ids=selected_ids,
        primary_camera_id=primary_id,
        use_triangulation=True,
        camera_rotations=cam_rotations,
        camera_baseline_m=baseline_m,
    )
    save_last_setup(setup)
    print(f"Setup saved: rig {selected_ids} (calibrated now: {selected_calibrated}), primary {primary_id}")
    print("Run main.py to use these cameras.")
    if len(selected_calibrated) >= 2:
        print("For 3D triangulation, run: python calibrate_extrinsics_3d.py")


if __name__ == "__main__":
    main()
