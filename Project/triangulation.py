"""
Multi-camera triangulation and 3D joint angle computation for the pose dashboard.

Uses projection matrices from camera_config and per-camera 2D landmarks to
triangulate 3D positions and compute joint angles in 3D when 2+ cameras are available.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from camera_config import Calibration, get_projection_matrix
from config import ANGLE_MAX, ANGLE_MIN, VIS_MIN
from landmarks import (
    L_ANKLE,
    L_ELBOW,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from pose_processor import ANGLE_KEYS

NUM_LANDMARKS = 33


def _triangulate_point_two_views(P1: np.ndarray, P2: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """Triangulate one 3D point from two views. pt1, pt2 are (2,) pixel coords. Returns (3,) world point."""
    pts1 = np.array(pt1, dtype=np.float64).reshape(2, 1)
    pts2 = np.array(pt2, dtype=np.float64).reshape(2, 1)
    X_h = cv2.triangulatePoints(P1, P2, pts1, pts2)
    X = (X_h[:3, 0] / X_h[3, 0])
    return X


def _triangulate_point_multiview(Ps: list[np.ndarray], points: list[np.ndarray]) -> np.ndarray:
    """
    Triangulate one 3D point from N views via linear least squares (SVD).
    Ps: list of 3x4 projection matrices. points: list of (2,) pixel coords.
    Returns (3,) world point.
    """
    if len(Ps) == 2:
        return _triangulate_point_two_views(Ps[0], Ps[1], points[0], points[1])
    # Build A such that A @ X = 0 (homogeneous). For each view: u*P3 - P1 = 0, v*P3 - P2 = 0.
    A = []
    for P, pt in zip(Ps, points):
        u, v = float(pt[0]), float(pt[1])
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]
    X = (X_h[:3] / X_h[3])
    return X


def triangulate_landmarks(
    calibrations: dict[str, Calibration],
    per_cam_pts_norm: dict[str, np.ndarray],
    per_cam_vis: dict[str, np.ndarray],
    per_cam_image_size: dict[str, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 33 landmarks from multiple camera views.

    per_cam_pts_norm: camera_id -> (33, 2) normalized [0,1] image coords (MediaPipe format).
    per_cam_vis: camera_id -> (33,) visibility per landmark.
    per_cam_image_size: camera_id -> (width, height).

    Returns:
        pts_3d: (33, 3) world coordinates (NaN where insufficient views).
        vis_agg: (33,) aggregated visibility (1.0 if triangulated, 0 otherwise).
    """
    import cv2
    pts_3d = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    vis_agg = np.zeros((NUM_LANDMARKS,), dtype=np.float32)
    cam_ids = list(per_cam_pts_norm.keys())
    for j in range(NUM_LANDMARKS):
        Ps = []
        points = []
        for cid in cam_ids:
            if cid not in calibrations or cid not in per_cam_vis:
                continue
            v = per_cam_vis[cid][j]
            if not (np.isfinite(v) and float(v) >= VIS_MIN):
                continue
            cal = calibrations[cid]
            P = get_projection_matrix(cal)
            w, h = per_cam_image_size.get(cid, (1, 1))
            x_norm = float(per_cam_pts_norm[cid][j, 0])
            y_norm = float(per_cam_pts_norm[cid][j, 1])
            u = x_norm * w
            v_px = y_norm * h
            Ps.append(P)
            points.append(np.array([u, v_px], dtype=np.float64))
        if len(Ps) < 2:
            continue
        try:
            X = _triangulate_point_multiview(Ps, points)
            if np.all(np.isfinite(X)):
                pts_3d[j, :] = X.astype(np.float32)
                vis_agg[j] = 1.0
        except Exception:
            pass
    return pts_3d, vis_agg


def _angle_deg_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b between ba and bc in degrees (works for 3D)."""
    ba = a - b
    bc = c - b
    na = float(np.linalg.norm(ba))
    nb = float(np.linalg.norm(bc))
    if na < 1e-6 or nb < 1e-6:
        return np.nan
    cosang = float(np.dot(ba, bc) / (na * nb))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def _clamp(x: float, lo: float, hi: float) -> float:
    if x is None or not np.isfinite(x):
        return np.nan
    return max(lo, min(hi, float(x)))


def compute_angles_3d(
    pts_3d: np.ndarray,
    vis_agg: np.ndarray,
) -> dict[str, float]:
    """
    Compute joint angles from 3D landmarks. Same keys as pose_processor.ANGLE_KEYS.
    Returns dict of angle name -> degrees (or np.nan).
    """
    vals = {k: np.nan for k in ANGLE_KEYS}

    def vis_ok(j: int) -> bool:
        return vis_agg[j] >= VIS_MIN and np.all(np.isfinite(pts_3d[j]))

    def angle_if(a: int, b: int, c: int) -> float:
        if vis_ok(a) and vis_ok(b) and vis_ok(c):
            return _clamp(_angle_deg_3d(pts_3d[a], pts_3d[b], pts_3d[c]), ANGLE_MIN, ANGLE_MAX)
        return np.nan

    def ankle_angle(side: str) -> float:
        if side == "L":
            knee, ankle, heel, toe = L_KNEE, L_ANKLE, L_HEEL, L_FOOT_INDEX
        else:
            knee, ankle, heel, toe = R_KNEE, R_ANKLE, R_HEEL, R_FOOT_INDEX
        if not (vis_ok(knee) and vis_ok(ankle) and vis_ok(heel) and vis_ok(toe)):
            return np.nan
        foot = pts_3d[toe] - pts_3d[heel]
        shin = pts_3d[ankle] - pts_3d[knee]
        if float(np.linalg.norm(foot)) < 1e-6 or float(np.linalg.norm(shin)) < 1e-6:
            return np.nan
        # Angle at heel between foot and shin directions
        ang = _angle_deg_3d(pts_3d[heel] + foot, pts_3d[heel], pts_3d[heel] + shin)
        return _clamp(ang, ANGLE_MIN, ANGLE_MAX)

    vals["Hip L"] = float(angle_if(L_SHOULDER, L_HIP, L_KNEE))
    vals["Hip R"] = float(angle_if(R_SHOULDER, R_HIP, R_KNEE))
    vals["Knee L"] = float(angle_if(L_HIP, L_KNEE, L_ANKLE))
    vals["Knee R"] = float(angle_if(R_HIP, R_KNEE, R_ANKLE))
    vals["Shoulder L"] = float(angle_if(L_ELBOW, L_SHOULDER, L_HIP))
    vals["Shoulder R"] = float(angle_if(R_ELBOW, R_SHOULDER, R_HIP))
    vals["Elbow L"] = float(angle_if(L_SHOULDER, L_ELBOW, L_WRIST))
    vals["Elbow R"] = float(angle_if(R_SHOULDER, R_ELBOW, R_WRIST))
    vals["Ank L"] = float(ankle_angle("L"))
    vals["Ank R"] = float(ankle_angle("R"))
    return vals


def process_multi_cam_poses(
    per_cam_results: list[tuple[str, Any, Any, Any, dict, np.ndarray | None, np.ndarray | None]],
    calibrations: dict[str, Calibration],
) -> tuple[dict[str, float], np.ndarray | None]:
    """
    Combine per-camera pose results and triangulate to get 3D angles.

    per_cam_results: list of (cam_id, pts, vis, pts_norm, vis_arr, vals, pts_norm_snapshot, vis_snapshot)
                    as returned by process_pose for each camera. pts_norm_snapshot (33,2), vis_snapshot (33,).
    calibrations: dict camera_id -> Calibration (must have extrinsics for triangulation).

    Returns:
        vals: ANGLE_KEYS -> angle in degrees (or np.nan).
        pts_3d: (33, 3) or None if triangulation not possible.
    """
    per_cam_pts_norm: dict[str, np.ndarray] = {}
    per_cam_vis: dict[str, np.ndarray] = {}
    per_cam_image_size: dict[str, tuple[int, int]] = {}
    for cam_id, _pts, _vis, _pts_norm, _vis_arr, _vals, pts_norm_snap, vis_snap in per_cam_results:
        if pts_norm_snap is None or vis_snap is None:
            continue
        cal = calibrations.get(cam_id)
        if cal is None or cal.extrinsics is None:
            continue
        per_cam_pts_norm[cam_id] = pts_norm_snap
        per_cam_vis[cam_id] = vis_snap
        per_cam_image_size[cam_id] = cal.intrinsics.image_size
    if len(per_cam_pts_norm) < 2:
        # Fallback: use first camera's 2D angles if available
        if per_cam_results:
            _, _, _, _, vals, _, _, _ = per_cam_results[0]
            return vals, None
        return {k: np.nan for k in ANGLE_KEYS}, None
    pts_3d, vis_agg = triangulate_landmarks(
        calibrations, per_cam_pts_norm, per_cam_vis, per_cam_image_size
    )
    vals = compute_angles_3d(pts_3d, vis_agg)
    return vals, pts_3d
