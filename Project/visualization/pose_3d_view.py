"""Orthographic 3D skeleton preview from triangulated MediaPipe landmarks."""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    PANEL_FONT,
    PANEL_TEXT_THICK,
    POSE_3D_WORLD_Y_ROT_DEG,
    POSE_3D_WORLD_Z_ROT_DEG,
    TEXT_MUTED,
    TEXT_PRIMARY,
    VIDEO_BG,
    VIS_MIN,
)
from landmarks import EDGES, EDGE_COLORS, edge_thickness


def _blit_title(canvas: np.ndarray, title: str, subtitle: str | None = None) -> None:
    cv2.putText(
        canvas,
        title,
        (12, 22),
        PANEL_FONT,
        0.52,
        TEXT_PRIMARY,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            canvas,
            subtitle,
            (12, 44),
            PANEL_FONT,
            0.45,
            TEXT_MUTED,
            PANEL_TEXT_THICK,
            cv2.LINE_AA,
        )


def draw_3d_pose_canvas(
    canvas: np.ndarray,
    pts_3d: np.ndarray | None,
    vis_3d: np.ndarray | None = None,
    *,
    yaw_deg: float = 24.0,
    pitch_deg: float = -14.0,
    title: str = "3D skeleton (triangulated)",
) -> None:
    """
    Draw a fixed-orientation orthographic view of the pose into canvas (BGR, in-place).
    pts_3d: (33, 3) world units; NaN rows are skipped. vis_3d optional (33,) in [0,1].
    """
    h, w = canvas.shape[:2]
    canvas[:] = VIDEO_BG

    if pts_3d is None:
        _blit_title(canvas, title, "Waiting for multi-view triangulation…")
        return

    p = np.asarray(pts_3d, dtype=np.float64)
    if p.shape != (33, 3):
        _blit_title(canvas, title, "Invalid 3D data")
        return

    mask = np.isfinite(p).all(axis=1)
    if vis_3d is not None:
        v = np.asarray(vis_3d, dtype=np.float64).reshape(-1)
        if v.shape[0] >= 33:
            mask = mask & (v[:33] >= float(VIS_MIN))

    if not np.any(mask):
        _blit_title(canvas, title, "Landmarks not visible in 2+ cameras")
        return

    torso_idx = [11, 12, 23, 24]
    valid_torso = [i for i in torso_idx if mask[i]]
    if valid_torso:
        center = p[valid_torso].mean(axis=0)
    else:
        center = p[mask].mean(axis=0)
    P = p - center

    # Align chessboard world to a natural upright view (extrinsics use board XY plane, Z out).
    def _rot_y(rad: float) -> np.ndarray:
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    def _rot_z(rad: float) -> np.ndarray:
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    ay = np.radians(float(POSE_3D_WORLD_Y_ROT_DEG))
    az = np.radians(float(POSE_3D_WORLD_Z_ROT_DEG))
    if abs(ay) > 1e-6 or abs(az) > 1e-6:
        R_align = _rot_y(ay) @ _rot_z(az)
        P = (R_align @ P.T).T

    yr, pr = np.radians(yaw_deg), np.radians(pitch_deg)
    cy, sy = np.cos(yr), np.sin(yr)
    cp, sp = np.cos(pr), np.sin(pr)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    R = Ry @ Rx
    V = (R @ P.T).T
    xv = V[:, 0]
    yv = -V[:, 2]

    finite_xy = np.isfinite(xv) & np.isfinite(yv) & mask
    if not np.any(finite_xy):
        _blit_title(canvas, title, "No points in view")
        return

    xmin = float(np.min(xv[finite_xy]))
    xmax = float(np.max(xv[finite_xy]))
    ymin = float(np.min(yv[finite_xy]))
    ymax = float(np.max(yv[finite_xy]))
    dx = max(xmax - xmin, 1e-3)
    dy = max(ymax - ymin, 1e-3)

    margin = 0.1
    cx0 = w * 0.5
    cy0 = h * 0.52
    scale = min((w * (1.0 - 2 * margin)) / dx, (h * (1.0 - 2 * margin)) / dy) * 0.88

    def proj(i: int) -> tuple[int, int]:
        return (
            int(round(cx0 + xv[i] * scale)),
            int(round(cy0 + yv[i] * scale)),
        )

    for (a, b), col in zip(EDGES, EDGE_COLORS):
        if not mask[a] or not mask[b]:
            continue
        pa, pb = proj(a), proj(b)
        cv2.line(
            canvas,
            pa,
            pb,
            col,
            max(2, edge_thickness(a, b)),
            cv2.LINE_AA,
        )

    _blit_title(canvas, title, None)
