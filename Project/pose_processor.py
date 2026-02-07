"""Pose processing: angle calculations and landmark extraction."""

import math

import numpy as np

from config import ANGLE_MAX, ANGLE_MIN, MIRROR_VIEW, VIS_MIN
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

ANGLE_KEYS = [
    "Hip L", "Hip R",
    "Knee L", "Knee R",
    "Ank L", "Ank R",
    "Shoulder L", "Shoulder R",
    "Elbow L", "Elbow R",
]

# Mirror swap pairs for L/R landmarks
SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6),   # eyes
    (7, 8), (9, 10),          # ears, mouth
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),  # arms
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),  # legs, feet
]


def clamp(x, lo, hi):
    if x is None or not np.isfinite(x):
        return np.nan
    return max(lo, min(hi, float(x)))


def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    na = float(np.linalg.norm(ba))
    nb = float(np.linalg.norm(bc))
    if na < 1e-6 or nb < 1e-6:
        return np.nan
    cosang = float(np.dot(ba, bc) / (na * nb))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def round_deg(x):
    if x is None or not np.isfinite(x):
        return None
    return int(round(float(x)))


def vis_ok(v):
    return v is not None and np.isfinite(v) and float(v) >= VIS_MIN


def process_pose(lm, frame_h, frame_w, mirror_view=MIRROR_VIEW):
    """
    Extract landmarks and compute angles from MediaPipe pose result.
    Returns (pts, vis, pts_norm, vis_arr, vals, pts_norm_snapshot, vis_snapshot)
    or (None, None, None, None, vals, None, None) if no pose.
    """
    vals = {k: np.nan for k in ANGLE_KEYS}
    if lm is None or len(lm) == 0:
        return None, None, None, None, vals, None, None

    pts = {}
    vis = {}
    pts_norm = np.full((33, 2), np.nan, dtype=np.float32)
    vis_arr = np.zeros((33,), dtype=np.float32)

    for i in range(len(lm)):
        x = float(lm[i].x)
        y = float(lm[i].y)
        pts[i] = np.array([x * frame_w, y * frame_h], dtype=np.float32)
        v = getattr(lm[i], "visibility", 0.0)
        vv = float(v) if v is not None else 0.0
        vis[i] = vv
        pts_norm[i, 0] = x
        pts_norm[i, 1] = y
        vis_arr[i] = vv

    if mirror_view:
        for a, b in SWAP_PAIRS:
            if a < len(pts) and b < len(pts):
                pts[a], pts[b] = pts[b], pts[a]
                vis[a], vis[b] = vis[b], vis[a]
                tmp = pts_norm[a, :].copy()
                pts_norm[a, :] = pts_norm[b, :]
                pts_norm[b, :] = tmp
                tv = float(vis_arr[a])
                vis_arr[a] = float(vis_arr[b])
                vis_arr[b] = tv

    def angle_if(a, b, c):
        if vis_ok(vis.get(a)) and vis_ok(vis.get(b)) and vis_ok(vis.get(c)):
            return clamp(angle_deg(pts[a], pts[b], pts[c]), ANGLE_MIN, ANGLE_MAX)
        return np.nan

    def ankle_angle(side):
        if side == "L":
            knee, ankle, heel, toe = L_KNEE, L_ANKLE, L_HEEL, L_FOOT_INDEX
        else:
            knee, ankle, heel, toe = R_KNEE, R_ANKLE, R_HEEL, R_FOOT_INDEX
        if not (vis_ok(vis.get(knee)) and vis_ok(vis.get(ankle)) and vis_ok(vis.get(heel)) and vis_ok(vis.get(toe))):
            return np.nan
        foot = pts[toe] - pts[heel]
        shin = pts[ankle] - pts[knee]
        if float(np.linalg.norm(foot)) < 1e-6 or float(np.linalg.norm(shin)) < 1e-6:
            return np.nan
        return clamp(angle_deg(pts[heel], pts[heel] + foot, pts[heel] + shin), ANGLE_MIN, ANGLE_MAX)

    L_hip = angle_if(L_SHOULDER, L_HIP, L_KNEE)
    R_hip = angle_if(R_SHOULDER, R_HIP, R_KNEE)
    L_knee = angle_if(L_HIP, L_KNEE, L_ANKLE)
    R_knee = angle_if(R_HIP, R_KNEE, R_ANKLE)
    L_sho = angle_if(L_ELBOW, L_SHOULDER, L_HIP)
    R_sho = angle_if(R_ELBOW, R_SHOULDER, R_HIP)
    L_elb = angle_if(L_SHOULDER, L_ELBOW, L_WRIST)
    R_elb = angle_if(R_SHOULDER, R_ELBOW, R_WRIST)
    L_ank = ankle_angle("L")
    R_ank = ankle_angle("R")

    vals["Hip L"] = float(L_hip)
    vals["Hip R"] = float(R_hip)
    vals["Knee L"] = float(L_knee)
    vals["Knee R"] = float(R_knee)
    vals["Ank L"] = float(L_ank)
    vals["Ank R"] = float(R_ank)
    vals["Shoulder L"] = float(L_sho)
    vals["Shoulder R"] = float(R_sho)
    vals["Elbow L"] = float(L_elb)
    vals["Elbow R"] = float(R_elb)

    return pts, vis, pts_norm, vis_arr, vals, pts_norm.copy(), vis_arr.copy()
