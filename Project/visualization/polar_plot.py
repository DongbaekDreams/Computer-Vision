"""Polar plot visualization for angle series."""

import math

import cv2
import numpy as np

from config import PLOT_AXIS, PLOT_BG, PLOT_RING


def _colormap_plasma_u8(t01_u8):
    ramp = t01_u8.reshape(-1, 1)
    ramp3 = cv2.applyColorMap(ramp, cv2.COLORMAP_PLASMA)
    return ramp3.reshape(-1, 3)


def draw_polar_plot_segment(canvas_bgr, seg_series_dict, play_idx, title="Angles (Polar)"):
    h, w = canvas_bgr.shape[:2]
    canvas_bgr[:] = PLOT_BG
    cv2.putText(canvas_bgr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (235, 235, 235), 2, cv2.LINE_AA)

    if not seg_series_dict:
        cv2.putText(canvas_bgr, "no segment", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
        return

    any_series = next(iter(seg_series_dict.values()))
    n = int(any_series.shape[0])
    if n <= 2:
        cv2.putText(canvas_bgr, "segment too short", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
        return

    cx = w // 2
    cy = h // 2 + 10
    R = int(min(w, h) * 0.40)
    inner = int(R * 0.25)
    outer = R

    for rr, a in [(inner, 0.45), (int((inner + outer) * 0.5), 0.30), (outer, 0.45)]:
        col = tuple(int(PLOT_RING[i] * (0.6 + 0.8 * a)) for i in range(3))
        cv2.circle(canvas_bgr, (cx, cy), rr, col, 1, cv2.LINE_AA)

    cv2.line(canvas_bgr, (cx - outer, cy), (cx + outer, cy), PLOT_AXIS, 1, cv2.LINE_AA)
    cv2.line(canvas_bgr, (cx, cy - outer), (cx, cy + outer), PLOT_AXIS, 1, cv2.LINE_AA)

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)

    for _, seg in seg_series_dict.items():
        ok = np.isfinite(seg)
        if ok.sum() < 2:
            continue

        v = seg[ok]
        vmin, vmax = float(np.min(v)), float(np.max(v))
        denom = (vmax - vmin) + 1e-12

        r = np.empty((n,), dtype=np.float32)
        r[:] = np.nan
        r_ok = (seg - vmin) / denom
        r_ok = np.clip(r_ok, 0.0, 1.0)
        r[ok] = inner + r_ok[ok] * (outer - inner)

        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)

        t_u8 = np.zeros((n,), dtype=np.uint8)
        t_u8[ok] = (255.0 * r_ok[ok]).astype(np.uint8)
        cols = _colormap_plasma_u8(t_u8)

        for i in range(n - 1):
            if not (np.isfinite(xs[i]) and np.isfinite(ys[i]) and np.isfinite(xs[i + 1]) and np.isfinite(ys[i + 1])):
                continue
            c = tuple(int(x) for x in cols[i])
            c = (int(c[0] * 0.85), int(c[1] * 0.85), int(c[2] * 0.85))
            cv2.line(canvas_bgr, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])), c, 2, cv2.LINE_AA)

    pi = int(np.clip(play_idx, 0, n - 1))
    ang = float(theta[pi])
    dot_r = int((inner + outer) * 0.5)
    dx = int(cx + dot_r * math.cos(ang))
    dy = int(cy + dot_r * math.sin(ang))
    cv2.circle(canvas_bgr, (dx, dy), 6, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(canvas_bgr, (dx, dy), 10, (120, 120, 120), 2, cv2.LINE_AA)
    cv2.circle(canvas_bgr, (cx, cy), 3, (220, 220, 220), -1, cv2.LINE_AA)
