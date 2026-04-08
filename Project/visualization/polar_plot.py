"""Polar plot visualization for angle series."""

import math
from pathlib import Path

import cv2
import numpy as np

from config import (
    DEFAULT_POLAR_PALETTE_KEY,
    PANEL_FONT,
    PANEL_TEXT_THICK,
    POLAR_EXPORT_H,
    POLAR_EXPORT_W,
    TEXT_MUTED,
    TEXT_PRIMARY,
    get_polar_palette,
)


def _palette_ramp_u8(t01_u8, palette):
    ramp = np.asarray(t01_u8, dtype=np.uint8).reshape(-1, 1)

    if palette.get("type") == "opencv":
        ramp3 = cv2.applyColorMap(ramp, int(palette["colormap"]))
        return ramp3.reshape(-1, 3)

    colors = np.asarray(palette.get("colors", []), dtype=np.float32)
    if len(colors) == 0:
        return np.repeat(ramp[:, :, None], 3, axis=2).reshape(-1, 3)
    if len(colors) == 1:
        return np.repeat(colors.astype(np.uint8), len(ramp), axis=0)

    u = ramp[:, 0].astype(np.float32) / 255.0
    pos = u * (len(colors) - 1)
    idx0 = np.floor(pos).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, len(colors) - 1)
    frac = (pos - idx0).reshape(-1, 1)
    out = colors[idx0] * (1.0 - frac) + colors[idx1] * frac
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_palette_preview(canvas_bgr, palette):
    h, w = canvas_bgr.shape[:2]
    canvas_bgr[:] = palette["bg"]
    cx = w // 2
    cy = h // 2
    outer = max(12, min(w, h) // 2 - 8)
    inner = max(6, int(outer * 0.42))
    ring = tuple(int(c) for c in palette["ring"])
    axis = tuple(int(c) for c in palette["axis"])
    cv2.circle(canvas_bgr, (cx, cy), outer, ring, 1, cv2.LINE_AA)
    cv2.circle(canvas_bgr, (cx, cy), inner, ring, 1, cv2.LINE_AA)
    cv2.line(canvas_bgr, (cx - outer, cy), (cx + outer, cy), axis, 1, cv2.LINE_AA)
    cv2.line(canvas_bgr, (cx, cy - outer), (cx, cy + outer), axis, 1, cv2.LINE_AA)

    theta = np.linspace(-0.20 * np.pi, 1.45 * np.pi, 32).astype(np.float32)
    r01 = np.linspace(0.08, 0.92, len(theta)).astype(np.float32)
    rs = inner + r01 * (outer - inner)
    xs = cx + rs * np.cos(theta)
    ys = cy + rs * np.sin(theta)
    cols = _palette_ramp_u8((r01 * 255.0).astype(np.uint8), palette)
    for i in range(len(theta) - 1):
        color = tuple(int(v) for v in cols[i])
        cv2.line(canvas_bgr, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])), color, 2, cv2.LINE_AA)

    marker = tuple(int(c) for c in palette["marker"])
    marker_ring = tuple(int(c) for c in palette["marker_ring"])
    cv2.circle(canvas_bgr, (int(xs[-1]), int(ys[-1])), 4, marker, -1, cv2.LINE_AA)
    cv2.circle(canvas_bgr, (int(xs[-1]), int(ys[-1])), 7, marker_ring, 1, cv2.LINE_AA)


def draw_polar_plot_segment(
    canvas_bgr,
    seg_series_dict,
    play_idx,
    title="Angles (Polar)",
    palette_key=DEFAULT_POLAR_PALETTE_KEY,
    *,
    show_playhead: bool = True,
):
    h, w = canvas_bgr.shape[:2]
    palette = get_polar_palette(palette_key)
    plot_bg = tuple(int(c) for c in palette["bg"])
    plot_ring = tuple(int(c) for c in palette["ring"])
    plot_axis = tuple(int(c) for c in palette["axis"])
    marker = tuple(int(c) for c in palette["marker"])
    marker_ring = tuple(int(c) for c in palette["marker_ring"])
    center = tuple(int(c) for c in palette["center"])

    # Scale strokes/fonts with canvas size so dashboard + high-res exports both look right.
    s = max(0.65, min(float(h), float(w)) / 420.0)
    ox = int(round(12 * s))
    title_y = int(round(30 * s))
    sub_y = int(round(52 * s))
    msg_y = int(round(76 * s))
    title_font = 0.74 * s
    sub_font = 0.48 * s
    title_th = PANEL_TEXT_THICK
    sub_th = PANEL_TEXT_THICK
    ring_th = max(1, int(round(1 * s)))
    axis_th = max(1, int(round(1 * s)))
    line_th = max(1, int(round(2 * s)))
    dot_fill_r = max(4, int(round(6 * s)))
    dot_ring_r = max(6, int(round(10 * s)))
    dot_ring_th = max(1, int(round(2 * s)))
    center_r = max(2, int(round(3 * s)))

    canvas_bgr[:] = plot_bg
    cv2.putText(
        canvas_bgr,
        title,
        (ox, title_y),
        PANEL_FONT,
        title_font,
        TEXT_PRIMARY,
        title_th,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas_bgr,
        palette["label"],
        (ox, sub_y),
        PANEL_FONT,
        sub_font,
        TEXT_MUTED,
        sub_th,
        cv2.LINE_AA,
    )

    if not seg_series_dict:
        cv2.putText(
            canvas_bgr,
            "no segment",
            (ox, msg_y),
            PANEL_FONT,
            sub_font,
            TEXT_MUTED,
            sub_th,
            cv2.LINE_AA,
        )
        return

    any_series = next(iter(seg_series_dict.values()))
    n = int(any_series.shape[0])
    if n <= 2:
        cv2.putText(
            canvas_bgr,
            "segment too short",
            (ox, msg_y),
            PANEL_FONT,
            sub_font,
            TEXT_MUTED,
            sub_th,
            cv2.LINE_AA,
        )
        return

    cx = w // 2
    cy = h // 2 + int(round(16 * s))
    R = int(min(w, h) * 0.38)
    inner = int(R * 0.25)
    outer = R

    for rr, a in [(inner, 0.45), (int((inner + outer) * 0.5), 0.30), (outer, 0.45)]:
        col = tuple(int(plot_ring[i] * (0.6 + 0.8 * a)) for i in range(3))
        cv2.circle(canvas_bgr, (cx, cy), rr, col, ring_th, cv2.LINE_AA)

    cv2.line(canvas_bgr, (cx - outer, cy), (cx + outer, cy), plot_axis, axis_th, cv2.LINE_AA)
    cv2.line(canvas_bgr, (cx, cy - outer), (cx, cy + outer), plot_axis, axis_th, cv2.LINE_AA)

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
        cols = _palette_ramp_u8(t_u8, palette)

        for i in range(n - 1):
            if not (np.isfinite(xs[i]) and np.isfinite(ys[i]) and np.isfinite(xs[i + 1]) and np.isfinite(ys[i + 1])):
                continue
            c = tuple(int(x) for x in cols[i])
            c = (int(c[0] * 0.85), int(c[1] * 0.85), int(c[2] * 0.85))
            cv2.line(
                canvas_bgr,
                (int(xs[i]), int(ys[i])),
                (int(xs[i + 1]), int(ys[i + 1])),
                c,
                line_th,
                cv2.LINE_AA,
            )

    if show_playhead:
        pi = int(np.clip(play_idx, 0, n - 1))
        ang = float(theta[pi])
        dot_r = int((inner + outer) * 0.5)
        dx = int(cx + dot_r * math.cos(ang))
        dy = int(cy + dot_r * math.sin(ang))
        cv2.circle(canvas_bgr, (dx, dy), dot_fill_r, marker, -1, cv2.LINE_AA)
        cv2.circle(canvas_bgr, (dx, dy), dot_ring_r, marker_ring, dot_ring_th, cv2.LINE_AA)

    cv2.circle(canvas_bgr, (cx, cy), center_r, center, -1, cv2.LINE_AA)


def export_polar_plot_png(
    seg_series_dict: dict,
    palette_key: str,
    out_path: str | Path,
    *,
    title: str,
    width_px: int | None = None,
    height_px: int | None = None,
) -> Path:
    """Save the current polar segment (same style/colors as on-screen) at high resolution."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = int(POLAR_EXPORT_W if width_px is None else width_px)
    h = int(POLAR_EXPORT_H if height_px is None else height_px)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    draw_polar_plot_segment(
        canvas,
        seg_series_dict,
        0,
        title=title,
        palette_key=palette_key,
        show_playhead=False,
    )
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return out_path
