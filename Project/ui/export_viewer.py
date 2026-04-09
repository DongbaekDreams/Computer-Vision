"""Polar export viewer: letterboxed image + caption, sized to fit the dashboard stage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import (
    EXPORT_VIEWER_CAPTION_H,
    PANEL_FONT,
    SURFACE_BG_ALT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WINDOW_BG,
)

# waitKeyEx codes: Linux/mac 65361/65363; Windows often 2424832 / 2555904
ARROW_LEFT_KEYS = frozenset({65361, 2424832})
ARROW_RIGHT_KEYS = frozenset({65363, 2555904})


def _compose_export_viewer_canvas(
    path: Path,
    index: int,
    total: int,
    max_canvas_w: int,
    max_canvas_h: int,
) -> tuple[np.ndarray, int, int]:
    """
    Letterboxed BGR canvas that fits inside max_canvas_w x max_canvas_h.
    Returns (canvas, footer_y0, width); y < footer_y0 is the image area.
    """
    max_canvas_w = max(1, int(max_canvas_w))
    max_canvas_h = max(1, int(max_canvas_h))
    cap_h = min(int(EXPORT_VIEWER_CAPTION_H), max(28, max_canvas_h // 9))
    cap_h = max(24, min(cap_h, max_canvas_h // 3))
    avail_h = max(60, max_canvas_h - cap_h)
    max_w = max_canvas_w
    max_h = avail_h

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((min(480, max_h), min(720, max_w), 3), dtype=np.uint8)
        img[:] = SURFACE_BG_ALT
        cv2.putText(
            img,
            "Could not read image",
            (24, img.shape[0] // 2),
            PANEL_FONT,
            0.8,
            TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    h, w = img.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas_h = nh + cap_h
    canvas_w = nw
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = WINDOW_BG
    canvas[0:nh, 0:nw] = resized

    footer_y0 = nh
    band = canvas[footer_y0:, :]
    band[:] = SURFACE_BG_ALT

    label = f"{path.name}   ({index + 1} / {total})"
    cv2.putText(
        band,
        label,
        (12, 22),
        PANEL_FONT,
        0.52,
        TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        band,
        "Arrows or click left/right third of image   |   q / Esc close",
        (12, 40),
        PANEL_FONT,
        0.38,
        TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    return canvas, footer_y0, canvas_w


def draw_export_viewer_in_stage(
    img: np.ndarray,
    stage_bounds: tuple[int, int, int, int],
    path: Path,
    index: int,
    total: int,
) -> tuple[int, int, int, int]:
    """
    Blit a centered viewer into the stage region. Returns (x0,y0,x1,y1) of the
    image area only (excluding caption), in full-frame coordinates.
    """
    sx0, sy0, sx1, sy1 = stage_bounds
    sw = max(1, sx1 - sx0)
    sh = max(1, sy1 - sy0)
    canvas, footer_y0, cw = _compose_export_viewer_canvas(path, index, total, sw, sh)
    ch = canvas.shape[0]
    px0 = sx0 + max(0, (sw - cw) // 2)
    py0 = sy0 + max(0, (sh - ch) // 2)
    px1 = min(sx1, px0 + cw)
    py1 = min(sy1, py0 + ch)
    cw_eff = px1 - px0
    ch_eff = py1 - py0
    if cw_eff < 1 or ch_eff < 1:
        return (px0, py0, px0, py0)
    src = canvas[0:ch_eff, 0:cw_eff]
    img[py0:py1, px0:px1] = src
    foot_clip = max(0, min(footer_y0, ch_eff))
    return (px0, py0, px0 + cw_eff, py0 + foot_clip)


def export_viewer_step_index(key_ex: int) -> int | None:
    """Map waitKeyEx code to -1 (prev), +1 (next), or None."""
    if key_ex in ARROW_LEFT_KEYS:
        return -1
    if key_ex in ARROW_RIGHT_KEYS:
        return 1
    lo = key_ex & 0xFF
    if lo == ord("[") or lo == ord(","):
        return -1
    if lo == ord("]") or lo == ord("."):
        return 1
    return None
