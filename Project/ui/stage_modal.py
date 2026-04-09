"""Center-stage overlays: dim the video pane, center modals (no extra OS windows)."""

from __future__ import annotations

import cv2
import numpy as np


def darken_full_image(
    img: np.ndarray,
    color: tuple[int, int, int],
    weight: float = 0.52,
) -> None:
    """Dim the entire composed frame (modal backdrop over panel + plot + video)."""
    h, w = img.shape[:2]
    darken_stage_region(img, (0, 0, w, h), color, weight)


def darken_stage_region(
    img: np.ndarray,
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int],
    weight: float = 0.52,
) -> None:
    """Blend a solid layer over img[bounds] in-place (BGR)."""
    sx0, sy0, sx1, sy1 = bounds
    sx0 = max(0, int(sx0))
    sy0 = max(0, int(sy0))
    sx1 = min(img.shape[1], int(sx1))
    sy1 = min(img.shape[0], int(sy1))
    if sx1 <= sx0 or sy1 <= sy0:
        return
    roi = img[sy0:sy1, sx0:sx1]
    layer = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(layer, weight, roi, 1.0 - weight, 0.0, dst=roi)


def center_box_in_bounds(
    box_w: int,
    box_h: int,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return (x0,y0,x1,y1) for a box centered in bounds, clamped inside."""
    sx0, sy0, sx1, sy1 = bounds
    sw = max(1, sx1 - sx0)
    sh = max(1, sy1 - sy0)
    bw = min(box_w, sw - 8)
    bh = min(box_h, sh - 8)
    bw = max(1, bw)
    bh = max(1, bh)
    mx0 = sx0 + (sw - bw) // 2
    my0 = sy0 + (sh - bh) // 2
    mx1 = mx0 + bw
    my1 = my0 + bh
    return mx0, my0, mx1, my1


def compute_video_stage_bounds(
    view_w: int,
    view_h: int,
    *,
    panel_w_eff: int,
    plot_enabled: bool,
    plot_w: int,
    plot_pad: int,
    video_pad: int,
    min_video_w: int = 280,
) -> tuple[int, int, int, int]:
    """
    Pixel rect (x0, y0, x1, y1) of the main video column inside the dashboard
    (between panel and polar strip when plot is on).
    """
    pane_w = view_w - panel_w_eff
    if plot_enabled:
        video_area_w = max(min_video_w, pane_w - (plot_w + 2 * plot_pad))
    else:
        video_area_w = pane_w
    pane_inner_w = max(1, video_area_w - 2 * video_pad)
    sx0 = panel_w_eff + video_pad
    sy0 = video_pad
    sx1 = sx0 + pane_inner_w
    sy1 = view_h - video_pad
    return sx0, sy0, sx1, sy1
