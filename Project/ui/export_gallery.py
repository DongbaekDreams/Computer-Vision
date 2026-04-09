"""Scrollable modal gallery for saved polar export PNGs (dashboard overlay)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from config import (
    ACCENT,
    EXPORT_GALLERY_CARD_H,
    EXPORT_GALLERY_CARD_W,
    EXPORT_GALLERY_COLS,
    EXPORT_GALLERY_FOOT_H,
    EXPORT_GALLERY_GAP,
    EXPORT_GALLERY_INNER_PAD,
    EXPORT_GALLERY_SCROLLBAR_W,
    EXPORT_GALLERY_SUBTITLE,
    EXPORT_GALLERY_TITLE,
    EXPORT_GALLERY_TITLE_H,
    PALETTE_GALLERY_BG,
    PALETTE_GALLERY_BORDER,
    PALETTE_GALLERY_PREVIEW_BG,
    PALETTE_GALLERY_SCRIM,
    PALETTE_GALLERY_TEXT,
    PALETTE_GALLERY_TEXT_MUTED,
    PANEL_FONT,
    PANEL_TEXT_THICK,
    PANEL_TITLE_THICK,
    SURFACE_BG_ALT,
    SURFACE_BORDER_SOFT,
    SURFACE_ELEVATED,
)
from ui.stage_modal import darken_full_image

INNER_PAD = EXPORT_GALLERY_INNER_PAD

_THUMB_CACHE: dict[tuple[str, int], np.ndarray] = {}
_THUMB_CACHE_MAX = 64


def polar_exports_dir(project_root: Path) -> Path:
    return project_root / "polar_exports"


def list_polar_export_files(export_dir: Path) -> list[Path]:
    if not export_dir.is_dir():
        return []
    out = [p for p in export_dir.iterdir() if p.suffix.lower() == ".png" and p.is_file()]
    try:
        out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        out.sort(key=lambda p: p.name, reverse=True)
    return out


def _truncate_label(name: str, max_chars: int) -> str:
    if len(name) <= max_chars:
        return name
    keep = max_chars - 3
    a = max(keep // 2, 1)
    b = keep - a
    return f"{name[:a]}...{name[-b:]}"


def _thumb_for_path(path: Path, tw: int, th: int) -> np.ndarray | None:
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()), int(st.st_mtime_ns))
    cached = _THUMB_CACHE.get(key)
    if cached is not None and cached.shape[0] == th and cached.shape[1] == tw:
        return cached

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    scale = min(tw / max(1, w), th / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[:] = PALETTE_GALLERY_PREVIEW_BG
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

    if len(_THUMB_CACHE) >= _THUMB_CACHE_MAX:
        _THUMB_CACHE.clear()
    _THUMB_CACHE[key] = canvas
    return canvas


def max_scroll_rows(num_items: int, cols: int, rows_visible: int) -> int:
    if num_items <= 0 or cols <= 0:
        return 0
    num_rows = int(math.ceil(num_items / float(cols)))
    return max(0, num_rows - rows_visible)


def draw_exports_gallery(
    img: np.ndarray,
    *,
    stage_bounds: tuple[int, int, int, int],
    export_paths: list[Path],
    scroll_row: int,
) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int, Path]], int]:
    """
    Dim the video stage and draw a centered gallery. Returns modal_rect, hitboxes
    (x0,y0,x1,y1, path), and max_scroll_row for clamping.
    """
    sx0, sy0, sx1, sy1 = stage_bounds
    sw = max(1, sx1 - sx0)
    sh = max(1, sy1 - sy0)
    iw, ih = sw, sh
    cols = max(1, int(EXPORT_GALLERY_COLS))
    card_w = int(EXPORT_GALLERY_CARD_W)
    card_h = int(EXPORT_GALLERY_CARD_H)
    gap = int(EXPORT_GALLERY_GAP)
    inner_pad = int(INNER_PAD)
    title_h = int(EXPORT_GALLERY_TITLE_H)
    foot_h = int(EXPORT_GALLERY_FOOT_H)
    sb_w = int(EXPORT_GALLERY_SCROLLBAR_W)

    row_stride = card_h + gap
    grid_w = cols * card_w + (cols - 1) * gap

    margin = 20
    max_modal_h = max(220, ih - margin)
    inner_h_budget = max_modal_h - title_h - foot_h - 2 * inner_pad
    rows_visible = max(1, inner_h_budget // row_stride)
    viewport_h = rows_visible * row_stride - gap
    if viewport_h < card_h:
        viewport_h = card_h
        rows_visible = 1

    modal_inner_w = grid_w
    modal_w = inner_pad * 2 + modal_inner_w + sb_w + 6
    modal_h = title_h + foot_h + 2 * inner_pad + viewport_h

    modal_x0 = sx0 + max(margin // 2, (iw - modal_w) // 2)
    modal_y0 = sy0 + max(margin // 2, (ih - modal_h) // 2)
    modal_x1 = min(sx0 + iw - 8, modal_x0 + modal_w)
    modal_y1 = min(sy0 + ih - 8, modal_y0 + modal_h)
    if modal_x1 - modal_x0 < modal_w:
        modal_x0 = max(sx0 + 8, sx0 + iw - modal_w - 8)
        modal_x1 = modal_x0 + modal_w
    if modal_y1 - modal_y0 < modal_h:
        modal_y0 = max(sy0 + 8, sy0 + ih - modal_h - 8)
        modal_y1 = modal_y0 + modal_h

    darken_full_image(img, PALETTE_GALLERY_SCRIM, 0.56)

    cv2.rectangle(img, (modal_x0, modal_y0), (modal_x1, modal_y1), PALETTE_GALLERY_BG, -1)
    cv2.rectangle(img, (modal_x0, modal_y0), (modal_x1, modal_y1), PALETTE_GALLERY_BORDER, 2)

    cv2.putText(
        img,
        EXPORT_GALLERY_TITLE,
        (modal_x0 + inner_pad, modal_y0 + 26),
        PANEL_FONT,
        0.74,
        PALETTE_GALLERY_TEXT,
        PANEL_TITLE_THICK,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        EXPORT_GALLERY_SUBTITLE,
        (modal_x0 + inner_pad, modal_y0 + 48),
        PANEL_FONT,
        0.44,
        PALETTE_GALLERY_TEXT_MUTED,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )

    n = len(export_paths)
    max_row = max_scroll_rows(n, cols, rows_visible)
    sr = int(np.clip(scroll_row, 0, max_row))

    vx0 = modal_x0 + inner_pad
    vy0 = modal_y0 + title_h + inner_pad
    vx1 = vx0 + modal_inner_w
    vy1 = vy0 + viewport_h

    # Viewport background
    cv2.rectangle(img, (vx0, vy0), (vx1, vy1), PALETTE_GALLERY_PREVIEW_BG, -1)
    cv2.rectangle(img, (vx0, vy0), (vx1, vy1), PALETTE_GALLERY_BORDER, 1)

    thumb_h = max(48, card_h - 36)
    thumb_w = max(48, card_w - 14)
    hitboxes: list[tuple[int, int, int, int, Path]] = []

    if n == 0:
        cv2.putText(
            img,
            "No PNGs yet - press o to export the polar plot",
            (vx0 + 14, vy0 + viewport_h // 2),
            PANEL_FONT,
            0.52,
            PALETTE_GALLERY_TEXT_MUTED,
            PANEL_TEXT_THICK,
            cv2.LINE_AA,
        )
    else:
        first_idx = sr * cols
        for slot in range(rows_visible * cols):
            i = first_idx + slot
            if i >= n:
                break
            path = export_paths[i]
            row = slot // cols
            col = slot % cols
            x0 = vx0 + col * (card_w + gap)
            y0 = vy0 + row * row_stride
            x1 = x0 + card_w
            y1 = y0 + card_h
            if y1 > vy1 + 2:
                break

            cv2.rectangle(img, (x0, y0), (x1, y1), PALETTE_GALLERY_PREVIEW_BG, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), PALETTE_GALLERY_BORDER, 1)

            tx = x0 + 7
            ty_thumb = y0 + 6
            thumb = _thumb_for_path(path, thumb_w, thumb_h)
            if thumb is not None:
                txi = tx
                tyi = ty_thumb
                img[tyi : tyi + thumb_h, txi : txi + thumb_w] = thumb
            else:
                cv2.rectangle(
                    img,
                    (tx, ty_thumb),
                    (tx + thumb_w, ty_thumb + thumb_h),
                    SURFACE_ELEVATED,
                    -1,
                )
                cv2.putText(
                    img,
                    "read err",
                    (tx + 8, ty_thumb + thumb_h // 2),
                    PANEL_FONT,
                    0.42,
                    PALETTE_GALLERY_TEXT_MUTED,
                    PANEL_TEXT_THICK,
                    cv2.LINE_AA,
                )

            label = _truncate_label(path.name, 22)
            ly = y0 + card_h - 22
            cv2.putText(
                img,
                label,
                (x0 + 8, ly),
                PANEL_FONT,
                0.42,
                PALETTE_GALLERY_TEXT,
                PANEL_TEXT_THICK,
                cv2.LINE_AA,
            )

            hitboxes.append((x0, y0, x1, y1, path))

    # Scrollbar
    if max_row > 0:
        sx0 = modal_x1 - inner_pad - sb_w - 4
        sy0 = vy0 + 4
        sy1 = vy1 - 4
        cv2.rectangle(img, (sx0, sy0), (sx0 + sb_w, sy1), SURFACE_BG_ALT, -1)
        cv2.rectangle(img, (sx0, sy0), (sx0 + sb_w, sy1), PALETTE_GALLERY_BORDER, 1)
        track_h = sy1 - sy0
        thumb_h_bar = max(18, int(track_h * (rows_visible / (max_row + rows_visible))))
        t0 = sy0 + int((track_h - thumb_h_bar) * (sr / max(1, max_row)))
        t1 = min(sy1, t0 + thumb_h_bar)
        thumb_col = tuple(
            int(0.42 * base + 0.58 * acc)
            for base, acc in zip(SURFACE_BORDER_SOFT, ACCENT)
        )
        cv2.rectangle(img, (sx0 + 1, t0), (sx0 + sb_w - 1, t1), thumb_col, -1)

    foot_y = modal_y1 - foot_h + 8
    cv2.putText(
        img,
        f"{n} file(s)   polar_exports/",
        (modal_x0 + inner_pad, foot_y),
        PANEL_FONT,
        0.38,
        PALETTE_GALLERY_TEXT_MUTED,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )

    return (modal_x0, modal_y0, modal_x1, modal_y1), hitboxes, max_row
