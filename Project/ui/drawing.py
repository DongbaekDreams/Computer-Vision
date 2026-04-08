"""UI drawing: panels, buttons, stat boxes, tables."""

import cv2
import numpy as np

from config import (
    ACCENT,
    BTN_BG,
    BTN_BORDER,
    BTN_TEXT,
    CARD_PAD,
    CONTROL_KEY_FONT,
    CONTROL_KEY_SCALE,
    KEY_HELP_HEAD_H,
    KEY_HELP_ROW_H,
    PANEL_CONTROLS_SECTION_GAP,
    PANEL_BG,
    PANEL_DIVIDER,
    PANEL_HEADER_BG,
    PANEL_FONT,
    PANEL_TEXT_THICK,
    PANEL_TITLE_THICK,
    SECTION_TITLE_SCALE,
    SEG_SECONDS,
    STAT_LABEL_SCALE,
    STAT_VALUE_SCALE,
    SUBTITLE_SCALE,
    SURFACE_BG,
    SURFACE_BG_ALT,
    SURFACE_BORDER,
    TABLE_LABEL_SCALE,
    TABLE_TITLE_SCALE,
    TABLE_VALUE_SCALE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TITLE_SCALE,
    UI_SCALE,
    UI_SCALE_SMALL,
    VIDEO_BG,
    VIDEO_MAX_SCALE,
    VIDEO_SCALE,
)


def put_text_deg(img, value_int_or_none, org, font, scale, color, thickness, lineType=cv2.LINE_AA):
    x, y = org
    if value_int_or_none is None:
        cv2.putText(img, "--", (x, y), font, scale, color, thickness, lineType)
        return
    s = str(int(value_int_or_none))
    (tw, th), _ = cv2.getTextSize(s, font, scale, thickness)
    cv2.putText(img, s, (x, y), font, scale, color, thickness, lineType)
    cx = x + tw + max(4, int(0.12 * th))
    cy = y - int(0.70 * th)
    r = max(2, int(0.12 * th))
    cv2.circle(img, (cx, cy), r, color, thickness, lineType)


def panel_bg(h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = PANEL_BG
    return img


def _draw_card(img, x, y, w, h, fill=SURFACE_BG, border=SURFACE_BORDER, accent=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)
    if accent is not None:
        cv2.rectangle(img, (x, y), (x + 4, y + h), accent, -1)


def draw_panel_header(panel, title, subtitle=None):
    w = panel.shape[1]
    header_h = 76 if subtitle else 62
    cv2.rectangle(panel, (0, 0), (w - 1, header_h), PANEL_HEADER_BG, -1)
    cv2.rectangle(panel, (0, header_h - 5), (w - 1, header_h), ACCENT, -1)
    cv2.line(panel, (0, header_h), (w - 1, header_h), PANEL_DIVIDER, 1)
    cv2.putText(
        panel,
        title,
        (18, 40),
        PANEL_FONT,
        TITLE_SCALE,
        TEXT_PRIMARY,
        PANEL_TITLE_THICK,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            panel,
            subtitle,
            (18, 62),
            PANEL_FONT,
            SUBTITLE_SCALE,
            TEXT_SECONDARY,
            PANEL_TEXT_THICK,
            cv2.LINE_AA,
        )
    return header_h


def draw_stat_box(panel, x, y, w, h, label, value):
    _draw_card(panel, x, y, w, h, fill=SURFACE_BG, border=SURFACE_BORDER, accent=ACCENT)
    cv2.putText(
        panel,
        label.upper(),
        (x + CARD_PAD, y + 24),
        PANEL_FONT,
        STAT_LABEL_SCALE,
        TEXT_SECONDARY,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        value,
        (x + CARD_PAD, y + 60),
        PANEL_FONT,
        STAT_VALUE_SCALE,
        TEXT_PRIMARY,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )


def draw_lr_table(panel, x, y, w, row_h, title, rows):
    hh = row_h * (len(rows) + 1) + 6
    _draw_card(panel, x, y, w, hh, fill=SURFACE_BG, border=SURFACE_BORDER)
    cv2.putText(
        panel,
        title,
        (x + CARD_PAD, y + 28),
        PANEL_FONT,
        TABLE_TITLE_SCALE,
        TEXT_PRIMARY,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    header_y = y + row_h
    cv2.line(panel, (x + 1, header_y), (x + w - 1, header_y), SURFACE_BORDER, 1)
    col_label = x + CARD_PAD
    col_L = x + int(w * 0.62)
    col_R = x + int(w * 0.82)
    cv2.putText(
        panel,
        "L",
        (col_L, y + row_h - 12),
        PANEL_FONT,
        TABLE_LABEL_SCALE,
        TEXT_SECONDARY,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "R",
        (col_R, y + row_h - 12),
        PANEL_FONT,
        TABLE_LABEL_SCALE,
        TEXT_SECONDARY,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    for i, (lab, lv, rv) in enumerate(rows):
        yy = y + row_h * (i + 1) + 10
        if i > 0:
            cv2.line(panel, (x + CARD_PAD, yy - 6), (x + w - CARD_PAD, yy - 6), SURFACE_BG_ALT, 1)
        cv2.putText(
            panel,
            lab,
            (col_label, yy + 18),
            PANEL_FONT,
            TABLE_LABEL_SCALE,
            TEXT_SECONDARY,
            PANEL_TEXT_THICK,
            cv2.LINE_AA,
        )
        put_text_deg(
            panel, lv, (col_L, yy + 22), PANEL_FONT, TABLE_VALUE_SCALE, TEXT_PRIMARY, PANEL_TEXT_THICK
        )
        put_text_deg(
            panel, rv, (col_R, yy + 22), PANEL_FONT, TABLE_VALUE_SCALE, TEXT_PRIMARY, PANEL_TEXT_THICK
        )
    return y + hh


def draw_section_title(panel, x, y, w, title, expanded):
    h = 38
    _draw_card(panel, x, y, w, h, fill=SURFACE_BG_ALT, border=SURFACE_BORDER)
    tri = "v" if expanded else ">"
    cv2.putText(
        panel,
        f"{tri} {title}",
        (x + CARD_PAD, y + 26),
        PANEL_FONT,
        SECTION_TITLE_SCALE,
        TEXT_PRIMARY,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    return y + h


def _fit_desc_text(text: str, font, scale: float, thick: int, max_w: int) -> str:
    """Truncate description so putText stays within max_w pixels (OpenCV has no wrap)."""
    if max_w < 24:
        return ""
    t = text.strip()
    while True:
        (tw, _), _ = cv2.getTextSize(t, font, scale, thick)
        if tw <= max_w:
            return t
        if len(t) <= 4:
            return "..."
        t = t[:-2] + ".."


def _draw_key_help_block(
    panel,
    x: int,
    y: int,
    w: int,
    subtitle: str,
    rows: list[tuple[str, str]],
) -> int:
    """One titled key list; returns y after the block."""
    font = PANEL_FONT
    row_h = KEY_HELP_ROW_H
    head_h = KEY_HELP_HEAD_H
    pad = CARD_PAD
    box_h = head_h + len(rows) * row_h + 6
    _draw_card(panel, x, y, w, box_h, fill=SURFACE_BG, border=SURFACE_BORDER)
    cv2.putText(
        panel,
        subtitle,
        (x + pad, y + 18),
        font,
        SECTION_TITLE_SCALE * 0.78,
        TEXT_PRIMARY,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    thick = PANEL_TEXT_THICK
    desc_scale = 0.42
    key_col_w = 94
    desc_x0 = x + pad + key_col_w
    max_desc_w = max(40, w - key_col_w - 2 * pad)
    yy = y + head_h + 12
    for k, d in rows:
        cv2.putText(
            panel,
            k,
            (x + pad, yy),
            CONTROL_KEY_FONT,
            CONTROL_KEY_SCALE,
            TEXT_PRIMARY,
            PANEL_TEXT_THICK,
            cv2.LINE_AA,
        )
        desc = _fit_desc_text(d, font, desc_scale, thick, max_desc_w)
        cv2.putText(
            panel,
            desc,
            (desc_x0, yy),
            font,
            desc_scale,
            TEXT_SECONDARY,
            thick,
            cv2.LINE_AA,
        )
        yy += row_h
    return y + box_h


def draw_controls_section(panel, x, y, w, expanded):
    y = draw_section_title(panel, x, y, w, "Controls", expanded)
    if not expanded:
        cv2.putText(
            panel,
            "? toggle expanded key help (not a camera key)",
            (x + CARD_PAD, y + 22),
            PANEL_FONT,
            UI_SCALE_SMALL,
            TEXT_MUTED,
            PANEL_TEXT_THICK,
            cv2.LINE_AA,
        )
        return y + 36
    # OpenCV putText: ASCII only (no Unicode).
    y += 4
    gap = PANEL_CONTROLS_SECTION_GAP
    y = _draw_key_help_block(
        panel,
        x,
        y,
        w,
        "Keys: panel & video",
        [
            ("u", "toggle left panel"),
            ("v", "toggle camera background"),
            ("d", "2+ cams: dual view vertical stack"),
        ],
    )
    y += gap
    y = _draw_key_help_block(
        panel,
        x,
        y,
        w,
        "Keys: skeleton & debug",
        [
            ("s", "toggle skeleton"),
            ("j", "toggle joint dots"),
            ("p", "toggle visibility text"),
            ("l", "toggle console log"),
        ],
    )
    y += gap
    y = _draw_key_help_block(
        panel,
        x,
        y,
        w,
        "Keys: polar plot & exports",
        [
            ("a", "toggle polar plot + clip strip"),
            ("g", "palette gallery"),
            ("o", "export polar PNG (exact file name)"),
            ("e", "export gallery (click image: viewer)"),
            ("esc", "close palette / export gallery"),
        ],
    )
    y += gap
    y = _draw_key_help_block(
        panel,
        x,
        y,
        w,
        "Keys: multi-cam overlay align",
        [
            ("k", "cycle primary (best cam -> main pane; saved)"),
            ("config", "2+ USB cams glitch: MULTICAM_* in config.py"),
            ("[ / ]", "primary overlay x -/+"),
            ("; / '", "secondary overlay x -/+"),
        ],
    )
    y += gap
    y = _draw_key_help_block(
        panel,
        x,
        y,
        w,
        "Keys: session & timeline",
        [
            ("q", "quit app"),
            ("b", "camera baseline (meters)"),
            ("m", "body profile editor (all segments, popup)"),
            ("?", "collapse / expand this key list"),
            ("mouse", "LIVE / REVIEW / REC"),
            (
                "mouse",
                f"review: drag {int(SEG_SECONDS)}s window",
            ),
        ],
    )
    return y + 6


def fit_video_to_pane(frame, pane_w, pane_h):
    h, w = frame.shape[:2]
    scale = min(pane_w / max(1, w), pane_h / max(1, h))
    scale = min(scale, VIDEO_MAX_SCALE) * float(VIDEO_SCALE)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
    canvas[:] = VIDEO_BG
    x0 = (pane_w - nw) // 2
    y0 = (pane_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def draw_button(img, rect, label, fill, text_col=BTN_TEXT, scale=UI_SCALE, thick=1):
    x0, y0, x1, y1 = rect
    cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), BTN_BORDER, 1)
    cv2.line(img, (x0 + 1, y0 + 1), (x1 - 1, y0 + 1), SURFACE_BORDER, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, PANEL_FONT, scale, thick)
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 + th) // 2 - 1
    cv2.putText(img, label, (tx, ty), PANEL_FONT, scale, text_col, thick, cv2.LINE_AA)
