"""UI drawing: panels, buttons, stat boxes, tables."""

import cv2
import numpy as np

from config import (
    ACCENT,
    BTN_BG,
    BTN_BORDER,
    BTN_TEXT,
    CARD_PAD,
    CONTROL_DESC_SCALE,
    CONTROL_KEY_SCALE,
    CONTROL_ROW_H,
    PANEL_BG,
    PANEL_DIVIDER,
    PANEL_HEADER_BG,
    SECTION_TITLE_SCALE,
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
    UI_FONT,
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
    cv2.putText(panel, title, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE, TEXT_PRIMARY, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, SUBTITLE_SCALE, TEXT_SECONDARY, 1, cv2.LINE_AA)
    return header_h


def draw_stat_box(panel, x, y, w, h, label, value):
    _draw_card(panel, x, y, w, h, fill=SURFACE_BG, border=SURFACE_BORDER, accent=ACCENT)
    cv2.putText(panel, label.upper(), (x + CARD_PAD, y + 24), cv2.FONT_HERSHEY_SIMPLEX, STAT_LABEL_SCALE, TEXT_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(panel, value, (x + CARD_PAD, y + 60), cv2.FONT_HERSHEY_SIMPLEX, STAT_VALUE_SCALE, TEXT_PRIMARY, 2, cv2.LINE_AA)


def draw_lr_table(panel, x, y, w, row_h, title, rows):
    hh = row_h * (len(rows) + 1) + 6
    _draw_card(panel, x, y, w, hh, fill=SURFACE_BG, border=SURFACE_BORDER)
    cv2.putText(panel, title, (x + CARD_PAD, y + 28), cv2.FONT_HERSHEY_SIMPLEX, TABLE_TITLE_SCALE, TEXT_PRIMARY, 2, cv2.LINE_AA)
    header_y = y + row_h
    cv2.line(panel, (x + 1, header_y), (x + w - 1, header_y), SURFACE_BORDER, 1)
    col_label = x + CARD_PAD
    col_L = x + int(w * 0.62)
    col_R = x + int(w * 0.82)
    cv2.putText(panel, "L", (col_L, y + row_h - 12), cv2.FONT_HERSHEY_SIMPLEX, TABLE_LABEL_SCALE, TEXT_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(panel, "R", (col_R, y + row_h - 12), cv2.FONT_HERSHEY_SIMPLEX, TABLE_LABEL_SCALE, TEXT_SECONDARY, 1, cv2.LINE_AA)
    for i, (lab, lv, rv) in enumerate(rows):
        yy = y + row_h * (i + 1) + 10
        if i > 0:
            cv2.line(panel, (x + CARD_PAD, yy - 6), (x + w - CARD_PAD, yy - 6), SURFACE_BG_ALT, 1)
        cv2.putText(panel, lab, (col_label, yy + 18), cv2.FONT_HERSHEY_SIMPLEX, TABLE_LABEL_SCALE, TEXT_SECONDARY, 1, cv2.LINE_AA)
        put_text_deg(panel, lv, (col_L, yy + 22), cv2.FONT_HERSHEY_SIMPLEX, TABLE_VALUE_SCALE, TEXT_PRIMARY, 2)
        put_text_deg(panel, rv, (col_R, yy + 22), cv2.FONT_HERSHEY_SIMPLEX, TABLE_VALUE_SCALE, TEXT_PRIMARY, 2)
    return y + hh


def draw_section_title(panel, x, y, w, title, expanded):
    h = 42
    _draw_card(panel, x, y, w, h, fill=SURFACE_BG_ALT, border=SURFACE_BORDER)
    tri = "v" if expanded else ">"
    cv2.putText(panel, f"{tri} {title}", (x + CARD_PAD, y + 28), cv2.FONT_HERSHEY_SIMPLEX, SECTION_TITLE_SCALE, TEXT_PRIMARY, 2, cv2.LINE_AA)
    return y + h


def draw_controls_section(panel, x, y, w, expanded):
    y = draw_section_title(panel, x, y, w, "Controls", expanded)
    if not expanded:
        cv2.putText(panel, "? to expand", (x + CARD_PAD, y + 22), cv2.FONT_HERSHEY_SIMPLEX, UI_SCALE_SMALL, TEXT_MUTED, 1, cv2.LINE_AA)
        return y + 36
    rows = [
        ("q", "quit"),
        ("v", "toggle camera bg"),
        ("u", "toggle panel"),
        ("s", "toggle skeleton"),
        ("a", "toggle polar+clip"),
        ("g", "open palette gallery"),
        ("esc", "close palette gallery"),
        ("j", "toggle joints"),
        ("p", "toggle vis text"),
        ("l", "toggle console"),
        ("?", "collapse this"),
        ("mouse", "LIVE/REVIEW buttons"),
        ("mouse", "REC captures up to 60s"),
        ("mouse", "REVIEW: drag 10s window"),
        ("mouse", "gallery: choose style"),
    ]
    box_h = 24 + len(rows) * CONTROL_ROW_H
    _draw_card(panel, x, y, w, box_h, fill=SURFACE_BG, border=SURFACE_BORDER)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 1
    xk, xd = x + CARD_PAD, x + 96
    yy = y + 24
    for k, d in rows:
        cv2.putText(panel, f"{k}", (xk, yy), font, CONTROL_KEY_SCALE, TEXT_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(panel, f"{d}", (xd, yy), font, CONTROL_DESC_SCALE, TEXT_SECONDARY, thick, cv2.LINE_AA)
        yy += CONTROL_ROW_H
    return y + box_h + 10


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
    (tw, th), _ = cv2.getTextSize(label, UI_FONT, scale, thick)
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 + th) // 2 - 1
    cv2.putText(img, label, (tx, ty), UI_FONT, scale, text_col, thick, cv2.LINE_AA)
