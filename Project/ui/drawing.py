"""UI drawing: panels, buttons, stat boxes, tables."""

import cv2
import numpy as np

from config import (
    BTN_BORDER,
    BTN_TEXT,
    UI_FONT,
    UI_SCALE,
    UI_SCALE_SMALL,
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
    img[:] = (16, 16, 16)
    return img


def draw_panel_header(panel, title, subtitle=None):
    w = panel.shape[1]
    header_h = 68 if subtitle else 58
    cv2.rectangle(panel, (0, 0), (w - 1, header_h), (26, 26, 26), -1)
    cv2.line(panel, (0, header_h), (w - 1, header_h), (45, 45, 45), 1)
    cv2.putText(panel, title, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (165, 165, 165), 1, cv2.LINE_AA)
    return header_h


def draw_stat_box(panel, x, y, w, h, label, value):
    cv2.rectangle(panel, (x, y), (x + w, y + h), (22, 22, 22), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (45, 45, 45), 1)
    cv2.putText(panel, label, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(panel, value, (x + 12, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (245, 245, 245), 2, cv2.LINE_AA)


def draw_lr_table(panel, x, y, w, row_h, title, rows):
    hh = row_h * (len(rows) + 1) + 6
    cv2.rectangle(panel, (x, y), (x + w, y + hh), (22, 22, 22), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + hh), (45, 45, 45), 1)
    cv2.putText(panel, title, (x + 12, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 2, cv2.LINE_AA)
    header_y = y + row_h
    cv2.line(panel, (x, header_y), (x + w, header_y), (45, 45, 45), 1)
    col_label = x + 12
    col_L = x + int(w * 0.62)
    col_R = x + int(w * 0.82)
    cv2.putText(panel, "L", (col_L, y + row_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(panel, "R", (col_R, y + row_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    for i, (lab, lv, rv) in enumerate(rows):
        yy = y + row_h * (i + 1) + 10
        cv2.putText(panel, lab, (col_label, yy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
        put_text_deg(panel, lv, (col_L, yy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)
        put_text_deg(panel, rv, (col_R, yy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)
    return y + hh


def draw_section_title(panel, x, y, w, title, expanded):
    h = 42
    cv2.rectangle(panel, (x, y), (x + w, y + h), (22, 22, 22), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (45, 45, 45), 1)
    tri = "v" if expanded else ">"
    cv2.putText(panel, f"{tri} {title}", (x + 12, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2, cv2.LINE_AA)
    return y + h


def draw_controls_section(panel, x, y, w, expanded):
    y = draw_section_title(panel, x, y, w, "Controls", expanded)
    if not expanded:
        cv2.putText(panel, "? to expand", (x + 12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (165, 165, 165), 1, cv2.LINE_AA)
        return y + 34
    box_h = 240
    cv2.rectangle(panel, (x, y), (x + w, y + box_h), (18, 18, 18), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + box_h), (45, 45, 45), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 1
    rows = [
        ("q", "quit"),
        ("v", "toggle camera bg"),
        ("u", "toggle panel"),
        ("s", "toggle skeleton"),
        ("a", "toggle polar+clip"),
        ("j", "toggle joints"),
        ("p", "toggle vis text"),
        ("l", "toggle console"),
        ("?", "collapse this"),
        ("mouse", "LIVE/REVIEW buttons"),
        ("mouse", "REC captures up to 60s"),
        ("mouse", "REVIEW: drag 10s window"),
        ("mouse", "REVIEW: PLAY loops window"),
    ]
    row_h = 26
    xk, xd = x + 14, x + 78
    yy = y + 22
    for k, d in rows:
        cv2.putText(panel, f"{k}", (xk, yy), font, scale, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(panel, f"{d}", (xd, yy), font, scale, (190, 190, 190), thick, cv2.LINE_AA)
        yy += row_h
    return y + box_h + 10


def fit_video_to_pane(frame, pane_w, pane_h):
    h, w = frame.shape[:2]
    scale = min(pane_w / max(1, w), pane_h / max(1, h))
    scale = min(scale, VIDEO_MAX_SCALE) * float(VIDEO_SCALE)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
    canvas[:] = (8, 8, 8)
    x0 = (pane_w - nw) // 2
    y0 = (pane_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def draw_button(img, rect, label, fill, text_col=BTN_TEXT, scale=UI_SCALE, thick=1):
    x0, y0, x1, y1 = rect
    cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), BTN_BORDER, 1)
    (tw, th), _ = cv2.getTextSize(label, UI_FONT, scale, thick)
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 + th) // 2 - 1
    cv2.putText(img, label, (tx, ty), UI_FONT, scale, text_col, thick, cv2.LINE_AA)
