"""Timeline UI: recording/playback controls, segment extraction."""

import time

import cv2
import numpy as np

from config import (
    BTN_BG,
    BTN_DIM,
    BTN_GREEN,
    BTN_RED,
    BTN_TEXT,
    BTN_YELLOW,
    MAX_REC_SECONDS,
    PLAY_TICK,
    PLAY_TICK_DIM,
    SEG_SECONDS,
    TL_BAR,
    TL_BG,
    TL_BORDER,
    TL_TEXT,
    TL_TEXT_DIM,
    TL_TICK,
    UI_FONT,
    UI_SCALE_SMALL,
    WIN_BORDER,
    WIN_FILL,
    WIN_HANDLE,
)
from pose_processor import ANGLE_KEYS
from state import clear_recording, t_rec
from ui.drawing import draw_button


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def trim_time_buffer(t_deque, *data_deques, keep_last_seconds=60.0):
    if not t_deque:
        return
    t_latest = float(t_deque[-1])
    t_cut = t_latest - float(keep_last_seconds)
    while t_deque and float(t_deque[0]) < t_cut:
        t_deque.popleft()
        for dq in data_deques:
            dq.popleft()


def _fit_text_to_width_ascii(text, max_w_px, font=UI_FONT, scale=UI_SCALE_SMALL, thickness=1):
    if max_w_px <= 0 or text == "":
        return ""
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw <= max_w_px:
        return text
    ell = "..."
    (ew, _), _ = cv2.getTextSize(ell, font, scale, thickness)
    if ew >= max_w_px:
        return ""
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid] + ell
        (cw, _), _ = cv2.getTextSize(cand, font, scale, thickness)
        if cw <= max_w_px:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def extract_segment_by_time(t_deque, angles_dict, pose_deque, start_t, end_t):
    if not t_deque:
        return None, None, None
    ts = np.asarray(t_deque, dtype=np.float64)
    i0 = int(np.searchsorted(ts, start_t, side="left"))
    i1 = int(np.searchsorted(ts, end_t, side="right"))
    i0 = max(0, min(i0, len(ts)))
    i1 = max(0, min(i1, len(ts)))
    if i1 - i0 < 2:
        return None, None, None
    seg_ts = ts[i0:i1]
    seg_series = {}
    for k, dq in angles_dict.items():
        arr = np.asarray(dq, dtype=np.float32)
        seg_series[k] = arr[i0:i1]
    seg_poses = list(pose_deque)[i0:i1]
    return seg_ts, seg_series, seg_poses


# UI hitboxes (set each frame)
timeline_bar_rect = None
window_rect = None
live_btn_rect = None
review_btn_rect = None
rec_btn_rect = None
play_btn_rect = None
clear_btn_rect = None

win_drag = False
drag_grab_dt = 0.0
x_to_t_map_fn = None

# Modes and recording/playback state
live_mode = True
recording = False
record_done = False
record_start_wall = None
record_elapsed = 0.0
playing = False
play_phase = 0.0
play_last_t = time.time()
pinned_start_t = 0.0


def draw_mode_toggle(img, x, y, h, is_live_mode):
    global live_btn_rect, review_btn_rect
    w_each = 58
    gap = 6
    live_btn_rect = (x, y, x + w_each, y + h)
    review_btn_rect = (x + w_each + gap, y, x + 2 * w_each + gap, y + h)
    draw_button(img, live_btn_rect, "LIVE", (220, 160, 60) if is_live_mode else BTN_BG, BTN_TEXT, scale=UI_SCALE_SMALL)
    draw_button(img, review_btn_rect, "REVIEW", (220, 160, 60) if (not is_live_mode) else BTN_BG, BTN_TEXT, scale=UI_SCALE_SMALL)
    return review_btn_rect[2]


def draw_timeline_ui(img, x0, y0, w, h,
                     is_live_mode, duration_s, pinned_start, play_ph,
                     is_recording, is_record_done, is_playing):
    global timeline_bar_rect, window_rect, rec_btn_rect, play_btn_rect, clear_btn_rect, x_to_t_map_fn

    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), TL_BG, -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), TL_BORDER, 1)

    pad = 10
    btn_row_h = 24
    btn_row_y0 = y0 + 6
    status_row_y = btn_row_y0 + btn_row_h + 16
    bar_h = 14
    bar_y0 = status_row_y + 14
    bar_y1 = bar_y0 + bar_h
    tick_y0 = bar_y1 + 10
    tick_y1 = tick_y0 + 7
    label_y = tick_y1 + 12

    right_of_toggle = draw_mode_toggle(img, x0 + pad, btn_row_y0, btn_row_h, is_live_mode)

    btn_w = 62
    gap = 8
    xR = x0 + w - pad

    clear_btn_rect = (xR - btn_w, btn_row_y0, xR, btn_row_y0 + btn_row_h)
    xR = clear_btn_rect[0] - gap
    play_btn_rect = (xR - btn_w, btn_row_y0, xR, btn_row_y0 + btn_row_h)
    xR = play_btn_rect[0] - gap
    rec_btn_rect = (xR - btn_w, btn_row_y0, xR, btn_row_y0 + btn_row_h)
    xR = rec_btn_rect[0] - gap

    if is_recording:
        rec_fill = BTN_YELLOW
        rec_label = "STOP"
    else:
        rec_fill = BTN_RED
        rec_label = "REC"

    can_play = (not is_live_mode) and (duration_s >= SEG_SECONDS) and (not is_recording)
    play_fill = (BTN_GREEN if is_playing else BTN_BG) if can_play else BTN_BG
    play_text_col = BTN_TEXT if can_play else BTN_DIM
    play_label = "PAUSE" if (can_play and is_playing) else "PLAY"

    can_clear = (not is_recording) and (duration_s > 0.0)
    clr_text_col = BTN_TEXT if can_clear else BTN_DIM

    draw_button(img, rec_btn_rect, rec_label, rec_fill, BTN_TEXT, scale=UI_SCALE_SMALL)
    draw_button(img, play_btn_rect, play_label, play_fill, play_text_col, scale=UI_SCALE_SMALL)
    draw_button(img, clear_btn_rect, "CLEAR", BTN_BG, clr_text_col, scale=UI_SCALE_SMALL)

    status_left = right_of_toggle + 12
    status_right = rec_btn_rect[0] - 10
    status_w = max(0, status_right - status_left)

    if is_recording:
        status = f"REC {duration_s:0.1f}/{int(MAX_REC_SECONDS)}s"
    elif is_record_done:
        status = f"REC DONE {duration_s:0.1f}s"
    elif duration_s > 0.0:
        status = f"REC READY {duration_s:0.1f}s"
    else:
        status = "REC READY"

    if not is_live_mode and duration_s >= SEG_SECONDS:
        ws = float(_clamp(pinned_start, 0.0, max(0.0, duration_s - SEG_SECONDS)))
        status += f" | WIN {ws:0.1f}-{(ws + SEG_SECONDS):0.1f}s"

    status = _fit_text_to_width_ascii(status, status_w, font=UI_FONT, scale=UI_SCALE_SMALL, thickness=1)
    if status:
        cv2.putText(img, status, (status_left, status_row_y), UI_FONT, UI_SCALE_SMALL, TL_TEXT, 1, cv2.LINE_AA)

    bar_x0 = x0 + pad
    bar_x1 = x0 + w - pad
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x1, bar_y1), TL_BAR, -1)
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x1, bar_y1), TL_BORDER, 1)

    timeline_bar_rect = (bar_x0, bar_y0, bar_x1, bar_y1)
    window_rect = None
    x_to_t_map_fn = None

    def draw_ticks(dur):
        if dur <= 0.0:
            return
        step = 10.0 if dur >= 40.0 else 5.0

        def t_to_x(t):
            u = _clamp(float(t) / dur, 0.0, 1.0)
            return int(round(bar_x0 + u * (bar_x1 - bar_x0)))

        t = 0.0
        last_x = None
        while t <= dur + 1e-6:
            x = t_to_x(t)
            if last_x is None or abs(x - last_x) >= 54:
                cv2.line(img, (x, tick_y0), (x, tick_y1), TL_TICK, 1, cv2.LINE_AA)
                cv2.putText(img, f"{int(round(t))}s", (x - 10, label_y), UI_FONT, 0.38, TL_TEXT_DIM, 1, cv2.LINE_AA)
                last_x = x
            t += step

    if is_live_mode or duration_s <= 1e-6:
        draw_ticks(float(duration_s))
        return

    dur = float(duration_s)

    def t_to_x(t):
        u = _clamp(float(t) / dur, 0.0, 1.0)
        return int(round(bar_x0 + u * (bar_x1 - bar_x0)))

    def x_to_t(x):
        u = (x - bar_x0) / float(max(1, (bar_x1 - bar_x0)))
        u = _clamp(float(u), 0.0, 1.0)
        return u * dur

    x_to_t_map_fn = x_to_t
    draw_ticks(dur)

    win_start = float(_clamp(pinned_start, 0.0, max(0.0, dur - SEG_SECONDS)))
    win_end = min(dur, win_start + SEG_SECONDS)

    wx0 = t_to_x(win_start)
    wx1 = t_to_x(win_end)

    min_w = 44
    if wx1 - wx0 < min_w:
        wx1 = min(bar_x1, wx0 + min_w)
        wx0 = max(bar_x0, wx1 - min_w)

    cv2.rectangle(img, (wx0, bar_y0), (wx1, bar_y1), WIN_FILL, -1)
    cv2.rectangle(img, (wx0, bar_y0), (wx1, bar_y1), WIN_BORDER, 2)

    handle_w = 10
    cv2.rectangle(img, (wx0, bar_y0), (wx0 + handle_w, bar_y1), (60, 60, 60), -1)
    cv2.rectangle(img, (wx1 - handle_w, bar_y0), (wx1, bar_y1), (60, 60, 60), -1)
    cv2.line(img, (wx0 + 4, bar_y0 + 2), (wx0 + 4, bar_y1 - 2), WIN_HANDLE, 1, cv2.LINE_AA)
    cv2.line(img, (wx1 - 5, bar_y0 + 2), (wx1 - 5, bar_y1 - 2), WIN_HANDLE, 1, cv2.LINE_AA)

    if is_playing and dur >= SEG_SECONDS:
        px = int(round(wx0 + _clamp(play_ph, 0.0, 1.0) * max(1, (wx1 - wx0 - 1))))
        cv2.line(img, (px, bar_y0 - 2), (px, bar_y1 + 2), PLAY_TICK, 1, cv2.LINE_AA)
    else:
        cv2.line(img, (wx0, bar_y0 - 2), (wx0, bar_y1 + 2), PLAY_TICK_DIM, 1, cv2.LINE_AA)

    window_rect = (wx0, bar_y0, wx1, bar_y1)


def make_mouse_cb():
    """Create mouse callback that closes over timeline module state."""

    def _mouse_cb(event, x, y, flags, param):
        global live_mode, recording, record_done, record_start_wall, record_elapsed
        global playing, play_phase, pinned_start_t, win_drag, drag_grab_dt

        if event == cv2.EVENT_LBUTTONDOWN:
            if live_btn_rect is not None:
                x0, y0, x1, y1 = live_btn_rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    live_mode = True
                    playing = False
                    play_phase = 0.0
                    win_drag = False
                    return

            if review_btn_rect is not None:
                x0, y0, x1, y1 = review_btn_rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    live_mode = False
                    playing = False
                    play_phase = 0.0
                    win_drag = False
                    return

            if rec_btn_rect is not None:
                x0, y0, x1, y1 = rec_btn_rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if recording:
                        recording = False
                        record_done = True
                        playing = False
                        play_phase = 0.0
                    else:
                        clear_recording()
                        recording = True
                        record_done = False
                        playing = False
                        play_phase = 0.0
                        pinned_start_t = 0.0
                        record_start_wall = time.time()
                        record_elapsed = 0.0
                    return

            if play_btn_rect is not None:
                x0, y0, x1, y1 = play_btn_rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if (not live_mode) and (not recording) and len(t_rec) >= 2 and (t_rec[-1] >= SEG_SECONDS):
                        playing = not playing
                        play_phase = 0.0
                    return

            if clear_btn_rect is not None:
                x0, y0, x1, y1 = clear_btn_rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if not recording:
                        clear_recording()
                        record_done = False
                        playing = False
                        play_phase = 0.0
                        pinned_start_t = 0.0
                        record_start_wall = None
                        record_elapsed = 0.0
                    return

            if live_mode or recording:
                return
            if timeline_bar_rect is None or x_to_t_map_fn is None or window_rect is None:
                return
            if not t_rec:
                return

            dur = float(t_rec[-1])
            if dur < SEG_SECONDS:
                return

            bx0, by0, bx1, by1 = timeline_bar_rect
            if not (bx0 <= x <= bx1 and by0 <= y <= by1):
                return

            wx0, wy0, wx1, wy1 = window_rect
            if (wx0 - 12) <= x <= (wx1 + 12):
                t_click = float(x_to_t_map_fn(x))
                win_start = float(pinned_start_t)
                grab = t_click - win_start
                grab = _clamp(grab, 0.0, SEG_SECONDS)
                drag_grab_dt = grab
                win_drag = True
                play_phase = 0.0
                return

            t_click = float(x_to_t_map_fn(x))
            pinned_start_t = float(t_click - (SEG_SECONDS * 0.5))
            pinned_start_t = _clamp(pinned_start_t, 0.0, max(0.0, dur - SEG_SECONDS))
            play_phase = 0.0
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if win_drag and (not live_mode) and (not recording) and (x_to_t_map_fn is not None) and t_rec:
                dur = float(t_rec[-1])
                if dur >= SEG_SECONDS:
                    t = float(x_to_t_map_fn(x))
                    pinned_start_t = float(t - drag_grab_dt)
                    pinned_start_t = _clamp(pinned_start_t, 0.0, max(0.0, dur - SEG_SECONDS))
            return

        if event == cv2.EVENT_LBUTTONUP:
            win_drag = False
            return

    return _mouse_cb


def clear_hitboxes():
    """Clear UI hitboxes when timeline is not drawn (e.g. polar disabled)."""
    global timeline_bar_rect, window_rect, live_btn_rect, review_btn_rect
    global rec_btn_rect, play_btn_rect, clear_btn_rect, x_to_t_map_fn
    timeline_bar_rect = None
    window_rect = None
    live_btn_rect = None
    review_btn_rect = None
    rec_btn_rect = None
    play_btn_rect = None
    clear_btn_rect = None
    x_to_t_map_fn = None
