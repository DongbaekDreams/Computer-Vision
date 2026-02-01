import time
import math
import sys
import os
import urllib.request
from collections import deque

import cv2
import numpy as np

# ============================
# CONFIG
# ============================
CAM_INDEX = 0
WINDOW = "MediaPipe Pose (Tasks) - Dashboard (q to quit)"

# Dashboard layout
PANEL_W = 420
VIEW_H = 720
VIEW_W = 1280
VIDEO_PAD = 12

# Right-side live polar plot
PLOT_W = 360                 # preferred plot width
PLOT_PAD = 12
HISTORY_LEN = 240            # ~4s @60fps, ~8s @30fps
PLOT_BG = (8, 8, 8)
PLOT_RING = (40, 40, 40)
PLOT_AXIS = (32, 32, 32)

# Display behavior
VIDEO_MAX_SCALE = 1.0
VIDEO_SCALE = 1.0

# Mirror preview (and swap L/R so labels match what you see)
MIRROR_VIEW = True

# Camera resolution
CAM_W, CAM_H = 1280, 720

# Angle clamp / sanity
ANGLE_MIN = 0.0
ANGLE_MAX = 180.0

# Drawing defaults
SHOW_SKELETON = True
SHOW_JOINTS = True
SHOW_VIS = False  # visibility text by each point (debug)

# Toggle "video" means toggle camera background only; drawings remain
SHOW_CAMERA_BG = True

# Console
LOG_INTERVAL = 1.0
SHOW_CONSOLE = True

# Visibility threshold for drawing
VIS_MIN = 0.30

# Tasks model asset (.task)
TASK_PATH = "pose_landmarker.task"
TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

# Style
EDGE_THICK = 4
FOOT_THICK = 5
JOINT_OUTLINE_THICK = 2
JOINT_OUTLINE_R = 7
JOINT_DOT_R = 2

# ============================
# REQUIRE: MediaPipe Tasks
# ============================
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "MediaPipe Tasks API not available. Use Python 3.11 and:\n"
        "  pip install mediapipe\n"
    ) from e

# ============================
# LANDMARK INDICES (33)
# ============================
NOSE = 0
L_EYE_INNER, L_EYE, L_EYE_OUTER = 1, 2, 3
R_EYE_INNER, R_EYE, R_EYE_OUTER = 4, 5, 6
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22

L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32

LEFT_IDXS = {
    L_EYE_INNER, L_EYE, L_EYE_OUTER, L_EAR, MOUTH_L,
    L_SHOULDER, L_ELBOW, L_WRIST, L_PINKY, L_INDEX, L_THUMB,
    L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT_INDEX
}
RIGHT_IDXS = {
    R_EYE_INNER, R_EYE, R_EYE_OUTER, R_EAR, MOUTH_R,
    R_SHOULDER, R_ELBOW, R_WRIST, R_PINKY, R_INDEX, R_THUMB,
    R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT_INDEX
}

# ============================
# EDGES (grouped for clean coloring)
# ============================
EDGES_FACE = [
    (NOSE, L_EYE_INNER), (L_EYE_INNER, L_EYE), (L_EYE, L_EYE_OUTER),
    (NOSE, R_EYE_INNER), (R_EYE_INNER, R_EYE), (R_EYE, R_EYE_OUTER),
    (L_EYE_OUTER, L_EAR), (R_EYE_OUTER, R_EAR),
    (MOUTH_L, MOUTH_R), (NOSE, MOUTH_L), (NOSE, MOUTH_R),
]

EDGES_TORSO = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
]

EDGES_ARMS = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_WRIST, L_THUMB), (L_WRIST, L_INDEX), (L_WRIST, L_PINKY),
    (R_WRIST, R_THUMB), (R_WRIST, R_INDEX), (R_WRIST, R_PINKY),
]

EDGES_LEGS = [
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

EDGES_FEET = [
    (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT_INDEX), (L_ANKLE, L_FOOT_INDEX),
    (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT_INDEX), (R_ANKLE, R_FOOT_INDEX),
]

EDGES = EDGES_FACE + EDGES_TORSO + EDGES_ARMS + EDGES_LEGS + EDGES_FEET

# ============================
# MUTED ANATOMICAL PALETTE (BGR)
# ============================
COL_FACE = (235, 235, 235)
COL_TORSO = (200, 200, 200)
COL_ARMS = (200, 150, 110)     # muted blue-ish (BGR)
COL_LEGS = (140, 190, 140)     # muted green
COL_FEET = (105, 160, 105)     # darker green

EDGE_COLORS = (
    [COL_FACE] * len(EDGES_FACE) +
    [COL_TORSO] * len(EDGES_TORSO) +
    [COL_ARMS] * len(EDGES_ARMS) +
    [COL_LEGS] * len(EDGES_LEGS) +
    [COL_FEET] * len(EDGES_FEET)
)

def is_foot_edge(u, v):
    return (u, v) in EDGES_FEET or (v, u) in EDGES_FEET

def edge_thickness(u, v):
    if is_foot_edge(u, v):
        return FOOT_THICK
    if (u, v) in EDGES_TORSO or (v, u) in EDGES_TORSO:
        return EDGE_THICK + 1
    if (u, v) in EDGES_FACE or (v, u) in EDGES_FACE:
        return max(2, EDGE_THICK - 2)
    return EDGE_THICK

# ============================
# UTILS
# ============================
def ensure_task_file(path: str, url: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    urllib.request.urlretrieve(url, path)

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

def joint_colors(idx):
    outline = (235, 235, 235)
    if idx in LEFT_IDXS:
        dot = (180, 170, 120)  # warm-ish muted
    elif idx in RIGHT_IDXS:
        dot = (160, 140, 200)  # cool-ish muted
    else:
        dot = (200, 200, 200)
    return outline, dot

def draw_joint_marker(img, p_xy, idx):
    x, y = int(p_xy[0]), int(p_xy[1])
    outline, dot = joint_colors(idx)
    cv2.circle(img, (x, y), JOINT_OUTLINE_R, outline, JOINT_OUTLINE_THICK, cv2.LINE_AA)
    cv2.circle(img, (x, y), JOINT_DOT_R, dot, -1, cv2.LINE_AA)

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

    box_h = 170
    cv2.rectangle(panel, (x, y), (x + w, y + box_h), (18, 18, 18), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + box_h), (45, 45, 45), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 1

    left = [
        ("q", "quit"),
        ("v", "toggle camera bg"),
        ("u", "toggle panel"),
        ("s", "toggle skeleton"),
        ("a", "toggle polar plot"),
    ]
    right = [
        ("j", "toggle joints"),
        ("p", "toggle vis text"),
        ("l", "toggle console"),
        ("?", "collapse this"),
        ("[sliders]", "plot start/end"),
    ]

    row_h = 30
    x1 = x + 14
    x2 = x + w // 2 + 6
    yy = y + 26

    for i in range(5):
        k, d = left[i]
        cv2.putText(panel, f"{k}", (x1, yy), font, scale, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(panel, f" {d}", (x1 + 18, yy), font, scale, (190, 190, 190), thick, cv2.LINE_AA)

        k, d = right[i]
        cv2.putText(panel, f"{k}", (x2, yy), font, scale, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(panel, f" {d}", (x2 + 18, yy), font, scale, (190, 190, 190), thick, cv2.LINE_AA)

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

# ============================
# LIVE POLAR PLOT (OpenCV)
# ============================
def _colormap_plasma_u8(t01_u8):
    ramp = t01_u8.reshape(-1, 1)
    ramp3 = cv2.applyColorMap(ramp, cv2.COLORMAP_PLASMA)  # (N,1,3) BGR
    return ramp3.reshape(-1, 3)

def draw_polar_plot(canvas_bgr, series_dict, start_idx, end_idx, title="Polar"):
    h, w = canvas_bgr.shape[:2]
    canvas_bgr[:] = PLOT_BG

    cv2.putText(canvas_bgr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "start/end via sliders", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 165, 165), 1, cv2.LINE_AA)

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

    any_series = next(iter(series_dict.values()))
    T = int(any_series.shape[0])
    if T <= 2:
        return

    s = int(np.clip(start_idx, 0, T - 1))
    e = int(np.clip(end_idx, 0, T - 1))
    if e <= s:
        e = min(T - 1, s + 1)

    n = e - s + 1
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)

    for name, arr in series_dict.items():
        seg = arr[s:e + 1]
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
            cv2.line(
                canvas_bgr,
                (int(xs[i]), int(ys[i])),
                (int(xs[i + 1]), int(ys[i + 1])),
                c,
                2,
                cv2.LINE_AA,
            )

    cv2.circle(canvas_bgr, (cx, cy), 3, (220, 220, 220), -1, cv2.LINE_AA)

# ============================
# CONSOLE (crisp single-line)
# ============================
ANSI_OK = sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")
RESET = "\x1b[0m"

def ansi_fg_gray(level_0_1):
    if not ANSI_OK:
        return ""
    level_0_1 = max(0.0, min(1.0, float(level_0_1)))
    code = 232 + int(round(level_0_1 * 23))
    return f"\x1b[38;5;{code}m"

def cgray(text, level):
    if not ANSI_OK:
        return text
    return f"{ansi_fg_gray(level)}{text}{RESET}"

def fmt_deg(x):
    return "--" if x is None else f"{int(x):3d}°"

def console_line(elapsed_s, fps, infer_ms, Lh, Rh, Lk, Rk, La, Ra):
    t_inf = (infer_ms - 6.0) / 20.0
    t_inf = max(0.0, min(1.0, t_inf))
    t_fps = (fps - 15.0) / 45.0
    t_fps = max(0.0, min(1.0, t_fps))

    s_fps = cgray(f"{fps:5.1f}", 0.35 + 0.55 * t_fps)
    s_inf = cgray(f"{infer_ms:5.1f}ms", 0.30 + 0.60 * t_inf)

    return (
        f"t={elapsed_s:6.1f}s | FPS {s_fps} | infer {s_inf} | "
        f"Hip L {fmt_deg(Lh)} R {fmt_deg(Rh)} | "
        f"Knee L {fmt_deg(Lk)} R {fmt_deg(Rk)} | "
        f"Ank L {fmt_deg(La)} R {fmt_deg(Ra)}"
    )

# ============================
# TASKS: CREATE LANDMARKER
# ============================
ensure_task_file(TASK_PATH, TASK_URL)

base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# ============================
# CAMERA
# ============================
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera not available")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, VIEW_W, VIEW_H)

# ============================
# STATE
# ============================
t0 = time.time()
last_log = 0.0
frame_i = 0
fps = 0.0

show_skeleton = SHOW_SKELETON
show_joints = SHOW_JOINTS
show_vis = SHOW_VIS
show_console = SHOW_CONSOLE
show_panel = True
show_camera_bg = SHOW_CAMERA_BG
controls_expanded = False
show_polar = True

# Angle history buffers (float, may be nan)
hist = {
    "Hip L": deque(maxlen=HISTORY_LEN),
    "Hip R": deque(maxlen=HISTORY_LEN),
    "Knee L": deque(maxlen=HISTORY_LEN),
    "Knee R": deque(maxlen=HISTORY_LEN),
    "Ank L": deque(maxlen=HISTORY_LEN),
    "Ank R": deque(maxlen=HISTORY_LEN),
}

def _noop(_=None):
    return

cv2.createTrackbar("start", WINDOW, 0, max(1, HISTORY_LEN - 1), _noop)
cv2.createTrackbar("end", WINDOW, max(1, HISTORY_LEN - 1), max(1, HISTORY_LEN - 1), _noop)

# ============================
# MAIN LOOP
# ============================
try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

        t_inf0 = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int((time.time() - t0) * 1000.0)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        infer_ms = (time.time() - t_inf0) * 1000.0

        frame_i += 1
        if frame_i % 10 == 0:
            fps = frame_i / max(1e-6, (time.time() - t0))

        if show_camera_bg:
            video = frame.copy()
        else:
            video = np.zeros_like(frame)
            video[:] = (8, 8, 8)

        L_hip_i = L_knee_i = L_ank_i = None
        R_hip_i = R_knee_i = R_ank_i = None

        L_hip_f = L_knee_f = L_ank_f = np.nan
        R_hip_f = R_knee_f = R_ank_f = np.nan

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = result.pose_landmarks[0]
            h, w = frame.shape[:2]

            pts = {}
            vis = {}

            for i in range(len(lm)):
                pts[i] = np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
                v = getattr(lm[i], "visibility", 0.0)
                vis[i] = float(v) if v is not None else 0.0

            if MIRROR_VIEW:
                swap_pairs = [
                    (L_EYE_INNER, R_EYE_INNER), (L_EYE, R_EYE), (L_EYE_OUTER, R_EYE_OUTER),
                    (L_EAR, R_EAR), (MOUTH_L, MOUTH_R),
                    (L_SHOULDER, R_SHOULDER),
                    (L_ELBOW, R_ELBOW),
                    (L_WRIST, R_WRIST),
                    (L_PINKY, R_PINKY),
                    (L_INDEX, R_INDEX),
                    (L_THUMB, R_THUMB),
                    (L_HIP, R_HIP),
                    (L_KNEE, R_KNEE),
                    (L_ANKLE, R_ANKLE),
                    (L_HEEL, R_HEEL),
                    (L_FOOT_INDEX, R_FOOT_INDEX),
                ]
                for a, b in swap_pairs:
                    pts[a], pts[b] = pts[b], pts[a]
                    vis[a], vis[b] = vis[b], vis[a]

            def angle_if(a, b, c):
                if vis_ok(vis.get(a)) and vis_ok(vis.get(b)) and vis_ok(vis.get(c)):
                    return clamp(angle_deg(pts[a], pts[b], pts[c]), ANGLE_MIN, ANGLE_MAX)
                return np.nan

            L_hip = angle_if(L_SHOULDER, L_HIP, L_KNEE)
            R_hip = angle_if(R_SHOULDER, R_HIP, R_KNEE)

            L_knee = angle_if(L_HIP, L_KNEE, L_ANKLE)
            R_knee = angle_if(R_HIP, R_KNEE, R_ANKLE)

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

            L_ank = ankle_angle("L")
            R_ank = ankle_angle("R")

            L_hip_f, R_hip_f = float(L_hip), float(R_hip)
            L_knee_f, R_knee_f = float(L_knee), float(R_knee)
            L_ank_f, R_ank_f = float(L_ank), float(R_ank)

            L_hip_i, R_hip_i = round_deg(L_hip), round_deg(R_hip)
            L_knee_i, R_knee_i = round_deg(L_knee), round_deg(R_knee)
            L_ank_i, R_ank_i = round_deg(L_ank), round_deg(R_ank)

            if show_skeleton:
                for (u, v), col in zip(EDGES, EDGE_COLORS):
                    if vis_ok(vis.get(u)) and vis_ok(vis.get(v)):
                        cv2.line(
                            video,
                            tuple(pts[u].astype(int)),
                            tuple(pts[v].astype(int)),
                            col,
                            edge_thickness(u, v),
                            cv2.LINE_AA,
                        )

            if show_joints:
                for idx, p in pts.items():
                    if vis_ok(vis.get(idx)):
                        draw_joint_marker(video, p, idx)
                        if show_vis:
                            cv2.putText(
                                video,
                                f"{vis[idx]:.2f}",
                                (int(p[0]) + 10, int(p[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (240, 240, 240),
                                1,
                                cv2.LINE_AA,
                            )

        # History append
        hist["Hip L"].append(L_hip_f)
        hist["Hip R"].append(R_hip_f)
        hist["Knee L"].append(L_knee_f)
        hist["Knee R"].append(R_knee_f)
        hist["Ank L"].append(L_ank_f)
        hist["Ank R"].append(R_ank_f)

        # ============================
        # BUILD DASHBOARD
        # ============================
        dash = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        dash[:] = (10, 10, 10)

        panel_w_eff = PANEL_W if show_panel else 0
        pane_w = VIEW_W - panel_w_eff
        pane_h = VIEW_H

        if show_panel:
            panel = panel_bg(VIEW_H, PANEL_W)

            y = draw_panel_header(panel, "Pose Dashboard", subtitle="Crisp panel • ? toggles Controls")

            col1_x = 16
            col2_x = PANEL_W // 2 + 6
            box_w = PANEL_W // 2 - 22
            y += 14
            box_h = 74
            draw_stat_box(panel, col1_x, y, box_w, box_h, "FPS", f"{fps:0.1f}")
            draw_stat_box(panel, col2_x, y, box_w, box_h, "Infer (ms)", f"{infer_ms:0.1f}")
            y += box_h + 14

            y = draw_lr_table(
                panel, 16, y, PANEL_W - 32, 46, "Angles",
                rows=[
                    ("Hip",   L_hip_i,  R_hip_i),
                    ("Knee",  L_knee_i, R_knee_i),
                    ("Ankle", L_ank_i,  R_ank_i),
                ],
            )
            y += 14

            y = draw_controls_section(panel, 16, y, PANEL_W - 32, controls_expanded)

            dash[:, :PANEL_W] = panel
            cv2.line(dash, (PANEL_W, 0), (PANEL_W, VIEW_H - 1), (45, 45, 45), 1)

        # ============================
        # RIGHT PANE: VIDEO + POLAR PLOT (ALWAYS SHOW WHEN ENABLED)
        # ============================
        plot_enabled = bool(show_polar)

        if plot_enabled:
            plot_pad = PLOT_PAD
            plot_y0 = plot_pad
            plot_h = pane_h - 2 * plot_pad

            min_video_w = 260
            video_area_w = max(min_video_w, pane_w - (PLOT_W + 2 * plot_pad))
            plot_w_eff = max(220, pane_w - video_area_w - 2 * plot_pad)

            video_pane = fit_video_to_pane(video, video_area_w - 2 * VIDEO_PAD, pane_h - 2 * VIDEO_PAD)
            x_off = panel_w_eff + VIDEO_PAD
            dash[VIDEO_PAD:VIDEO_PAD + video_pane.shape[0], x_off:x_off + video_pane.shape[1]] = video_pane

            plot_x0 = panel_w_eff + video_area_w + plot_pad
            plot_canvas = np.zeros((plot_h, plot_w_eff, 3), dtype=np.uint8)

            Tcur = len(hist["Hip L"])
            padn = max(0, HISTORY_LEN - Tcur)

            series = {}
            for k, dq in hist.items():
                arr = np.array(dq, dtype=np.float32)
                if padn > 0:
                    arr = np.concatenate([np.full((padn,), np.nan, dtype=np.float32), arr], axis=0)
                else:
                    arr = arr[-HISTORY_LEN:]
                series[k] = arr

            s = cv2.getTrackbarPos("start", WINDOW)
            e = cv2.getTrackbarPos("end", WINDOW)
            if e <= s:
                e = min(HISTORY_LEN - 1, s + 1)
                cv2.setTrackbarPos("end", WINDOW, e)

            draw_polar_plot(
                plot_canvas,
                series_dict=series,
                start_idx=s,
                end_idx=e,
                title="Angles (Polar)",
            )

            dash[plot_y0:plot_y0 + plot_h, plot_x0:plot_x0 + plot_w_eff] = plot_canvas
            cv2.line(dash, (panel_w_eff + video_area_w, 0), (panel_w_eff + video_area_w, VIEW_H - 1), (35, 35, 35), 1)
        else:
            video_pane = fit_video_to_pane(video, pane_w - 2 * VIDEO_PAD, pane_h - 2 * VIDEO_PAD)
            x_off = panel_w_eff + VIDEO_PAD
            dash[VIDEO_PAD:VIDEO_PAD + video_pane.shape[0], x_off:x_off + video_pane.shape[1]] = video_pane

        # ============================
        # CONSOLE
        # ============================
        if show_console and (time.time() - last_log) >= LOG_INTERVAL:
            line = console_line(
                time.time() - t0, fps, infer_ms,
                L_hip_i, R_hip_i, L_knee_i, R_knee_i, L_ank_i, R_ank_i
            )
            sys.stdout.write("\r" + line + " " * 10)
            sys.stdout.flush()
            last_log = time.time()

        cv2.imshow(WINDOW, dash)

        key = (cv2.waitKey(1) & 0xFF)
        if key == ord("q"):
            break
        elif key in (ord("v"), ord("V")):
            show_camera_bg = not show_camera_bg
        elif key in (ord("u"), ord("U")):
            show_panel = not show_panel
        elif key in (ord("s"), ord("S")):
            show_skeleton = not show_skeleton
        elif key in (ord("j"), ord("J")):
            show_joints = not show_joints
        elif key in (ord("p"), ord("P")):
            show_vis = not show_vis
        elif key in (ord("l"), ord("L")):
            show_console = not show_console
            if not show_console:
                sys.stdout.write("\n")
                sys.stdout.flush()
        elif key == ord("?"):
            controls_expanded = not controls_expanded
        elif key in (ord("a"), ord("A")):
            show_polar = not show_polar

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    if SHOW_CONSOLE:
        sys.stdout.write("\n")
        sys.stdout.flush()
