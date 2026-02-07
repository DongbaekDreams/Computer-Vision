import time
import math
import sys
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# ============================
# CONFIG
# ============================
CAM_INDEX = 0
WINDOW = "MoveNet ONNX - Dashboard (q to quit)"

# Dashboard layout
PANEL_W = 420
VIEW_H = 720
VIEW_W = 1280
VIDEO_PAD = 12

# Display behavior: do NOT upscale video (reduces "zoomed in" feel)
VIDEO_MAX_SCALE = 1.0
VIDEO_SCALE = 1.0  # <= 1.0

# Mirror preview; swap L/R after inference so labels match what you see.
MIRROR_VIEW = True

# Model
IN_SIZE = 192
HF_REPO = "Xenova/movenet-singlepose-lightning"
HF_FILE = "onnx/model.onnx"
HF_REV = "main"

# Keypoint smoothing (very responsive; minimizes lag)
KP_EMA_ALPHA = 0.992
KP_ALPHA_MIN = 0.45
KP_SCORE_REF = 0.10

# Angle filter (snappy)
AB_ALPHA = 0.88
AB_BETA = 0.22
ANGLE_MIN = 0.0
ANGLE_MAX = 180.0
MAX_DEG_PER_FRAME = 140.0

# Score thresholds
SCORE_MIN_DEFAULT = 0.12
SCORE_MIN_HIP = 0.14
SCORE_MIN_KNEE = 0.14
SCORE_MIN_ANKLE = 0.10

# Drawing
SHOW_SKELETON = True
SHOW_JOINTS = True
SHOW_SCORES = False

# Console
LOG_INTERVAL = 1.0
SHOW_CONSOLE = True

# Camera resolution
CAM_W, CAM_H = 1280, 720

# ============================
# MOVENET KEYPOINTS
# ============================
KP_NAMES = [
    "NOSE",
    "L_EYE", "R_EYE",
    "L_EAR", "R_EAR",
    "L_SHOULDER", "R_SHOULDER",
    "L_ELBOW", "R_ELBOW",
    "L_WRIST", "R_WRIST",
    "L_HIP", "R_HIP",
    "L_KNEE", "R_KNEE",
    "L_ANKLE", "R_ANKLE",
]

LEFT_RIGHT_SWAPS = [
    ("L_EYE", "R_EYE"),
    ("L_EAR", "R_EAR"),
    ("L_SHOULDER", "R_SHOULDER"),
    ("L_ELBOW", "R_ELBOW"),
    ("L_WRIST", "R_WRIST"),
    ("L_HIP", "R_HIP"),
    ("L_KNEE", "R_KNEE"),
    ("L_ANKLE", "R_ANKLE"),
]

EDGES = [
    ("NOSE", "L_EYE"),
    ("NOSE", "R_EYE"),
    ("L_EYE", "L_EAR"),
    ("R_EYE", "R_EAR"),
    ("L_SHOULDER", "R_SHOULDER"),
    ("L_SHOULDER", "L_ELBOW"),
    ("L_ELBOW", "L_WRIST"),
    ("R_SHOULDER", "R_ELBOW"),
    ("R_ELBOW", "R_WRIST"),
    ("L_SHOULDER", "L_HIP"),
    ("R_SHOULDER", "R_HIP"),
    ("L_HIP", "R_HIP"),
    ("L_HIP", "L_KNEE"),
    ("L_KNEE", "L_ANKLE"),
    ("R_HIP", "R_KNEE"),
    ("R_KNEE", "R_ANKLE"),
]

# ============================
# UTILS
# ============================
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

def clamp(x, lo, hi):
    if np.isnan(x):
        return x
    return max(lo, min(hi, x))

def ema_point(prev, x, alpha):
    if prev is None:
        return x
    return (1 - alpha) * prev + alpha * x

def score_min_for(name):
    if "ANKLE" in name:
        return SCORE_MIN_ANKLE
    if "KNEE" in name:
        return SCORE_MIN_KNEE
    if "HIP" in name:
        return SCORE_MIN_HIP
    return SCORE_MIN_DEFAULT

def kp_alpha_for_score(score):
    if score is None or not np.isfinite(score):
        return KP_ALPHA_MIN
    s = float(score)
    if s <= 0.0:
        return KP_ALPHA_MIN
    t = (s - KP_SCORE_REF) / max(1e-6, (0.25 - KP_SCORE_REF))
    t = max(0.0, min(1.0, t))
    return float(KP_ALPHA_MIN + t * (KP_EMA_ALPHA - KP_ALPHA_MIN))

def greedy_edge_coloring(edges):
    edge_nodes = [(a, b) for a, b in edges]
    adj = [[] for _ in range(len(edge_nodes))]
    for i in range(len(edge_nodes)):
        ai, bi = edge_nodes[i]
        si = {ai, bi}
        for j in range(i + 1, len(edge_nodes)):
            aj, bj = edge_nodes[j]
            if len(si.intersection({aj, bj})) > 0:
                adj[i].append(j)
                adj[j].append(i)

    palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
        (128, 0, 128),
        (0, 128, 255),
        (128, 128, 0),
    ]

    order = sorted(range(len(edge_nodes)), key=lambda i: len(adj[i]), reverse=True)
    color_idx = [-1] * len(edge_nodes)

    for e in order:
        used = set(color_idx[n] for n in adj[e] if color_idx[n] != -1)
        c = 0
        while c in used:
            c += 1
        if c >= len(palette):
            palette.append(((37 * c) % 256, (91 * c) % 256, (183 * c) % 256))
        color_idx[e] = c

    return [palette[color_idx[i]] for i in range(len(edge_nodes))]

EDGE_COLORS = greedy_edge_coloring(EDGES)

def select_keypoint_tensor(outs):
    for o in outs:
        if isinstance(o, np.ndarray) and o.shape[-2:] == (17, 3):
            return o
    return None

def swap_lr(pts, scr):
    for a, b in LEFT_RIGHT_SWAPS:
        pts[a], pts[b] = pts[b], pts[a]
        scr[a], scr[b] = scr[b], scr[a]

def round_deg(x):
    if x is None or not np.isfinite(x):
        return None
    return int(round(float(x)))

def fmt_int(x):
    return "--" if x is None else f"{x:d}°"

# ============================
# FULL-FRAME LETTERBOX PREPROCESS
# ============================
def preprocess_full_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    side = int(max(h, w))

    pad_x = (side - w) // 2
    pad_y = (side - h) // 2

    sq = np.zeros((side, side, 3), dtype=rgb.dtype)
    sq[pad_y:pad_y + h, pad_x:pad_x + w] = rgb

    resized = cv2.resize(sq, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.int32)
    x = np.expand_dims(x, axis=0)  # NHWC

    return x, (h, w, pad_y, pad_x, side)

def extract_points_and_scores_full_frame(kps, meta):
    h, w, pad_y, pad_x, side = meta
    k = kps[0, 0] if kps.ndim == 4 else kps[0]
    pts, scr = {}, {}
    for i, name in enumerate(KP_NAMES):
        y, x, s = k[i]
        px_sq = float(x) * side
        py_sq = float(y) * side
        px = px_sq - pad_x
        py = py_sq - pad_y
        px = max(0.0, min(w - 1.0, px))
        py = max(0.0, min(h - 1.0, py))
        pts[name] = np.array([px, py], dtype=np.float32)
        scr[name] = float(s)
    return pts, scr

# ============================
# ANGLE FILTER (alpha-beta)
# ============================
class AlphaBeta1D:
    def __init__(self, alpha=0.88, beta=0.22):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.x = None
        self.v = 0.0
        self.t_last = None

    def update(self, z, t, use_meas=True):
        if self.x is None or self.t_last is None:
            self.x = z if (z is not None and np.isfinite(z)) else np.nan
            self.v = 0.0
            self.t_last = t
            return self.x

        dt = max(1e-3, float(t - self.t_last))
        x_pred = self.x + self.v * dt if np.isfinite(self.x) else (z if (use_meas and z is not None and np.isfinite(z)) else np.nan)
        v_pred = self.v

        if (not use_meas) or (z is None) or (not np.isfinite(z)) or (not np.isfinite(x_pred)):
            self.x = x_pred
            self.v = v_pred
            self.t_last = t
            return self.x

        r = float(z - x_pred)
        self.x = x_pred + self.alpha * r
        self.v = v_pred + (self.beta * r / dt)
        self.t_last = t
        return self.x

# ============================
# DASHBOARD UI
# ============================
def panel_bg(h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (16, 16, 16)
    return img

def draw_panel_header(panel, title, subtitle=None):
    _, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w - 1, 68), (26, 26, 26), -1)
    cv2.line(panel, (0, 68), (w - 1, 68), (45, 45, 45), 1)
    cv2.putText(panel, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (245, 245, 245), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (195, 195, 195), 1, cv2.LINE_AA)

def draw_stat_box(panel, x, y, w, h, label, value):
    cv2.rectangle(panel, (x, y), (x + w, y + h), (22, 22, 22), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (45, 45, 45), 1)
    cv2.putText(panel, label, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(panel, value, (x + 12, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (245, 245, 245), 2, cv2.LINE_AA)

def draw_lr_table(panel, x, y, w, row_h, title, rows):
    # rows: list of (label, L_value, R_value)
    h = row_h * (len(rows) + 1) + 6
    cv2.rectangle(panel, (x, y), (x + w, y + h), (22, 22, 22), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (45, 45, 45), 1)

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
        cv2.putText(panel, lv, (col_L, yy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(panel, rv, (col_R, yy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)

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
# LOAD MODEL
# ============================
model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, revision=HF_REV)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = max(1, (cv2.getNumberOfCPUs() or 4) // 2)
so.inter_op_num_threads = 1

sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
INP_NAME = sess.get_inputs()[0].name

_warm = np.zeros((1, IN_SIZE, IN_SIZE, 3), dtype=np.int32)
for _ in range(3):
    _ = sess.run(None, {INP_NAME: _warm})

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

pts_s = {k: None for k in KP_NAMES}
scr_s = {k: 0.0 for k in KP_NAMES}

# Filters for 4 angles (L/R hip and knee)
flt = {
    "L_HIP": AlphaBeta1D(AB_ALPHA, AB_BETA),
    "L_KNEE": AlphaBeta1D(AB_ALPHA, AB_BETA),
    "R_HIP": AlphaBeta1D(AB_ALPHA, AB_BETA),
    "R_KNEE": AlphaBeta1D(AB_ALPHA, AB_BETA),
}
last_meas = {k: None for k in flt.keys()}

show_skeleton = SHOW_SKELETON
show_joints = SHOW_JOINTS
show_scores = SHOW_SCORES
show_console = SHOW_CONSOLE
show_panel = True
show_video = True

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

        x_in, meta = preprocess_full_frame(frame)
        outs = sess.run(None, {INP_NAME: x_in})
        kps = select_keypoint_tensor(outs)
        if kps is None:
            continue
        raw_pts, raw_scr = extract_points_and_scores_full_frame(kps, meta)

        if MIRROR_VIEW:
            swap_lr(raw_pts, raw_scr)

        if pts_s["NOSE"] is None:
            for k in KP_NAMES:
                pts_s[k] = raw_pts[k].copy()
                scr_s[k] = raw_scr[k]

        # Keypoint smoothing (responsive)
        for k in KP_NAMES:
            s = raw_scr[k]
            scr_s[k] = float(s)
            a = kp_alpha_for_score(s)
            pts_s[k] = ema_point(pts_s[k], raw_pts[k], a)

        infer_ms = (time.time() - t_inf0) * 1000.0

        frame_i += 1
        if frame_i % 10 == 0:
            fps = frame_i / max(1e-6, (time.time() - t0))

        # ============================
        # ANGLES (2D projected)
        # Hip: shoulder-hip-knee
        # Knee: hip-knee-ankle
        # ============================
        now_t = time.time()

        def meas_if_conf(a, b, c):
            if scr_s.get(a, 0.0) >= score_min_for(a) and scr_s.get(b, 0.0) >= score_min_for(b) and scr_s.get(c, 0.0) >= score_min_for(c):
                return clamp(angle_deg(pts_s[a], pts_s[b], pts_s[c]), ANGLE_MIN, ANGLE_MAX)
            return np.nan

        L_hip_m = meas_if_conf("L_SHOULDER", "L_HIP", "L_KNEE")
        L_knee_m = meas_if_conf("L_HIP", "L_KNEE", "L_ANKLE")
        R_hip_m = meas_if_conf("R_SHOULDER", "R_HIP", "R_KNEE")
        R_knee_m = meas_if_conf("R_HIP", "R_KNEE", "R_ANKLE")

        def update_angle(key, meas):
            lm = last_meas[key]
            if lm is not None and np.isfinite(meas) and np.isfinite(lm):
                if abs(float(meas) - float(lm)) > MAX_DEG_PER_FRAME:
                    meas = lm
            if np.isfinite(meas):
                last_meas[key] = float(meas)
            out = flt[key].update(meas if np.isfinite(meas) else None, now_t, use_meas=np.isfinite(meas))
            return out

        L_hip = update_angle("L_HIP", L_hip_m)
        L_knee = update_angle("L_KNEE", L_knee_m)
        R_hip = update_angle("R_HIP", R_hip_m)
        R_knee = update_angle("R_KNEE", R_knee_m)

        L_hip_i = round_deg(L_hip)
        L_knee_i = round_deg(L_knee)
        R_hip_i = round_deg(R_hip)
        R_knee_i = round_deg(R_knee)

        # ============================
        # DRAW (right pane only)
        # ============================
        video = frame.copy()

        if show_video:
            if show_skeleton:
                for (a, b), col in zip(EDGES, EDGE_COLORS):
                    sa = scr_s.get(a, 0.0)
                    sb = scr_s.get(b, 0.0)
                    if sa >= score_min_for(a) and sb >= score_min_for(b):
                        cv2.line(video, tuple(pts_s[a].astype(int)), tuple(pts_s[b].astype(int)), col, 3)

            if show_joints:
                for k in KP_NAMES:
                    s = scr_s.get(k, 0.0)
                    if s >= score_min_for(k):
                        p = tuple(pts_s[k].astype(int))
                        radius = 6 if "ANKLE" in k else 4
                        cv2.circle(video, p, radius, (245, 245, 245), -1)
                        if show_scores:
                            cv2.putText(video, f"{s:.2f}", (p[0] + 6, p[1] - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        else:
            video[:] = (8, 8, 8)

        # ============================
        # BUILD DASHBOARD
        # ============================
        dash = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        dash[:] = (10, 10, 10)

        pane_w = VIEW_W - PANEL_W
        pane_h = VIEW_H

        if show_panel:
            panel = panel_bg(VIEW_H, PANEL_W)
            draw_panel_header(panel, "Pose Dashboard", "Keys: q quit | v video | u panel | s skel | j joints | p scores | l console")

            col1_x = 16
            col2_x = PANEL_W // 2 + 6
            box_w = PANEL_W // 2 - 22
            y = 86
            box_h = 74
            gap_y = 14

            draw_stat_box(panel, col1_x, y, box_w, box_h, "FPS", f"{fps:0.1f}")
            draw_stat_box(panel, col2_x, y, box_w, box_h, "Infer (ms)", f"{infer_ms:0.1f}")
            y += box_h + gap_y

            draw_lr_table(
                panel, 16, y, PANEL_W - 32, 46, "Angles (deg)",
                rows=[
                    ("Hip",  fmt_int(L_hip_i),  fmt_int(R_hip_i)),
                    ("Knee", fmt_int(L_knee_i), fmt_int(R_knee_i)),
                ],
            )

            dash[:, :PANEL_W] = panel
            cv2.line(dash, (PANEL_W, 0), (PANEL_W, VIEW_H - 1), (45, 45, 45), 1)
        else:
            # no panel: video uses full width
            PANEL_W = 0
            pane_w = VIEW_W

        video_pane = fit_video_to_pane(video, pane_w - 2 * VIDEO_PAD, pane_h - 2 * VIDEO_PAD)

        x_off = PANEL_W + VIDEO_PAD
        dash[VIDEO_PAD:VIDEO_PAD + video_pane.shape[0], x_off:x_off + video_pane.shape[1]] = video_pane

        # Console
        if show_console and (time.time() - last_log) >= LOG_INTERVAL:
            msg = (
                f"[{time.time() - t0:7.2f}s] FPS={fps:4.1f} infer={infer_ms:5.1f}ms | "
                f"Hip L/R {fmt_int(L_hip_i)}/{fmt_int(R_hip_i)}  "
                f"Knee L/R {fmt_int(L_knee_i)}/{fmt_int(R_knee_i)}"
            )
            sys.stdout.write("\r" + msg + " " * 10)
            sys.stdout.flush()
            last_log = time.time()

        cv2.imshow(WINDOW, dash)

        key = (cv2.waitKey(1) & 0xFF)
        if key == ord("q"):
            break
        elif key in (ord("v"), ord("V")):
            show_video = not show_video
        elif key in (ord("u"), ord("U")):
            show_panel = not show_panel
        elif key in (ord("s"), ord("S")):
            show_skeleton = not show_skeleton
        elif key in (ord("j"), ord("J")):
            show_joints = not show_joints
        elif key in (ord("p"), ord("P")):
            show_scores = not show_scores
        elif key in (ord("l"), ord("L")):
            show_console = not show_console
            if not show_console:
                sys.stdout.write("\n")
                sys.stdout.flush()

finally:
    cap.release()
    cv2.destroyAllWindows()
    if SHOW_CONSOLE:
        sys.stdout.write("\n")
        sys.stdout.flush()
