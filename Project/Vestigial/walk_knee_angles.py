import time
import math
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# ============================
# CONFIG
# ============================
CAM_INDEX = 0
WINDOW = "MoveNet ONNX - Knee Angles (stable/fast) (q to quit)"

# Keypoint smoothing (EMA on x,y). Higher = more responsive.
KP_EMA_ALPHA = 0.8

# Bone-length warp gate
LEN_BASELINE_ALPHA = 0.20        # baseline adapts faster
LEN_DEV_FRAC = 0.60              # allow more deviation before rejecting

# Angle smoothing (small; keypoints already smoothed)
ANGLE_EMA_ALPHA = 0.20

# Angle clamp and velocity clamp
ANGLE_MIN = 0.0
ANGLE_MAX = 180.0
MAX_DEG_PER_FRAME = 60.0         # allow faster changes

# Console logging rate (seconds)
LOG_INTERVAL = 1.0

# Xenova MoveNet ONNX (NHWC int32 input)
HF_REPO = "Xenova/movenet-singlepose-lightning"
HF_FILE = "onnx/model.onnx"
HF_REV = "main"
IN_SIZE = 192

# MoveNet keypoint indices
KP = {
    "L_HIP": 11, "R_HIP": 12,
    "L_KNEE": 13, "R_KNEE": 14,
    "L_ANKLE": 15, "R_ANKLE": 16,
}

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

def ema_point(prev, x, alpha):
    if prev is None:
        return x
    return (1 - alpha) * prev + alpha * x

def ema_scalar(prev, x, alpha):
    if prev is None or np.isnan(prev):
        return x
    if np.isnan(x):
        return prev
    return (1 - alpha) * prev + alpha * x

def ema_len(prev, x, alpha):
    if prev is None or not np.isfinite(prev):
        return x
    return (1 - alpha) * prev + alpha * x

def fmt(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}"

def clamp(x, lo, hi):
    if np.isnan(x):
        return x
    return max(lo, min(hi, x))

def center_crop_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img, (h, w, 0, 0, h)
    if h > w:
        y0 = (h - w) // 2
        return img[y0:y0 + w], (h, w, y0, 0, w)
    x0 = (w - h) // 2
    return img[:, x0:x0 + h], (h, w, 0, x0, h)

def preprocess(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crop, meta = center_crop_square(img)
    resized = cv2.resize(crop, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.int32)
    x = np.expand_dims(x, axis=0)  # NHWC
    return x, meta

def select_keypoint_tensor(outs):
    for o in outs:
        if isinstance(o, np.ndarray) and o.shape[-2:] == (17, 3):
            return o
    return None

def extract_points(kps, meta):
    h, w, y0, x0, side = meta
    k = kps[0, 0] if kps.ndim == 4 else kps[0]
    pts = {}
    for name, idx in KP.items():
        y, x, _ = k[idx]
        px = float(x) * side + x0
        py = float(y) * side + y0
        px = max(0.0, min(w - 1.0, px))
        py = max(0.0, min(h - 1.0, py))
        pts[name] = np.array([px, py], dtype=np.float32)
    return pts

def seg_len(a, b):
    return float(np.linalg.norm(a - b))

def draw_text_block(img, lines, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.85
    thick = 2
    pad = 8
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    bw = max(s[0] for s in sizes) + pad * 2
    bh = sum(s[1] for s in sizes) + pad * (len(lines) + 1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 0), -1)
    yy = y + pad + sizes[0][1]
    for t, s in zip(lines, sizes):
        cv2.putText(img, t, (x + pad, yy), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        yy += s[1] + pad

# ============================
# LOAD MODEL
# ============================
model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, revision=HF_REV)
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
INP_NAME = sess.get_inputs()[0].name

# ============================
# CAMERA
# ============================
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera not available")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

# ============================
# STATE
# ============================
t0 = time.time()
last_log = 0.0
frame_i = 0
fps = 0.0

pts_s = {k: None for k in KP.keys()}

L_thigh_base = None
L_shank_base = None
R_thigh_base = None
R_shank_base = None

L_angle_s = None
R_angle_s = None
L_angle_last = None
R_angle_last = None

reject_L = 0
reject_R = 0
accept_L = 0
accept_R = 0

# ============================
# MAIN LOOP
# ============================
try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = cv2.flip(frame, 1)
        t_inf0 = time.time()

        x, meta = preprocess(frame)
        outs = sess.run(None, {INP_NAME: x})
        kps = select_keypoint_tensor(outs)
        if kps is None:
            continue

        raw = extract_points(kps, meta)

        if pts_s["L_HIP"] is None:
            for k in pts_s.keys():
                pts_s[k] = raw[k].copy()
            L_thigh_base = seg_len(pts_s["L_HIP"], pts_s["L_KNEE"])
            L_shank_base = seg_len(pts_s["L_KNEE"], pts_s["L_ANKLE"])
            R_thigh_base = seg_len(pts_s["R_HIP"], pts_s["R_KNEE"])
            R_shank_base = seg_len(pts_s["R_KNEE"], pts_s["R_ANKLE"])

        L_thigh = seg_len(raw["L_HIP"], raw["L_KNEE"])
        L_shank = seg_len(raw["L_KNEE"], raw["L_ANKLE"])
        R_thigh = seg_len(raw["R_HIP"], raw["R_KNEE"])
        R_shank = seg_len(raw["R_KNEE"], raw["R_ANKLE"])

        L_thigh_base = ema_len(L_thigh_base, L_thigh, LEN_BASELINE_ALPHA)
        L_shank_base = ema_len(L_shank_base, L_shank, LEN_BASELINE_ALPHA)
        R_thigh_base = ema_len(R_thigh_base, R_thigh, LEN_BASELINE_ALPHA)
        R_shank_base = ema_len(R_shank_base, R_shank, LEN_BASELINE_ALPHA)

        def ok_len(xv, base):
            if base is None or base < 1e-6:
                return True
            return abs(xv - base) <= LEN_DEV_FRAC * base

        L_ok = ok_len(L_thigh, L_thigh_base) and ok_len(L_shank, L_shank_base)
        R_ok = ok_len(R_thigh, R_thigh_base) and ok_len(R_shank, R_shank_base)

        if L_ok:
            accept_L += 1
            pts_s["L_HIP"] = ema_point(pts_s["L_HIP"], raw["L_HIP"], KP_EMA_ALPHA)
            pts_s["L_KNEE"] = ema_point(pts_s["L_KNEE"], raw["L_KNEE"], KP_EMA_ALPHA)
            pts_s["L_ANKLE"] = ema_point(pts_s["L_ANKLE"], raw["L_ANKLE"], KP_EMA_ALPHA)
        else:
            reject_L += 1

        if R_ok:
            accept_R += 1
            pts_s["R_HIP"] = ema_point(pts_s["R_HIP"], raw["R_HIP"], KP_EMA_ALPHA)
            pts_s["R_KNEE"] = ema_point(pts_s["R_KNEE"], raw["R_KNEE"], KP_EMA_ALPHA)
            pts_s["R_ANKLE"] = ema_point(pts_s["R_ANKLE"], raw["R_ANKLE"], KP_EMA_ALPHA)
        else:
            reject_R += 1

        L_ang = angle_deg(pts_s["L_HIP"], pts_s["L_KNEE"], pts_s["L_ANKLE"])
        R_ang = angle_deg(pts_s["R_HIP"], pts_s["R_KNEE"], pts_s["R_ANKLE"])

        L_ang = clamp(L_ang, ANGLE_MIN, ANGLE_MAX)
        R_ang = clamp(R_ang, ANGLE_MIN, ANGLE_MAX)

        if L_angle_last is not None and not np.isnan(L_ang) and not np.isnan(L_angle_last):
            if abs(L_ang - L_angle_last) > MAX_DEG_PER_FRAME:
                L_ang = L_angle_last
        if R_angle_last is not None and not np.isnan(R_ang) and not np.isnan(R_angle_last):
            if abs(R_ang - R_angle_last) > MAX_DEG_PER_FRAME:
                R_ang = R_angle_last

        if not np.isnan(L_ang):
            L_angle_last = L_ang
        if not np.isnan(R_ang):
            R_angle_last = R_ang

        L_angle_s = ema_scalar(L_angle_s, L_ang, ANGLE_EMA_ALPHA)
        R_angle_s = ema_scalar(R_angle_s, R_ang, ANGLE_EMA_ALPHA)

        infer_ms = (time.time() - t_inf0) * 1000.0

        frame_i += 1
        if frame_i % 10 == 0:
            fps = frame_i / max(1e-6, (time.time() - t0))

        for a, b in [
            ("L_HIP", "L_KNEE"), ("L_KNEE", "L_ANKLE"),
            ("R_HIP", "R_KNEE"), ("R_KNEE", "R_ANKLE"),
        ]:
            cv2.line(frame, tuple(pts_s[a].astype(int)), tuple(pts_s[b].astype(int)), (0, 255, 0), 3)

        draw_text_block(frame, [
            f"FPS: {fps:.1f}   infer: {infer_ms:.1f} ms",
            f"KP EMA: {KP_EMA_ALPHA:.2f}  len_dev<= {LEN_DEV_FRAC:.2f}  vel<= {MAX_DEG_PER_FRAME:.0f} deg/frame",
            f"L knee: {fmt(L_angle_s)} deg   (acc {accept_L} rej {reject_L})",
            f"R knee: {fmt(R_angle_s)} deg   (acc {accept_R} rej {reject_R})",
        ], 10, 10)

        now = time.time()
        if now - last_log >= LOG_INTERVAL:
            print(
                f"[{now - t0:7.2f}s] FPS={fps:4.1f} infer={infer_ms:5.1f}ms | "
                f"L={fmt(L_angle_s)} R={fmt(R_angle_s)} | "
                f"L_ok={int(L_ok)} R_ok={int(R_ok)}"
            )
            last_log = now

        # --- DEBUG: draw the exact inference crop box (what MoveNet sees) ---
        h0, w0 = frame.shape[:2]
        if h0 == w0:
            cv2.rectangle(frame, (0, 0), (w0 - 1, h0 - 1), (0, 0, 255), 2)
        elif h0 < w0:
            x0 = (w0 - h0) // 2
            cv2.rectangle(frame, (x0, 0), (x0 + h0 - 1, h0 - 1), (0, 0, 255), 2)
        else:
            y0 = (h0 - w0) // 2
            cv2.rectangle(frame, (0, y0), (w0 - 1, y0 + w0 - 1), (0, 0, 255), 2)

        cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
