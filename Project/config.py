"""Configuration constants for the pose dashboard application."""

import urllib.request
from pathlib import Path

import cv2

# Resolve paths relative to this package (works regardless of cwd)
_PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = _PROJECT_DIR / "models"

# ============================
# CAMERA & WINDOW
# ============================
CAM_INDEX = 0
WINDOW = "MediaPipe Pose (Tasks) - Dashboard (q to quit)"

# Dashboard layout
PANEL_W = 420
VIEW_H = 720
VIEW_W = 1280
VIDEO_PAD = 12

# Right-side plot + preview
PLOT_W = 480
PLOT_PAD = 12

# Live buffer + recording
MAX_REC_SECONDS = 60.0
LIVE_BUFFER_SECONDS = 60.0
SEG_SECONDS = 10.0

# Plot styling
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

# Tasks model asset (.task) - resolved relative to package
TASK_PATH = str(MODELS_DIR / "pose_landmarker.task")
TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

# Style
EDGE_THICK = 4
FOOT_THICK = 5
JOINT_OUTLINE_THICK = 2
JOINT_OUTLINE_R = 7
JOINT_DOT_R = 2

# Clip preview (bottom)
CLIP_H_FRAC = 0.34
CLIP_BG = (10, 10, 10)

# ============================
# TIMELINE UI
# ============================
TIMELINE_H = 74
TL_BG = (18, 18, 18)
TL_BORDER = (45, 45, 45)
TL_BAR = (26, 26, 26)
TL_TICK = (90, 90, 90)
TL_TEXT = (210, 210, 210)
TL_TEXT_DIM = (150, 150, 150)

WIN_FILL = (40, 40, 40)
WIN_BORDER = (210, 210, 210)
WIN_HANDLE = (245, 245, 245)

PLAY_TICK = (245, 245, 245)
PLAY_TICK_DIM = (160, 160, 160)

BTN_BG = (24, 24, 24)
BTN_BORDER = (55, 55, 55)
BTN_TEXT = (235, 235, 235)
BTN_DIM = (150, 150, 150)

BTN_RED = (60, 60, 220)     # BGR
BTN_GREEN = (120, 200, 120)
BTN_YELLOW = (80, 200, 220)
BTN_BLUE = (220, 160, 60)   # mode highlight (BGR)

UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_SCALE = 0.46
UI_SCALE_SMALL = 0.40


def ensure_task_file(path: str, url: str):
    """Download MediaPipe task file if missing or empty."""
    import os
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    urllib.request.urlretrieve(url, path)
