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
PANEL_W = 500
VIEW_H = 1020
VIEW_W = 1600
VIDEO_PAD = 14

# Right-side plot + preview
PLOT_W = 600
PLOT_PAD = 14

# Live buffer + recording
MAX_REC_SECONDS = 60.0
LIVE_BUFFER_SECONDS = 60.0
SEG_SECONDS = 10.0

# Plot styling
PLOT_BG = (8, 8, 8)
PLOT_RING = (40, 40, 40)
PLOT_AXIS = (32, 32, 32)

# Global theme
WINDOW_BG = (10, 12, 18)
PANEL_BG = (18, 22, 30)
PANEL_HEADER_BG = (28, 33, 44)
PANEL_DIVIDER = (54, 62, 78)
SURFACE_BG = (24, 29, 39)
SURFACE_BG_ALT = (20, 24, 33)
SURFACE_ELEVATED = (31, 37, 49)
SURFACE_BORDER = (60, 70, 90)
SURFACE_BORDER_SOFT = (44, 52, 68)
TEXT_PRIMARY = (238, 242, 248)
TEXT_SECONDARY = (177, 188, 204)
TEXT_MUTED = (136, 148, 166)
TEXT_DISABLED = (107, 118, 134)
ACCENT = (239, 170, 84)
ACCENT_ALT = (189, 135, 255)
ACCENT_SUCCESS = (122, 205, 145)
ACCENT_WARNING = (94, 204, 228)
ACCENT_DANGER = (92, 96, 236)
ACCENT_SOFT = (70, 86, 118)
CARD_GLOW = (44, 58, 92)
VIDEO_BG = (8, 10, 14)

# Spacing + sizing
APP_PAD = 16
CARD_PAD = 14
CARD_GAP = 16
SECTION_GAP = 14
BTN_ROW_H = 30
CONTROL_ROW_H = 28
TIMELINE_H = 86

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

# Triangulated 3D preview (pose_3d_view): points live in chessboard / extrinsics world
# coordinates. Tweak if the skeleton looks rotated (e.g. facing sideways vs cameras).
POSE_3D_WORLD_Z_ROT_DEG = 90.0  # spin about board normal (+Z in OpenCV object frame); try -90 if wrong
POSE_3D_WORLD_Y_ROT_DEG = 0.0  # optional extra yaw about +Y

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
CLIP_BG = SURFACE_BG_ALT

# ============================
# TIMELINE UI
# ============================
TL_BG = SURFACE_BG
TL_BORDER = SURFACE_BORDER
TL_BAR = SURFACE_BG_ALT
TL_TICK = (114, 126, 146)
TL_TEXT = TEXT_PRIMARY
TL_TEXT_DIM = TEXT_MUTED

WIN_FILL = (56, 73, 105)
WIN_BORDER = (224, 233, 246)
WIN_HANDLE = (248, 250, 252)

PLAY_TICK = (245, 245, 245)
PLAY_TICK_DIM = TEXT_MUTED

BTN_BG = SURFACE_ELEVATED
BTN_BORDER = SURFACE_BORDER
BTN_TEXT = TEXT_PRIMARY
BTN_DIM = TEXT_DISABLED

BTN_RED = ACCENT_DANGER
BTN_GREEN = ACCENT_SUCCESS
BTN_YELLOW = ACCENT_WARNING
BTN_BLUE = ACCENT

UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_SCALE = 0.56
UI_SCALE_SMALL = 0.47
UI_SCALE_TINY = 0.40
TITLE_SCALE = 0.90
SUBTITLE_SCALE = 0.52
STAT_LABEL_SCALE = 0.50
STAT_VALUE_SCALE = 1.04
SECTION_TITLE_SCALE = 0.60
TABLE_TITLE_SCALE = 0.64
TABLE_LABEL_SCALE = 0.53
TABLE_VALUE_SCALE = 0.78
CONTROL_KEY_SCALE = 0.52
CONTROL_DESC_SCALE = 0.49

# Polar palette gallery
PALETTE_GALLERY_TITLE = "Polar Styles"
PALETTE_GALLERY_SUBTITLE = "Click a palette to recolor the graph"
PALETTE_GALLERY_CARD_W = 146
PALETTE_GALLERY_CARD_H = 112
PALETTE_GALLERY_CARD_GAP = 14
PALETTE_GALLERY_COLS = 3
PALETTE_GALLERY_BG = (17, 21, 29)
PALETTE_GALLERY_BORDER = (88, 100, 124)
PALETTE_GALLERY_PREVIEW_BG = (21, 26, 35)
PALETTE_GALLERY_SCRIM = (6, 8, 14)
PALETTE_GALLERY_TEXT = TEXT_PRIMARY
PALETTE_GALLERY_TEXT_MUTED = TEXT_SECONDARY
PALETTE_GALLERY_SELECTED = ACCENT
PALETTE_GALLERY_KEY = "g"


def _bgr(hex_rgb: str):
    hex_rgb = hex_rgb.lstrip("#")
    r = int(hex_rgb[0:2], 16)
    g = int(hex_rgb[2:4], 16)
    b = int(hex_rgb[4:6], 16)
    return (b, g, r)


POLAR_PALETTES = [
    {
        "key": "plasma",
        "label": "Plasma",
        "type": "opencv",
        "colormap": cv2.COLORMAP_PLASMA,
        "bg": _bgr("#10131B"),
        "ring": _bgr("#313848"),
        "axis": _bgr("#465065"),
        "marker": _bgr("#F3F6FA"),
        "marker_ring": _bgr("#95A1B5"),
        "center": _bgr("#E7EDF7"),
    },
    {
        "key": "viridis",
        "label": "Viridis",
        "type": "opencv",
        "colormap": cv2.COLORMAP_VIRIDIS,
        "bg": _bgr("#0F1419"),
        "ring": _bgr("#2C3B44"),
        "axis": _bgr("#40545F"),
        "marker": _bgr("#E8F7F3"),
        "marker_ring": _bgr("#8FB7AD"),
        "center": _bgr("#D4F0E7"),
    },
    {
        "key": "turbo",
        "label": "Turbo",
        "type": "opencv",
        "colormap": cv2.COLORMAP_TURBO,
        "bg": _bgr("#11131A"),
        "ring": _bgr("#374055"),
        "axis": _bgr("#4A5D78"),
        "marker": _bgr("#F8F3EC"),
        "marker_ring": _bgr("#BFAF9B"),
        "center": _bgr("#F3E3CF"),
    },
    {
        "key": "ocean",
        "label": "Ocean",
        "type": "gradient",
        "colors": [_bgr("#71F7F0"), _bgr("#38B6FF"), _bgr("#5F6BFF"), _bgr("#D76BFF")],
        "bg": _bgr("#0E1420"),
        "ring": _bgr("#2A3A4E"),
        "axis": _bgr("#41617B"),
        "marker": _bgr("#EAF6FF"),
        "marker_ring": _bgr("#91B1CE"),
        "center": _bgr("#D6EDFF"),
    },
    {
        "key": "ember",
        "label": "Ember",
        "type": "gradient",
        "colors": [_bgr("#FCE38A"), _bgr("#F38181"), _bgr("#C06C84"), _bgr("#6C5B7B")],
        "bg": _bgr("#171116"),
        "ring": _bgr("#48313D"),
        "axis": _bgr("#6B4455"),
        "marker": _bgr("#FFF0E5"),
        "marker_ring": _bgr("#C79590"),
        "center": _bgr("#FFE4D4"),
    },
    {
        "key": "mint",
        "label": "Mint",
        "type": "gradient",
        "colors": [_bgr("#E3FFE7"), _bgr("#B7F7D8"), _bgr("#71D7C7"), _bgr("#2F9C95")],
        "bg": _bgr("#101816"),
        "ring": _bgr("#314743"),
        "axis": _bgr("#45655F"),
        "marker": _bgr("#F3FFF8"),
        "marker_ring": _bgr("#A2CABA"),
        "center": _bgr("#DDF9EC"),
    },
    {
        "key": "mono",
        "label": "Mono Ice",
        "type": "gradient",
        "colors": [_bgr("#EAF2FF"), _bgr("#CAD7F2"), _bgr("#97A8C7"), _bgr("#64748B")],
        "bg": _bgr("#101318"),
        "ring": _bgr("#343B48"),
        "axis": _bgr("#4A5568"),
        "marker": _bgr("#FFFFFF"),
        "marker_ring": _bgr("#9AA6B8"),
        "center": _bgr("#E1E7F0"),
    },
    {
        "key": "sunset",
        "label": "Sunset",
        "type": "gradient",
        "colors": [_bgr("#FFD166"), _bgr("#F4978E"), _bgr("#845EC2"), _bgr("#2C73D2")],
        "bg": _bgr("#15131B"),
        "ring": _bgr("#433A55"),
        "axis": _bgr("#635574"),
        "marker": _bgr("#FFF4EA"),
        "marker_ring": _bgr("#C4A7B9"),
        "center": _bgr("#FFE6C7"),
    },
    {
        "key": "aurora",
        "label": "Aurora",
        "type": "gradient",
        "colors": [_bgr("#8EF6E4"), _bgr("#5BC0EB"), _bgr("#A16AE8"), _bgr("#FF8FAB")],
        "bg": _bgr("#111521"),
        "ring": _bgr("#314058"),
        "axis": _bgr("#4A6488"),
        "marker": _bgr("#F7FBFF"),
        "marker_ring": _bgr("#93AECF"),
        "center": _bgr("#E3F1FF"),
    },
]

DEFAULT_POLAR_PALETTE_KEY = POLAR_PALETTES[0]["key"]


def get_polar_palette(key: str):
    """Return a polar palette by key, defaulting to the first preset."""
    for palette in POLAR_PALETTES:
        if palette["key"] == key:
            return palette
    return POLAR_PALETTES[0]


def ensure_task_file(path: str, url: str):
    """Download MediaPipe task file if missing or empty."""
    import os
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    urllib.request.urlretrieve(url, path)
