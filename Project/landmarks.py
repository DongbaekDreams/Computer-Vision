"""Landmark indices and skeleton edge definitions for MediaPipe pose (33 landmarks)."""

from config import EDGE_THICK, FOOT_THICK

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
COL_ARMS = (200, 150, 110)
COL_LEGS = (140, 190, 140)
COL_FEET = (105, 160, 105)

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
