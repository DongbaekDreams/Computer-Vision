"""Application state: buffers, display toggles, timing."""

from collections import deque

from pose_processor import ANGLE_KEYS

# Live buffer (app-time seconds)
t_live = deque()
pose_live = deque()
angles_live = {k: deque() for k in ANGLE_KEYS}

# Recording buffer (record-time seconds)
t_rec = deque()
pose_rec = deque()
angles_rec = {k: deque() for k in ANGLE_KEYS}


def clear_recording():
    t_rec.clear()
    pose_rec.clear()
    for dq in angles_rec.values():
        dq.clear()
