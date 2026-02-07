"""Skeleton and joint drawing on video frame."""

import cv2

from config import JOINT_DOT_R, JOINT_OUTLINE_R, JOINT_OUTLINE_THICK
from landmarks import EDGES, EDGE_COLORS, LEFT_IDXS, RIGHT_IDXS, edge_thickness
from pose_processor import vis_ok


def joint_colors(idx):
    outline = (235, 235, 235)
    if idx in LEFT_IDXS:
        dot = (180, 170, 120)
    elif idx in RIGHT_IDXS:
        dot = (160, 140, 200)
    else:
        dot = (200, 200, 200)
    return outline, dot


def draw_joint_marker(img, p_xy, idx):
    x, y = int(p_xy[0]), int(p_xy[1])
    outline, dot = joint_colors(idx)
    cv2.circle(img, (x, y), JOINT_OUTLINE_R, outline, JOINT_OUTLINE_THICK, cv2.LINE_AA)
    cv2.circle(img, (x, y), JOINT_DOT_R, dot, -1, cv2.LINE_AA)


def draw_skeleton_on_video(video, pts, vis, show_skeleton, show_joints, show_vis):
    """Draw skeleton edges and joint markers on video frame."""
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
