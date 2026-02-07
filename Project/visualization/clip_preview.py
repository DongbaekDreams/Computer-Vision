"""Clip preview: animated pose skeleton in a box."""

import cv2
import numpy as np

from config import VIS_MIN
from landmarks import EDGES, EDGE_COLORS, edge_thickness
from pose_processor import vis_ok
from visualization.skeleton import joint_colors


def draw_pose_clip(canvas_bgr, pts_norm, vis_arr, title="Clip (loop)"):
    h, w = canvas_bgr.shape[:2]
    canvas_bgr[:] = (10, 10, 10)

    cv2.putText(canvas_bgr, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    box_x0, box_y0 = 10, 32
    box_x1, box_y1 = w - 10, h - 10
    cv2.rectangle(canvas_bgr, (box_x0, box_y0), (box_x1, box_y1), (14, 14, 14), -1)
    cv2.rectangle(canvas_bgr, (box_x0, box_y0), (box_x1, box_y1), (45, 45, 45), 1)

    if pts_norm is None or vis_arr is None:
        cv2.putText(canvas_bgr, "no pose", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)
        return

    pts_norm = pts_norm.astype(np.float32)
    vis_arr = vis_arr.astype(np.float32)

    ok = np.isfinite(pts_norm[:, 0]) & np.isfinite(pts_norm[:, 1]) & (vis_arr >= VIS_MIN)
    if int(ok.sum()) < 2:
        cv2.putText(canvas_bgr, "no pose", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)
        return

    xs = pts_norm[ok, 0]
    ys = pts_norm[ok, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    pad_norm = 0.06
    xmin -= pad_norm
    xmax += pad_norm
    ymin -= pad_norm
    ymax += pad_norm

    bw = max(1e-6, xmax - xmin)
    bh = max(1e-6, ymax - ymin)

    inner_w = max(1, (box_x1 - box_x0 - 14))
    inner_h = max(1, (box_y1 - box_y0 - 14))

    scale = min(inner_w / bw, inner_h / bh)
    tx = box_x0 + 7 + (inner_w - bw * scale) * 0.5
    ty = box_y0 + 7 + (inner_h - bh * scale) * 0.5

    pts = np.empty((33, 2), dtype=np.float32)
    pts[:, 0] = tx + (pts_norm[:, 0] - xmin) * scale
    pts[:, 1] = ty + (pts_norm[:, 1] - ymin) * scale

    pts[:, 0] = np.clip(pts[:, 0], box_x0 + 2, box_x1 - 2)
    pts[:, 1] = np.clip(pts[:, 1], box_y0 + 2, box_y1 - 2)

    for (u, v), col in zip(EDGES, EDGE_COLORS):
        if vis_ok(float(vis_arr[u])) and vis_ok(float(vis_arr[v])):
            cv2.line(
                canvas_bgr,
                (int(pts[u, 0]), int(pts[u, 1])),
                (int(pts[v, 0]), int(pts[v, 1])),
                col,
                max(2, edge_thickness(u, v) - 2),
                cv2.LINE_AA,
            )

    for idx in range(pts.shape[0]):
        if vis_ok(float(vis_arr[idx])):
            x, y = int(pts[idx, 0]), int(pts[idx, 1])
            outline, dot = joint_colors(idx)
            cv2.circle(canvas_bgr, (x, y), 5, outline, 2, cv2.LINE_AA)
            cv2.circle(canvas_bgr, (x, y), 2, dot, -1, cv2.LINE_AA)
