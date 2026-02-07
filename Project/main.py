"""Pose dashboard: MediaPipe pose estimation with angle tracking and recording."""

import sys
import time

import cv2
import numpy as np

from config import (
    CAM_H,
    CAM_INDEX,
    CAM_W,
    CLIP_H_FRAC,
    LIVE_BUFFER_SECONDS,
    LOG_INTERVAL,
    MAX_REC_SECONDS,
    MIRROR_VIEW,
    PANEL_W,
    PLOT_PAD,
    PLOT_W,
    SEG_SECONDS,
    SHOW_CAMERA_BG,
    SHOW_CONSOLE,
    SHOW_JOINTS,
    SHOW_SKELETON,
    SHOW_VIS,
    TASK_PATH,
    TASK_URL,
    TIMELINE_H,
    VIDEO_PAD,
    VIEW_H,
    VIEW_W,
    WINDOW,
    ensure_task_file,
)
from pose_processor import ANGLE_KEYS, process_pose, round_deg
from state import angles_live, angles_rec, pose_live, pose_rec, t_live, t_rec
from ui.console import console_line
from ui.drawing import (
    draw_controls_section,
    draw_lr_table,
    draw_panel_header,
    draw_stat_box,
    fit_video_to_pane,
    panel_bg,
)
from ui.timeline import (
    clear_hitboxes,
    draw_timeline_ui,
    extract_segment_by_time,
    make_mouse_cb,
    trim_time_buffer,
)
from visualization.clip_preview import draw_pose_clip
from visualization.polar_plot import draw_polar_plot_segment
from visualization.skeleton import draw_skeleton_on_video

# MediaPipe
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "MediaPipe Tasks API not available. Use Python 3.11 and:\n"
        "  pip install mediapipe\n"
    ) from e

# Timeline state (live_mode, recording, etc.)
from ui import timeline as timeline_module


def main():
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

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, VIEW_W, VIEW_H)
    cv2.setMouseCallback(WINDOW, make_mouse_cb())

    # State
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
            L_sho_i = R_sho_i = L_elb_i = R_elb_i = None

            vals = {k: np.nan for k in ANGLE_KEYS}
            pts_norm_snapshot = None
            vis_snapshot = None

            lm = result.pose_landmarks[0] if (result.pose_landmarks and len(result.pose_landmarks) > 0) else None
            h, w = frame.shape[:2]
            out = process_pose(lm, h, w)

            if out[0] is not None:
                pts, vis, _, _, vals, pts_norm_snapshot, vis_snapshot = out
                L_hip_i, R_hip_i = round_deg(vals["Hip L"]), round_deg(vals["Hip R"])
                L_knee_i, R_knee_i = round_deg(vals["Knee L"]), round_deg(vals["Knee R"])
                L_ank_i, R_ank_i = round_deg(vals["Ank L"]), round_deg(vals["Ank R"])
                L_sho_i, R_sho_i = round_deg(vals["Shoulder L"]), round_deg(vals["Shoulder R"])
                L_elb_i, R_elb_i = round_deg(vals["Elbow L"]), round_deg(vals["Elbow R"])
                draw_skeleton_on_video(video, pts, vis, show_skeleton, show_joints, show_vis)

            # Live buffer
            t_app = float(time.time() - t0)
            t_live.append(t_app)
            for k in ANGLE_KEYS:
                angles_live[k].append(vals[k])
            pose_live.append(None if pts_norm_snapshot is None else (pts_norm_snapshot, vis_snapshot))

            trim_time_buffer(
                t_live,
                pose_live,
                *[angles_live[k] for k in ANGLE_KEYS],
                keep_last_seconds=LIVE_BUFFER_SECONDS,
            )

            # Recording
            if timeline_module.recording and timeline_module.record_start_wall is not None:
                timeline_module.record_elapsed = float(time.time() - timeline_module.record_start_wall)
                if timeline_module.record_elapsed >= MAX_REC_SECONDS:
                    timeline_module.recording = False
                    timeline_module.record_done = True
                    timeline_module.playing = False
                    timeline_module.play_phase = 0.0
                    timeline_module.record_elapsed = MAX_REC_SECONDS

            if timeline_module.recording and timeline_module.record_start_wall is not None:
                t_rel = float(time.time() - timeline_module.record_start_wall)
                if t_rel <= MAX_REC_SECONDS + 1e-6:
                    t_rec.append(t_rel)
                    for k in ANGLE_KEYS:
                        angles_rec[k].append(vals[k])
                    pose_rec.append(None if pts_norm_snapshot is None else (pts_norm_snapshot, vis_snapshot))

            rec_duration_s = float(t_rec[-1]) if t_rec else 0.0

            if rec_duration_s >= SEG_SECONDS:
                timeline_module.pinned_start_t = max(
                    0.0,
                    min(timeline_module.pinned_start_t, rec_duration_s - SEG_SECONDS),
                )
            else:
                timeline_module.pinned_start_t = 0.0

            # Playback phase
            now_t2 = time.time()
            dt = now_t2 - timeline_module.play_last_t
            timeline_module.play_last_t = now_t2

            if (
                (not timeline_module.live_mode)
                and timeline_module.playing
                and (rec_duration_s >= SEG_SECONDS)
                and (not timeline_module.recording)
            ):
                timeline_module.play_phase = (
                    timeline_module.play_phase + dt / max(1e-6, SEG_SECONDS)
                ) % 1.0
            else:
                timeline_module.play_phase = 0.0

            # Segment source
            if timeline_module.live_mode:
                if not t_live:
                    seg_ts, seg_series, seg_poses = None, None, None
                else:
                    t_end = float(t_live[-1])
                    t_start = max(0.0, t_end - SEG_SECONDS)
                    seg_ts, seg_series, seg_poses = extract_segment_by_time(
                        t_live, angles_live, pose_live, t_start, t_end
                    )
                play_idx = (len(seg_ts) - 1) if seg_ts is not None else 0
            else:
                start_t = float(timeline_module.pinned_start_t)
                end_t = start_t + SEG_SECONDS
                seg_ts, seg_series, seg_poses = extract_segment_by_time(
                    t_rec, angles_rec, pose_rec, start_t, end_t
                )
                if seg_ts is None:
                    play_idx = 0
                else:
                    nseg = int(len(seg_ts))
                    play_idx = int(np.clip(round(timeline_module.play_phase * (nseg - 1)), 0, nseg - 1))

            # Build dashboard
            dash = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
            dash[:] = (10, 10, 10)

            panel_w_eff = PANEL_W if show_panel else 0
            pane_w = VIEW_W - panel_w_eff
            pane_h = VIEW_H

            if show_panel:
                panel = panel_bg(VIEW_H, PANEL_W)
                mode_txt = "LIVE" if timeline_module.live_mode else "REVIEW"
                rec_txt = (
                    "REC"
                    if timeline_module.recording
                    else ("DONE" if timeline_module.record_done else "READY")
                )
                y = draw_panel_header(
                    panel,
                    "Pose Dashboard",
                    subtitle=f"{mode_txt} | REC {rec_txt} | cap {int(MAX_REC_SECONDS)}s",
                )

                col1_x = 16
                col2_x = PANEL_W // 2 + 6
                box_w = PANEL_W // 2 - 22
                y += 14
                box_h = 74
                draw_stat_box(panel, col1_x, y, box_w, box_h, "FPS", f"{fps:0.1f}")
                draw_stat_box(panel, col2_x, y, box_w, box_h, "Infer (ms)", f"{infer_ms:0.1f}")
                y += box_h + 14

                y = draw_lr_table(
                    panel,
                    16,
                    y,
                    PANEL_W - 32,
                    46,
                    "Angles",
                    rows=[
                        ("Hip", L_hip_i, R_hip_i),
                        ("Knee", L_knee_i, R_knee_i),
                        ("Ankle", L_ank_i, R_ank_i),
                        ("Shoulder", L_sho_i, R_sho_i),
                        ("Elbow", L_elb_i, R_elb_i),
                    ],
                )
                y += 14

                y = draw_controls_section(panel, 16, y, PANEL_W - 32, controls_expanded)

                dash[:, :PANEL_W] = panel
                cv2.line(dash, (PANEL_W, 0), (PANEL_W, VIEW_H - 1), (45, 45, 45), 1)

            plot_enabled = bool(show_polar)

            if plot_enabled:
                plot_y0 = PLOT_PAD
                plot_h = pane_h - 2 * PLOT_PAD

                min_video_w = 280
                video_area_w = max(min_video_w, pane_w - (PLOT_W + 2 * PLOT_PAD))
                plot_w_eff = max(340, pane_w - video_area_w - 2 * PLOT_PAD)

                video_pane = fit_video_to_pane(
                    video, video_area_w - 2 * VIDEO_PAD, pane_h - 2 * VIDEO_PAD
                )
                x_off = panel_w_eff + VIDEO_PAD
                dash[VIDEO_PAD : VIDEO_PAD + video_pane.shape[0], x_off : x_off + video_pane.shape[1]] = video_pane

                plot_x0 = panel_w_eff + video_area_w + PLOT_PAD

                polar_h = int(plot_h * (1.0 - CLIP_H_FRAC)) - TIMELINE_H - 10
                clip_h = plot_h - polar_h - TIMELINE_H - 10

                polar_canvas = np.zeros((max(180, polar_h), plot_w_eff, 3), dtype=np.uint8)
                clip_canvas = np.zeros((max(180, clip_h), plot_w_eff, 3), dtype=np.uint8)

                draw_polar_plot_segment(
                    polar_canvas,
                    seg_series_dict={} if seg_series is None else seg_series,
                    play_idx=play_idx,
                    title="Angles (Polar)",
                )
                dash[plot_y0 : plot_y0 + polar_canvas.shape[0], plot_x0 : plot_x0 + plot_w_eff] = polar_canvas

                t_y = plot_y0 + polar_canvas.shape[0] + 6
                draw_timeline_ui(
                    dash,
                    x0=plot_x0,
                    y0=t_y,
                    w=plot_w_eff,
                    h=TIMELINE_H,
                    is_live_mode=timeline_module.live_mode,
                    duration_s=rec_duration_s if not timeline_module.recording else min(rec_duration_s, MAX_REC_SECONDS),
                    pinned_start=timeline_module.pinned_start_t,
                    play_ph=timeline_module.play_phase,
                    is_recording=timeline_module.recording,
                    is_record_done=timeline_module.record_done,
                    is_playing=timeline_module.playing,
                )

                if seg_poses is None or play_idx >= (len(seg_poses) if seg_poses is not None else 0):
                    ptsn, visn = None, None
                else:
                    item = seg_poses[play_idx]
                    ptsn, visn = (None, None) if item is None else item

                clip_title = (
                    "Clip (live)"
                    if timeline_module.live_mode
                    else ("Clip (loop)" if timeline_module.playing else "Clip (window)")
                )
                draw_pose_clip(clip_canvas, pts_norm=ptsn, vis_arr=visn, title=clip_title)

                y_clip = t_y + TIMELINE_H + 6
                y_clip_end = min(plot_y0 + plot_h, y_clip + clip_canvas.shape[0])
                dash[y_clip:y_clip_end, plot_x0 : plot_x0 + plot_w_eff] = clip_canvas[: (y_clip_end - y_clip)]

                cv2.line(
                    dash,
                    (panel_w_eff + video_area_w, 0),
                    (panel_w_eff + video_area_w, VIEW_H - 1),
                    (35, 35, 35),
                    1,
                )
            else:
                clear_hitboxes()

                video_pane = fit_video_to_pane(video, pane_w - 2 * VIDEO_PAD, pane_h - 2 * VIDEO_PAD)
                x_off = panel_w_eff + VIDEO_PAD
                dash[VIDEO_PAD : VIDEO_PAD + video_pane.shape[0], x_off : x_off + video_pane.shape[1]] = video_pane

            if show_console and (time.time() - last_log) >= LOG_INTERVAL:
                line = console_line(
                    time.time() - t0,
                    fps,
                    infer_ms,
                    L_hip_i,
                    R_hip_i,
                    L_knee_i,
                    R_knee_i,
                    L_ank_i,
                    R_ank_i,
                    L_sho_i,
                    R_sho_i,
                    L_elb_i,
                    R_elb_i,
                )
                sys.stdout.write("\r" + line + " " * 10)
                sys.stdout.flush()
                last_log = time.time()

            cv2.imshow(WINDOW, dash)

            key = cv2.waitKey(1) & 0xFF
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


if __name__ == "__main__":
    main()
