"""Pose dashboard: MediaPipe pose estimation with angle tracking and recording."""

import ctypes
import sys
import time

#
import cv2
import numpy as np
from config import (
    ACCENT,
    APP_PAD,
    CAM_H,
    CAM_INDEX,
    CAM_W,
    CARD_GAP,
    CLIP_H_FRAC,
    DEFAULT_POLAR_PALETTE_KEY,
    LIVE_BUFFER_SECONDS,
    LOG_INTERVAL,
    MAX_REC_SECONDS,
    MIRROR_VIEW,
    PALETTE_GALLERY_BG,
    PALETTE_GALLERY_BORDER,
    PALETTE_GALLERY_CARD_GAP,
    PALETTE_GALLERY_CARD_H,
    PALETTE_GALLERY_CARD_W,
    PALETTE_GALLERY_COLS,
    PALETTE_GALLERY_PREVIEW_BG,
    PALETTE_GALLERY_SCRIM,
    PALETTE_GALLERY_SELECTED,
    PALETTE_GALLERY_SUBTITLE,
    PALETTE_GALLERY_TEXT,
    PALETTE_GALLERY_TEXT_MUTED,
    PALETTE_GALLERY_TITLE,
    PANEL_DIVIDER,
    PANEL_W,
    PLOT_PAD,
    PLOT_W,
    POLAR_PALETTES,
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
    WINDOW_BG,
    ensure_task_file,
    get_polar_palette,
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
from visualization.polar_plot import draw_palette_preview, draw_polar_plot_segment
from visualization.skeleton import draw_skeleton_on_video

# MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as e:
    raise RuntimeError(
        "MediaPipe Tasks API not available. Use Python 3.11 and:\n"
        "  pip install mediapipe\n"
    ) from e

# Timeline state (live_mode, recording, etc.)
from ui import timeline as timeline_module


def _enable_high_dpi():
    """Ask Windows to avoid bitmap-scaling the OpenCV window."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _point_in_rect(x, y, rect):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _draw_palette_gallery(img, selected_palette_key):
    base = img.copy()
    scrim = np.zeros_like(img)
    scrim[:] = PALETTE_GALLERY_SCRIM
    cv2.addWeighted(scrim, 0.62, base, 0.38, 0.0, dst=img)

    n_items = len(POLAR_PALETTES)
    cols = min(PALETTE_GALLERY_COLS, max(1, n_items))
    rows = int(np.ceil(n_items / float(cols)))
    inner_pad = 18
    title_h = 66
    foot_h = 20
    modal_w = (
        cols * PALETTE_GALLERY_CARD_W
        + (cols - 1) * PALETTE_GALLERY_CARD_GAP
        + 2 * inner_pad
    )
    modal_h = (
        rows * PALETTE_GALLERY_CARD_H
        + (rows - 1) * PALETTE_GALLERY_CARD_GAP
        + title_h
        + foot_h
        + 2 * inner_pad
    )
    modal_x0 = max(20, (img.shape[1] - modal_w) // 2)
    modal_y0 = max(20, (img.shape[0] - modal_h) // 2)
    modal_x1 = modal_x0 + modal_w
    modal_y1 = modal_y0 + modal_h

    cv2.rectangle(
        img, (modal_x0, modal_y0), (modal_x1, modal_y1), PALETTE_GALLERY_BG, -1
    )
    cv2.rectangle(
        img, (modal_x0, modal_y0), (modal_x1, modal_y1), PALETTE_GALLERY_BORDER, 2
    )
    cv2.putText(
        img,
        PALETTE_GALLERY_TITLE,
        (modal_x0 + inner_pad, modal_y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        PALETTE_GALLERY_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        PALETTE_GALLERY_SUBTITLE,
        (modal_x0 + inner_pad, modal_y0 + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        PALETTE_GALLERY_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "ESC or click outside to close",
        (modal_x1 - 220, modal_y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        PALETTE_GALLERY_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    hitboxes = []
    cards_y0 = modal_y0 + title_h + inner_pad - 2
    for idx, palette in enumerate(POLAR_PALETTES):
        row = idx // cols
        col = idx % cols
        x0 = (
            modal_x0
            + inner_pad
            + col * (PALETTE_GALLERY_CARD_W + PALETTE_GALLERY_CARD_GAP)
        )
        y0 = cards_y0 + row * (PALETTE_GALLERY_CARD_H + PALETTE_GALLERY_CARD_GAP)
        x1 = x0 + PALETTE_GALLERY_CARD_W
        y1 = y0 + PALETTE_GALLERY_CARD_H
        is_selected = palette["key"] == selected_palette_key
        border_col = PALETTE_GALLERY_SELECTED if is_selected else PALETTE_GALLERY_BORDER
        fill_col = (
            PALETTE_GALLERY_PREVIEW_BG
            if not is_selected
            else tuple(int(min(255, c + 10)) for c in PALETTE_GALLERY_PREVIEW_BG)
        )
        cv2.rectangle(img, (x0, y0), (x1, y1), fill_col, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), border_col, 2 if is_selected else 1)

        preview = img[y0 + 8 : y0 + 72, x0 + 8 : x1 - 8]
        draw_palette_preview(preview, palette)

        cv2.putText(
            img,
            palette["label"],
            (x0 + 10, y0 + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            PALETTE_GALLERY_TEXT,
            1,
            cv2.LINE_AA,
        )
        descriptor = "selected" if is_selected else "click to apply"
        cv2.putText(
            img,
            descriptor,
            (x0 + 10, y0 + 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            PALETTE_GALLERY_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
        hitboxes.append((x0, y0, x1, y1, palette["key"]))

    return (modal_x0, modal_y0, modal_x1, modal_y1), hitboxes


def main():
    ensure_task_file(TASK_PATH, TASK_URL)
    _enable_high_dpi()

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
    timeline_mouse_cb = make_mouse_cb()

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
    selected_palette_key = DEFAULT_POLAR_PALETTE_KEY
    palette_gallery_open = False
    palette_modal_rect = None
    palette_hitboxes = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal \
            palette_gallery_open, \
            selected_palette_key, \
            palette_modal_rect, \
            palette_hitboxes

        if palette_gallery_open:
            if event == cv2.EVENT_LBUTTONDOWN:
                for x0, y0, x1, y1, palette_key in palette_hitboxes:
                    if _point_in_rect(x, y, (x0, y0, x1, y1)):
                        selected_palette_key = palette_key
                        palette_gallery_open = False
                        return
                if palette_modal_rect is None or (
                    not _point_in_rect(x, y, palette_modal_rect)
                ):
                    palette_gallery_open = False
                return
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
                return

        timeline_mouse_cb(event, x, y, flags, param)

    cv2.setMouseCallback(WINDOW, mouse_cb)

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

            lm = (
                result.pose_landmarks[0]
                if (result.pose_landmarks and len(result.pose_landmarks) > 0)
                else None
            )
            h, w = frame.shape[:2]
            out = process_pose(lm, h, w)

            if out[0] is not None:
                pts, vis, _, _, vals, pts_norm_snapshot, vis_snapshot = out
                L_hip_i, R_hip_i = round_deg(vals["Hip L"]), round_deg(vals["Hip R"])
                L_knee_i, R_knee_i = (
                    round_deg(vals["Knee L"]),
                    round_deg(vals["Knee R"]),
                )
                L_ank_i, R_ank_i = round_deg(vals["Ank L"]), round_deg(vals["Ank R"])
                L_sho_i, R_sho_i = (
                    round_deg(vals["Shoulder L"]),
                    round_deg(vals["Shoulder R"]),
                )
                L_elb_i, R_elb_i = (
                    round_deg(vals["Elbow L"]),
                    round_deg(vals["Elbow R"]),
                )
                draw_skeleton_on_video(
                    video, pts, vis, show_skeleton, show_joints, show_vis
                )

            # Live buffer
            t_app = float(time.time() - t0)
            t_live.append(t_app)
            for k in ANGLE_KEYS:
                angles_live[k].append(vals[k])
            pose_live.append(
                None if pts_norm_snapshot is None else (pts_norm_snapshot, vis_snapshot)
            )

            trim_time_buffer(
                t_live,
                pose_live,
                *[angles_live[k] for k in ANGLE_KEYS],
                keep_last_seconds=LIVE_BUFFER_SECONDS,
            )

            # Recording
            if (
                timeline_module.recording
                and timeline_module.record_start_wall is not None
            ):
                timeline_module.record_elapsed = float(
                    time.time() - timeline_module.record_start_wall
                )
                if timeline_module.record_elapsed >= MAX_REC_SECONDS:
                    timeline_module.recording = False
                    timeline_module.record_done = True
                    timeline_module.playing = False
                    timeline_module.play_phase = 0.0
                    timeline_module.record_elapsed = MAX_REC_SECONDS

            if (
                timeline_module.recording
                and timeline_module.record_start_wall is not None
            ):
                t_rel = float(time.time() - timeline_module.record_start_wall)
                if t_rel <= MAX_REC_SECONDS + 1e-6:
                    t_rec.append(t_rel)
                    for k in ANGLE_KEYS:
                        angles_rec[k].append(vals[k])
                    pose_rec.append(
                        None
                        if pts_norm_snapshot is None
                        else (pts_norm_snapshot, vis_snapshot)
                    )

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
                    play_idx = int(
                        np.clip(
                            round(timeline_module.play_phase * (nseg - 1)), 0, nseg - 1
                        )
                    )

            # Build dashboard
            dash = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
            dash[:] = WINDOW_BG

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
                palette_label = get_polar_palette(selected_palette_key)["label"]

                col1_x = APP_PAD
                col2_x = PANEL_W // 2 + 8
                box_w = PANEL_W // 2 - 24
                y += CARD_GAP
                box_h = 78
                draw_stat_box(panel, col1_x, y, box_w, box_h, "FPS", f"{fps:0.1f}")
                draw_stat_box(
                    panel, col2_x, y, box_w, box_h, "Infer (ms)", f"{infer_ms:0.1f}"
                )
                y += box_h + CARD_GAP

                y = draw_lr_table(
                    panel,
                    APP_PAD,
                    y,
                    PANEL_W - 2 * APP_PAD,
                    48,
                    "Angles",
                    rows=[
                        ("Hip", L_hip_i, R_hip_i),
                        ("Knee", L_knee_i, R_knee_i),
                        ("Ankle", L_ank_i, R_ank_i),
                        ("Shoulder", L_sho_i, R_sho_i),
                        ("Elbow", L_elb_i, R_elb_i),
                    ],
                )
                y += CARD_GAP

                cv2.putText(
                    panel,
                    f"Polar style: {palette_label}",
                    (APP_PAD, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    ACCENT,
                    1,
                    cv2.LINE_AA,
                )
                y += 8

                y = draw_controls_section(
                    panel, APP_PAD, y, PANEL_W - 2 * APP_PAD, controls_expanded
                )

                dash[:, :PANEL_W] = panel
                cv2.line(dash, (PANEL_W, 0), (PANEL_W, VIEW_H - 1), PANEL_DIVIDER, 1)

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
                dash[
                    VIDEO_PAD : VIDEO_PAD + video_pane.shape[0],
                    x_off : x_off + video_pane.shape[1],
                ] = video_pane

                plot_x0 = panel_w_eff + video_area_w + PLOT_PAD

                polar_h = int(plot_h * (1.0 - CLIP_H_FRAC)) - TIMELINE_H - 10
                clip_h = plot_h - polar_h - TIMELINE_H - 10

                polar_canvas = np.zeros(
                    (max(180, polar_h), plot_w_eff, 3), dtype=np.uint8
                )
                clip_canvas = np.zeros(
                    (max(180, clip_h), plot_w_eff, 3), dtype=np.uint8
                )

                draw_polar_plot_segment(
                    polar_canvas,
                    seg_series_dict={} if seg_series is None else seg_series,
                    play_idx=play_idx,
                    title="Angles (Polar)",
                    palette_key=selected_palette_key,
                )
                dash[
                    plot_y0 : plot_y0 + polar_canvas.shape[0],
                    plot_x0 : plot_x0 + plot_w_eff,
                ] = polar_canvas

                t_y = plot_y0 + polar_canvas.shape[0] + 6
                draw_timeline_ui(
                    dash,
                    x0=plot_x0,
                    y0=t_y,
                    w=plot_w_eff,
                    h=TIMELINE_H,
                    is_live_mode=timeline_module.live_mode,
                    duration_s=rec_duration_s
                    if not timeline_module.recording
                    else min(rec_duration_s, MAX_REC_SECONDS),
                    pinned_start=timeline_module.pinned_start_t,
                    play_ph=timeline_module.play_phase,
                    is_recording=timeline_module.recording,
                    is_record_done=timeline_module.record_done,
                    is_playing=timeline_module.playing,
                )

                if seg_poses is None or play_idx >= (
                    len(seg_poses) if seg_poses is not None else 0
                ):
                    ptsn, visn = None, None
                else:
                    item = seg_poses[play_idx]
                    ptsn, visn = (None, None) if item is None else item

                clip_title = (
                    "Clip (live)"
                    if timeline_module.live_mode
                    else ("Clip (loop)" if timeline_module.playing else "Clip (window)")
                )
                draw_pose_clip(
                    clip_canvas, pts_norm=ptsn, vis_arr=visn, title=clip_title
                )

                y_clip = t_y + TIMELINE_H + 6
                y_clip_end = min(plot_y0 + plot_h, y_clip + clip_canvas.shape[0])
                dash[y_clip:y_clip_end, plot_x0 : plot_x0 + plot_w_eff] = clip_canvas[
                    : (y_clip_end - y_clip)
                ]

                cv2.line(
                    dash,
                    (panel_w_eff + video_area_w, 0),
                    (panel_w_eff + video_area_w, VIEW_H - 1),
                    PANEL_DIVIDER,
                    1,
                )
            else:
                clear_hitboxes()
                palette_gallery_open = False
                palette_modal_rect = None
                palette_hitboxes = []

                video_pane = fit_video_to_pane(
                    video, pane_w - 2 * VIDEO_PAD, pane_h - 2 * VIDEO_PAD
                )
                x_off = panel_w_eff + VIDEO_PAD
                dash[
                    VIDEO_PAD : VIDEO_PAD + video_pane.shape[0],
                    x_off : x_off + video_pane.shape[1],
                ] = video_pane

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

            if palette_gallery_open and plot_enabled:
                palette_modal_rect, palette_hitboxes = _draw_palette_gallery(
                    dash, selected_palette_key
                )
            else:
                palette_modal_rect = None
                palette_hitboxes = []

            cv2.imshow(WINDOW, dash)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 27:
                palette_gallery_open = False
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
                if not show_polar:
                    palette_gallery_open = False
            elif key in (ord("g"), ord("G")) and show_polar:
                palette_gallery_open = not palette_gallery_open

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if SHOW_CONSOLE:
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
