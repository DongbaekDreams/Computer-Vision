"""Pose dashboard: MediaPipe pose estimation with angle tracking and recording."""

import ctypes
import sys
import time

#
import cv2
import numpy as np
from body_profile import BodyProfile, load_body_profile, resolve_body_profile, save_body_profile
from config import (
    ACCENT,
    APP_PAD,
    CAM_H,
    CAM_W,
    VIDEO_BG,
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
from camera_config import (
    LastCameraSetup,
    MultiCameraReader,
    apply_rotation,
    detect_connected_cameras,
    get_camera_baseline_world,
    is_local_camera_id,
    load_calibrations,
    load_last_setup,
    open_camera,
    save_last_setup,
)

# (Legacy) Wall-clock sync between cameras was too strict with serialized USB reads;
# triangulation now only requires a fresh frame from each active camera each tick.
from pose_processor import ANGLE_KEYS, process_pose, round_deg
from state import angles_live, angles_rec, pose_live, pose_rec, t_live, t_rec
from triangulation import process_multi_cam_poses
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
from visualization.pose_3d_view import draw_3d_pose_canvas
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

# Default camera index when no calibration/setup exists
DEFAULT_CAM_INDEX = 0
WINDOW_CAM_SELECT = "Camera selection - number keys toggle, P=primary, Enter=confirm"


def _short_label(cam_id: str) -> str:
    return f"Cam {cam_id}"


def _merged_display_frames(
    fresh: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    cam_ids: list[str],
) -> dict[str, np.ndarray | None]:
    """Prefer this-tick frames; fall back to last good copy so UI tiles do not flicker black."""
    m: dict[str, np.ndarray | None] = {}
    for cid in cam_ids:
        if cid in fresh:
            m[cid] = fresh[cid]
        elif cid in cache:
            m[cid] = cache[cid]
        else:
            m[cid] = None
    return m


def _annotate_cam_video(
    cid: str,
    frame: np.ndarray,
    per_cam_results: list,
    *,
    show_camera_bg: bool,
    show_skeleton: bool,
    show_joints: bool,
    show_vis: bool,
) -> np.ndarray:
    out = frame.copy() if show_camera_bg else np.zeros_like(frame)
    if not show_camera_bg:
        out[:] = (8, 8, 8)
    for r in per_cam_results:
        if r[0] != cid:
            continue
        _, pts, vis, _, _, _, _, _ = r
        if pts is not None:
            draw_skeleton_on_video(out, pts, vis, show_skeleton, show_joints, show_vis)
        break
    return out


def _compose_dual_cam_and_3d(
    cam_order: list[str],
    frames_by_id: dict,
    per_cam_results: list,
    pts_3d_live: np.ndarray | None,
    vis_3d_live: np.ndarray | None,
    pane_inner_w: int,
    pane_inner_h: int,
    *,
    show_camera_bg: bool,
    show_skeleton: bool,
    show_joints: bool,
    show_vis: bool,
    use_triangulation: bool,
) -> np.ndarray:
    gap = 10
    c0, c1 = cam_order[0], cam_order[1]
    cam_band_h = max(200, int((pane_inner_h - gap) * 0.55))
    view3d_h = max(140, pane_inner_h - cam_band_h - gap)
    half_w = max(120, (pane_inner_w - gap) // 2)

    f0 = frames_by_id.get(c0)
    f1 = frames_by_id.get(c1)
    if f0 is None:
        f0 = np.zeros((720, 1280, 3), dtype=np.uint8)
    if f1 is None:
        f1 = np.zeros((720, 1280, 3), dtype=np.uint8)

    v0 = _annotate_cam_video(
        c0,
        f0,
        per_cam_results,
        show_camera_bg=show_camera_bg,
        show_skeleton=show_skeleton,
        show_joints=show_joints,
        show_vis=show_vis,
    )
    v1 = _annotate_cam_video(
        c1,
        f1,
        per_cam_results,
        show_camera_bg=show_camera_bg,
        show_skeleton=show_skeleton,
        show_joints=show_joints,
        show_vis=show_vis,
    )

    p0 = fit_video_to_pane(v0, half_w, cam_band_h)
    p1 = fit_video_to_pane(v1, half_w, cam_band_h)
    cv2.putText(
        p0,
        f"Cam {c0}  (primary)",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 220, 170),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        p1,
        f"Cam {c1}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 245),
        2,
        cv2.LINE_AA,
    )

    sep = np.full((cam_band_h, gap, 3), 24, dtype=np.uint8)
    top = np.hstack((p0, sep, p1))

    canvas3d = np.zeros((view3d_h, pane_inner_w, 3), dtype=np.uint8)
    if use_triangulation:
        draw_3d_pose_canvas(
            canvas3d,
            pts_3d_live,
            vis_3d_live,
            title="3D skeleton (triangulated)",
        )
    else:
        canvas3d[:] = VIDEO_BG
        cv2.putText(
            canvas3d,
            "3D: run calibrate_extrinsics_3d.py (chessboard, 2+ cameras)",
            (12, view3d_h // 2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (150, 155, 185),
            1,
            cv2.LINE_AA,
        )

    vsep = np.full((gap, pane_inner_w, 3), 24, dtype=np.uint8)
    return np.vstack((top, vsep, canvas3d))


def _run_camera_confirmation_ui(calibrations, last_setup):
    """
    Show which calibrated local cameras are connected; user toggles selection and confirms.
    Returns (list of selected camera_ids, primary_camera_id) or (None, None).
    """
    connected = detect_connected_cameras()
    available_ids: list[str] = []
    for c in connected:
        cid = str(c.index)
        if cid in calibrations:
            available_ids.append(cid)
    if not available_ids:
        return None, None

    selected = set(last_setup.selected_camera_ids) if last_setup else set()
    selected = {cid for cid in selected if cid in available_ids}
    if not selected:
        selected = {available_ids[0]}
    primary = last_setup.primary_camera_id if last_setup else available_ids[0]
    if primary not in available_ids:
        primary = available_ids[0]

    caps = []
    for cid in available_ids:
        cap = open_camera(cid, 640, 480)
        if cap.isOpened():
            caps.append((cid, cap))

    print("Camera confirmation: 0-9=toggle | P=cycle primary | Enter=confirm")
    while True:
        for cid, cap in caps:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            disp = frame.copy()
            sel = "SELECTED" if cid in selected else "off"
            prim = " [PRIMARY]" if cid == primary else ""
            label = _short_label(cid)
            cv2.putText(disp, f"{label} {sel}{prim}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(disp, "0-9=toggle | P=primary | Enter=done",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow(f"Cam: {label}", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:
            break
        if key in (ord("p"), ord("P")) and selected:
            sel_list = sorted(selected)
            try:
                idx = sel_list.index(primary)
                primary = sel_list[(idx + 1) % len(sel_list)]
            except ValueError:
                primary = sel_list[0]
            print(f"  Primary: {primary}")
        if ord("0") <= key <= ord("9"):
            k = str(key - ord("0"))
            if k in available_ids:
                if k in selected:
                    selected.discard(k)
                else:
                    selected.add(k)
                if primary not in selected and selected:
                    primary = sorted(selected)[0]
                print(f"  Selected: {sorted(selected)}, primary: {primary}")
        if key == ord("q"):
            for _, cap in caps:
                cap.release()
            for cid, _ in caps:
                cv2.destroyWindow(f"Cam: {_short_label(cid)}")
            return None, None
    for _, cap in caps:
        cap.release()
    for cid, _ in caps:
        cv2.destroyWindow(f"Cam: {_short_label(cid)}")
    return sorted(selected), primary


def _enable_high_dpi():
    """Ask Windows to avoid bitmap-scaling the OpenCV window."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _ask_optional_float_dialog(
    title: str,
    prompt: str,
    initial: float | None,
) -> tuple[bool, float | None]:
    """Prompt for a positive float, allow blank to clear, cancel to abort."""
    import tkinter as tk
    from tkinter import messagebox, simpledialog

    initial_text = "" if initial is None else f"{float(initial):0.3f}"
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            raw = simpledialog.askstring(title, prompt, initialvalue=initial_text, parent=root)
            if raw is None:
                return True, initial
            raw = raw.strip()
            if not raw:
                return False, None
            value = float(raw)
            if value <= 0.0:
                raise ValueError
            return False, value
        except ValueError:
            messagebox.showerror(
                title,
                "Enter a positive number, leave blank to clear, or Cancel to abort.",
                parent=root,
            )
        finally:
            root.destroy()


def _edit_body_profile_dialog(profile: BodyProfile) -> BodyProfile | None:
    """Edit saved body measurements with lightweight dialogs."""
    updated = BodyProfile(height_m=profile.height_m, segments_m=dict(profile.segments_m))
    prompts = [
        ("height_m", "Height (m)"),
        ("shoulder_width", "Shoulder width (m)"),
        ("hip_width", "Hip width (m)"),
        ("torso", "Torso length (m)"),
        ("upper_arm", "Upper arm length (m)"),
        ("forearm", "Forearm length (m)"),
        ("thigh", "Thigh length (m)"),
        ("shank", "Shank length (m)"),
    ]
    for key, label in prompts:
        current = updated.height_m if key == "height_m" else updated.segments_m.get(key)
        cancelled, value = _ask_optional_float_dialog(
            "Body profile",
            f"{label}\n\nLeave blank to clear this field.",
            current,
        )
        if cancelled:
            return None
        if key == "height_m":
            updated.height_m = value
        elif value is None:
            updated.segments_m.pop(key, None)
        else:
            updated.segments_m[key] = value
    return updated


def _resolve_runtime_camera_setup(
    calibrations,
    last_setup,
    connected,
) -> tuple[list[str] | None, str | None, bool, dict[str, int], float | None]:
    """
    Restore the saved multi-camera rig. Uses last_camera_setup order; does not
    require each index to appear in detect_connected_cameras() (probe can miss
    high indices or a slow second USB cam).
    """
    connected_ids = [str(c.index) for c in connected]
    selected_ids: list[str] = []

    if last_setup and last_setup.selected_camera_ids:
        for cid in last_setup.selected_camera_ids:
            s = str(cid)
            if not is_local_camera_id(s) or s not in calibrations:
                continue
            if s not in selected_ids:
                selected_ids.append(s)

    if not selected_ids:
        for cid in connected_ids:
            if cid in calibrations and cid not in selected_ids:
                selected_ids.append(cid)

    if not selected_ids:
        return None, None, False, {}, None

    # If last_camera_setup was overwritten with a single camera but exactly two calibrated
    # cameras have extrinsics (typical 0+2 rig), restore the pair from calibrations.
    ext_cams = sorted(
        [
            str(k)
            for k, c in calibrations.items()
            if is_local_camera_id(str(k)) and c.extrinsics is not None
        ],
        key=int,
    )
    if len(selected_ids) == 1 and len(ext_cams) == 2 and selected_ids[0] in ext_cams:
        selected_ids = ext_cams
        print(
            f"Recovered dual-camera rig from calibrations (extrinsics on both): {selected_ids}"
        )

    # Exactly two entries in camera_calibrations.json -> use both (fixes last_setup stuck on ["0"]
    # when cam 2 has intrinsics but extrinsics were never saved on it).
    cal_keys = sorted(
        [str(k) for k in calibrations if is_local_camera_id(str(k))],
        key=int,
    )
    if len(selected_ids) == 1 and len(cal_keys) == 2:
        selected_ids = cal_keys
        print(
            f"Using both calibrated camera indices {selected_ids} "
            f"(camera_calibrations.json has exactly two). "
            "Update last_camera_setup.json if you need a different rig."
        )

    primary = selected_ids[0]
    if last_setup and last_setup.primary_camera_id in selected_ids:
        primary = last_setup.primary_camera_id

    rotations = {}
    if last_setup and last_setup.camera_rotations:
        rotations = {
            cid: int(last_setup.camera_rotations.get(cid, 0))
            for cid in selected_ids
            if cid in last_setup.camera_rotations
        }

    # 3D whenever two+ calibrated views include extrinsics (ignore saved use_triangulation=False).
    use_triangulation = len(selected_ids) >= 2 and all(
        calibrations.get(cid) and getattr(calibrations.get(cid), "extrinsics", None)
        for cid in selected_ids
    )
    baseline_m = last_setup.camera_baseline_m if last_setup else None
    return selected_ids, primary, use_triangulation, rotations, baseline_m


def _compute_metric_scale(
    active_camera_ids: list[str],
    primary_camera_id: str,
    calibrations,
    baseline_m: float | None,
) -> tuple[float, float | None, tuple[str, str] | None]:
    """Return meters-per-world-unit scale and optional solved baseline."""
    base_scale_m = 0.001  # Extrinsics are solved in chessboard millimeters.
    ordered_ids = [primary_camera_id] + [cid for cid in active_camera_ids if cid != primary_camera_id]
    if len(ordered_ids) < 2:
        return base_scale_m, None, None

    pair = (ordered_ids[0], ordered_ids[1])
    cal_a = calibrations.get(pair[0])
    cal_b = calibrations.get(pair[1])
    if cal_a is None or cal_b is None:
        return base_scale_m, None, pair

    baseline_world = get_camera_baseline_world(cal_a, cal_b)
    if baseline_world is None or baseline_world <= 0.0:
        return base_scale_m, None, pair

    solved_baseline_m = baseline_world * base_scale_m
    if baseline_m is None or baseline_m <= 0.0:
        return base_scale_m, solved_baseline_m, pair
    return float(baseline_m / baseline_world), solved_baseline_m, pair


def _save_runtime_setup(
    active_camera_ids: list[str],
    primary_camera_id: str,
    use_triangulation: bool,
    camera_rotations: dict[str, int],
    baseline_m: float | None,
) -> None:
    save_last_setup(
        LastCameraSetup(
            selected_camera_ids=active_camera_ids,
            primary_camera_id=primary_camera_id,
            use_triangulation=use_triangulation,
            camera_rotations=camera_rotations,
            camera_baseline_m=baseline_m,
        )
    )


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

    calibrations = load_calibrations()
    last_setup = load_last_setup()
    body_profile = load_body_profile()
    active_camera_ids = None
    primary_camera_id = None
    use_triangulation = False
    camera_rotations: dict[str, int] = {}
    baseline_m = last_setup.camera_baseline_m if last_setup else None

    connected = detect_connected_cameras()
    runtime_setup = _resolve_runtime_camera_setup(calibrations, last_setup, connected)
    if runtime_setup[0]:
        active_camera_ids, primary_camera_id, use_triangulation, camera_rotations, baseline_m = (
            runtime_setup
        )
    # Do not save last_camera_setup on every launch: that used to overwrite a multi-cam
    # rig with a single detected camera and drop indices like 2 from the JSON.
    if active_camera_ids is None:
        active_camera_ids = [str(DEFAULT_CAM_INDEX)]
        primary_camera_id = str(DEFAULT_CAM_INDEX)
        use_triangulation = False

    if (
        active_camera_ids
        and len(active_camera_ids) == 1
        and last_setup
        and len(last_setup.selected_camera_ids) >= 2
    ):
        print(
            "Note: last_camera_setup lists multiple cameras but only one is active. "
            "Check that each index is in camera_calibrations.json and has extrinsics for 3D "
            "(run calibrate_extrinsics_3d.py after intrinsics)."
        )

    metric_scale, solved_baseline_m, baseline_pair = _compute_metric_scale(
        active_camera_ids,
        primary_camera_id,
        calibrations,
        baseline_m,
    )

    usb_read_order = [primary_camera_id] + [
        c for c in active_camera_ids if c != primary_camera_id
    ]
    multi_cap = MultiCameraReader.from_camera_ids(
        list(active_camera_ids),
        CAM_W,
        CAM_H,
        read_order=usb_read_order,
    )
    print(
        f"Starting dashboard: cameras {active_camera_ids} (primary {primary_camera_id}), "
        f"USB read order {usb_read_order}, triangulation={'on' if use_triangulation else 'off'}"
    )

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, VIEW_W, VIEW_H)
    timeline_mouse_cb = make_mouse_cb()

    # State
    t0 = time.time()
    last_log = 0.0
    frame_i = 0
    fps = 0.0
    # MediaPipe VIDEO mode requires strictly increasing timestamps on every detect_for_video call.
    # Multiple cameras in one loop iteration must not reuse the same value.
    mp_pose_ts = 0

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
    show_dual_cam_strip = len(active_camera_ids) >= 2
    display_frame_cache: dict[str, np.ndarray] = {}
    last_infer_out: dict[str, tuple] = {}

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
            # One USB-locked batch read on this thread (see MultiCameraReader).
            frames_by_id = {}
            raw_batch = multi_cap.read_batch()
            for cid in active_camera_ids:
                if cid not in raw_batch:
                    continue
                frame = raw_batch[cid]
                rot = camera_rotations.get(cid, 0)
                if rot:
                    frame = apply_rotation(frame, rot)
                if MIRROR_VIEW:
                    frame = cv2.flip(frame, 1)
                frames_by_id[cid] = frame
                display_frame_cache[cid] = frame
            if primary_camera_id not in frames_by_id:
                # Primary camera hasn't produced a frame yet; wait briefly
                time.sleep(0.005)
                continue

            # Triangulate whenever each active camera produced a frame this tick (USB-safe).
            frames_ready_for_3d = (
                len(frames_by_id) == len(active_camera_ids) and len(active_camera_ids) >= 2
            )

            merged_display = _merged_display_frames(
                frames_by_id, display_frame_cache, active_camera_ids
            )

            t_inf0 = time.time()
            per_cam_results = []
            for cid in active_camera_ids:
                if cid not in frames_by_id:
                    if cid in last_infer_out and not use_triangulation:
                        per_cam_results.append((cid, *last_infer_out[cid]))
                    continue
                frame = frames_by_id[cid]
                throttle_sec = (
                    not use_triangulation
                    and len(active_camera_ids) >= 2
                    and cid != primary_camera_id
                    and (frame_i % 3) != 0
                    and cid in last_infer_out
                )
                if throttle_sec:
                    per_cam_results.append((cid, *last_infer_out[cid]))
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_pose_ts += 1
                result = landmarker.detect_for_video(mp_img, mp_pose_ts)
                lm = (
                    result.pose_landmarks[0]
                    if (result.pose_landmarks and len(result.pose_landmarks) > 0)
                    else None
                )
                h, w = frame.shape[:2]
                out = process_pose(lm, h, w)
                last_infer_out[cid] = out
                per_cam_results.append((cid, *out))
            infer_ms = (time.time() - t_inf0) * 1000.0

            frame_i += 1
            if frame_i % 10 == 0:
                fps = frame_i / max(1e-6, (time.time() - t0))

            primary_frame = frames_by_id[primary_camera_id]
            if show_camera_bg:
                video = primary_frame.copy()
            else:
                video = np.zeros_like(primary_frame)
                video[:] = (8, 8, 8)

            L_hip_i = L_knee_i = L_ank_i = None
            R_hip_i = R_knee_i = R_ank_i = None
            L_sho_i = R_sho_i = L_elb_i = R_elb_i = None

            vals = {k: np.nan for k in ANGLE_KEYS}
            pts_norm_snapshot = None
            vis_snapshot = None
            pts_3d_live = None
            vis_3d_live = None

            if use_triangulation and len(per_cam_results) >= 2 and frames_ready_for_3d:
                vals, pts_3d_live, vis_3d_live = process_multi_cam_poses(
                    per_cam_results,
                    calibrations,
                    metric_scale=metric_scale,
                )
                for r in per_cam_results:
                    if r[0] == primary_camera_id:
                        _, pts, vis, _, _, _, pts_norm_snapshot, vis_snapshot = r
                        if pts is not None:
                            draw_skeleton_on_video(
                                video, pts, vis, show_skeleton, show_joints, show_vis
                            )
                        break
            else:
                for r in per_cam_results:
                    if r[0] == primary_camera_id:
                        _, pts, vis, _, _, vals, pts_norm_snapshot, vis_snapshot = r
                        if pts is not None:
                            draw_skeleton_on_video(
                                video, pts, vis, show_skeleton, show_joints, show_vis
                            )
                        break
                else:
                    if per_cam_results:
                        _, _, _, _, _, vals, pts_norm_snapshot, vis_snapshot = per_cam_results[0]

            resolved_body = resolve_body_profile(body_profile, pts_3d_live)

            L_hip_i, R_hip_i = round_deg(vals["Hip L"]), round_deg(vals["Hip R"])
            L_knee_i, R_knee_i = round_deg(vals["Knee L"]), round_deg(vals["Knee R"])
            L_ank_i, R_ank_i = round_deg(vals["Ank L"]), round_deg(vals["Ank R"])
            L_sho_i, R_sho_i = round_deg(vals["Shoulder L"]), round_deg(vals["Shoulder R"])
            L_elb_i, R_elb_i = round_deg(vals["Elbow L"]), round_deg(vals["Elbow R"])

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
                angle_mode = "3D" if use_triangulation else "2D"
                if len(active_camera_ids) >= 2 and not use_triangulation:
                    angle_mode = "2D (need extrinsics)"
                y = draw_panel_header(
                    panel,
                    "Pose Dashboard",
                    subtitle=f"{mode_txt} | REC {rec_txt} | {angle_mode} | cap {int(MAX_REC_SECONDS)}s",
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

                rig_line = f"Rig: target {baseline_m:.3f}m" if baseline_m else "Rig: target --"
                if solved_baseline_m is not None:
                    rig_line += f" | solved {solved_baseline_m:.3f}m"
                if baseline_pair is not None:
                    rig_line += f" | pair {baseline_pair[0]}-{baseline_pair[1]}"
                cv2.putText(
                    panel,
                    rig_line,
                    (APP_PAD, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (177, 188, 204),
                    1,
                    cv2.LINE_AA,
                )
                y += 24

                body_line = (
                    f"Body: H {resolved_body.height_m:.2f}m | "
                    f"thigh {resolved_body.segments_m['thigh']:.2f}m | "
                    f"shank {resolved_body.segments_m['shank']:.2f}m"
                )
                cv2.putText(
                    panel,
                    body_line,
                    (APP_PAD, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (177, 188, 204),
                    1,
                    cv2.LINE_AA,
                )
                y += 30

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

                pane_inner_w = video_area_w - 2 * VIDEO_PAD
                pane_inner_h = pane_h - 2 * VIDEO_PAD
                if len(active_camera_ids) >= 2 and show_dual_cam_strip:
                    order = [primary_camera_id] + [
                        c for c in active_camera_ids if c != primary_camera_id
                    ]
                    video_pane = _compose_dual_cam_and_3d(
                        order[:2],
                        merged_display,
                        per_cam_results,
                        pts_3d_live,
                        vis_3d_live,
                        pane_inner_w,
                        pane_inner_h,
                        show_camera_bg=show_camera_bg,
                        show_skeleton=show_skeleton,
                        show_joints=show_joints,
                        show_vis=show_vis,
                        use_triangulation=use_triangulation,
                    )
                else:
                    video_pane = fit_video_to_pane(video, pane_inner_w, pane_inner_h)
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

                pane_inner_w = pane_w - 2 * VIDEO_PAD
                pane_inner_h = pane_h - 2 * VIDEO_PAD
                if len(active_camera_ids) >= 2 and show_dual_cam_strip:
                    order = [primary_camera_id] + [
                        c for c in active_camera_ids if c != primary_camera_id
                    ]
                    video_pane = _compose_dual_cam_and_3d(
                        order[:2],
                        merged_display,
                        per_cam_results,
                        pts_3d_live,
                        vis_3d_live,
                        pane_inner_w,
                        pane_inner_h,
                        show_camera_bg=show_camera_bg,
                        show_skeleton=show_skeleton,
                        show_joints=show_joints,
                        show_vis=show_vis,
                        use_triangulation=use_triangulation,
                    )
                else:
                    video_pane = fit_video_to_pane(video, pane_inner_w, pane_inner_h)
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
            elif key in (ord("d"), ord("D")):
                if len(active_camera_ids) >= 2:
                    show_dual_cam_strip = not show_dual_cam_strip
            elif key in (ord("a"), ord("A")):
                show_polar = not show_polar
                if not show_polar:
                    palette_gallery_open = False
            elif key in (ord("g"), ord("G")) and show_polar:
                palette_gallery_open = not palette_gallery_open
            elif key in (ord("b"), ord("B")):
                cancelled, new_baseline_m = _ask_optional_float_dialog(
                    "Camera baseline",
                    "Distance between the primary camera and secondary camera in meters.\n\nLeave blank to clear and use calibration scale.",
                    baseline_m,
                )
                if not cancelled:
                    baseline_m = new_baseline_m
                    metric_scale, solved_baseline_m, baseline_pair = _compute_metric_scale(
                        active_camera_ids,
                        primary_camera_id,
                        calibrations,
                        baseline_m,
                    )
                    _save_runtime_setup(
                        active_camera_ids,
                        primary_camera_id,
                        use_triangulation,
                        camera_rotations,
                        baseline_m,
                    )
                    if baseline_m is None:
                        print("Cleared camera baseline override.")
                    else:
                        print(f"Saved camera baseline: {baseline_m:.3f} m")
            elif key in (ord("m"), ord("M")):
                updated_profile = _edit_body_profile_dialog(body_profile)
                if updated_profile is not None:
                    body_profile = updated_profile
                    save_body_profile(body_profile)
                    print("Saved body profile.")

    finally:
        multi_cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if SHOW_CONSOLE:
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
