"""Pose dashboard: MediaPipe pose estimation with angle tracking and recording."""

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

#
import cv2
import numpy as np
from adaptive_filter import LandmarkArrayFilter
from body_profile import (
    BodyProfile,
    ResolvedBodyProfile,
    load_body_profile,
    resolve_body_profile,
    save_body_profile,
)
from config import (
    ACCENT,
    ACCENT_ALT,
    ACCENT_SUCCESS,
    APP_PAD,
    CAM_H,
    CAM_W,
    CARD_GAP,
    CLIP_H_FRAC,
    DEFAULT_POLAR_PALETTE_KEY,
    LIVE_BUFFER_SECONDS,
    LOG_INTERVAL,
    MAX_REC_SECONDS,
    MIRROR_VIEW,
    INFER_INPUT_SCALE,
    MULTICAM_NONPRIMARY_INFER_EVERY_N,
    MULTICAM_SECONDARY_INFER_INPUT_SCALE,
    MULTICAM_USB_EXTRA_DRAIN,
    PRIMARY_INFER_INPUT_SCALE,
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
    PANEL_FONT,
    PANEL_TEXT_THICK,
    PANEL_TITLE_THICK,
    PANEL_W,
    POSE_MIN_DETECTION_CONF,
    POSE_MIN_PRESENCE_CONF,
    POSE_MIN_TRACKING_CONF,
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
    TEXT_SECONDARY,
    TIMELINE_H,
    SMOOTH_2D_BETA,
    SMOOTH_2D_D_CUTOFF,
    SMOOTH_2D_ENABLED,
    SMOOTH_2D_MIN_CUTOFF,
    SMOOTH_3D_BETA,
    SMOOTH_3D_D_CUTOFF,
    SMOOTH_3D_ENABLED,
    SMOOTH_3D_MIN_CUTOFF,
    VIDEO_PAD,
    VIS_MIN,
    VIEW_H,
    VIEW_W,
    WINDOW,
    WINDOW_BG,
    ensure_task_file,
    get_polar_palette,
)
from camera_config import (
    CameraInfo,
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
from pose_processor import ANGLE_KEYS, best_2d_angle_values, process_pose, round_deg
from state import angles_live, angles_rec, pose_live, pose_rec, t_live, t_rec
from triangulation import compute_angles_3d, process_multi_cam_poses
from ui.console import console_line
from ui.drawing import (
    draw_controls_section,
    draw_lr_table,
    draw_panel_header,
    draw_stat_box,
    fit_video_to_pane,
    panel_bg,
)
from ui.export_gallery import draw_exports_gallery, list_polar_export_files
from ui.export_viewer import draw_export_viewer_in_stage, export_viewer_step_index
from ui.stage_modal import compute_video_stage_bounds, darken_full_image
from ui.timeline import (
    clear_hitboxes,
    draw_timeline_ui,
    extract_segment_by_time,
    make_mouse_cb,
    trim_time_buffer,
)
from visualization.clip_preview import draw_pose_clip
from visualization.polar_plot import (
    draw_palette_preview,
    draw_polar_plot_segment,
    export_polar_plot_png,
)
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


def _open_file_default_app(path: Path) -> None:
    p = str(path)
    try:
        if sys.platform == "win32":
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.run(["open", p], check=False)
        else:
            subprocess.run(["xdg-open", p], check=False)
    except OSError:
        pass


def _sanitize_polar_export_basename(name: str) -> str:
    """Strip path chars and Windows-forbidden symbols only; keep spaces and most punctuation."""
    s = name.strip().strip('"')
    s = s.replace("\\", "").replace("/", "")
    for c in '<>:"|?*':
        s = s.replace(c, "")
    s = s.rstrip(" .")
    return s


def _polar_export_save_dialog(
    exp_dir: Path, default_chart_title: str
) -> tuple[Path, str] | None:
    """
    Ask for file name (saved under exp_dir). Returns (full_path, chart_title) or None if cancelled.
    Blank name -> polar.png with default_chart_title on the figure.
    """
    import tkinter as tk
    from tkinter import ttk

    result: list[tuple[Path, str] | None] = [None]

    root = tk.Tk()
    root.withdraw()
    top = tk.Toplevel(root)
    top.title("Export polar plot")
    top.attributes("-topmost", True)
    top.resizable(False, False)

    frm = ttk.Frame(top, padding=14)
    frm.grid(row=0, column=0, sticky="nsew")

    ttk.Label(
        frm,
        text="File name (saved in polar_exports/)",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(
        frm,
        text="Type the file name you want. .png is added only if you omit it.\n"
        "Leave blank to save as polar.png with the automatic chart title.",
        font=("Segoe UI", 9),
        wraplength=420,
        justify="left",
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 8))

    name_var = tk.StringVar(value="")
    entry = ttk.Entry(frm, textvariable=name_var, width=44, font=("Segoe UI", 10))
    entry.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    entry.focus_set()

    def on_ok() -> None:
        base = _sanitize_polar_export_basename(name_var.get())
        if not base:
            result[0] = (exp_dir / "polar.png", default_chart_title)
        else:
            fname = base if base.lower().endswith(".png") else f"{base}.png"
            chart = fname[:-4] if fname.lower().endswith(".png") else fname
            result[0] = (exp_dir / fname, chart)
        top.destroy()
        root.destroy()

    def on_cancel() -> None:
        result[0] = None
        top.destroy()
        root.destroy()

    btn_row = ttk.Frame(frm)
    btn_row.grid(row=3, column=0, columnspan=2, sticky="e", pady=(4, 0))
    ttk.Button(btn_row, text="Cancel", command=on_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btn_row, text="Save", command=on_ok).grid(row=0, column=1)

    top.protocol("WM_DELETE_WINDOW", on_cancel)
    top.grab_set()
    entry.bind("<Return>", lambda e: on_ok())
    root.wait_window(top)
    return result[0]


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
    overlay_offsets_px: dict[str, int],
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
            dx = int(overlay_offsets_px.get(cid, 0))
            if dx != 0:
                pts = {k: (np.asarray(v, dtype=np.float32) + np.array([dx, 0], dtype=np.float32)) for k, v in pts.items()}
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
    overlay_offsets_px: dict[str, int],
    show_camera_bg: bool,
    show_skeleton: bool,
    show_joints: bool,
    show_vis: bool,
    use_triangulation: bool,
) -> np.ndarray:
    gap = 10
    c0, c1 = cam_order[0], cam_order[1]
    half_h = max(120, (pane_inner_h - gap) // 2)

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
        overlay_offsets_px=overlay_offsets_px,
        show_camera_bg=show_camera_bg,
        show_skeleton=show_skeleton,
        show_joints=show_joints,
        show_vis=show_vis,
    )
    v1 = _annotate_cam_video(
        c1,
        f1,
        per_cam_results,
        overlay_offsets_px=overlay_offsets_px,
        show_camera_bg=show_camera_bg,
        show_skeleton=show_skeleton,
        show_joints=show_joints,
        show_vis=show_vis,
    )

    p0 = fit_video_to_pane(v0, pane_inner_w, half_h)
    p1 = fit_video_to_pane(v1, pane_inner_w, half_h)
    cv2.putText(
        p0,
        f"Cam {c0}  (primary)",
        (8, 24),
        PANEL_FONT,
        0.55,
        ACCENT_SUCCESS,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    cv2.putText(
        p1,
        f"Cam {c1}",
        (8, 24),
        PANEL_FONT,
        0.55,
        ACCENT_ALT,
        PANEL_TEXT_THICK,
        cv2.LINE_AA,
    )
    hsep = np.full((gap, pane_inner_w, 3), 24, dtype=np.uint8)
    return np.vstack((p0, hsep, p1))


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
                        (10, 30), PANEL_FONT, 0.65, (0, 255, 0), 2)
            cv2.putText(disp, "0-9=toggle | P=primary | Enter=done",
                        (10, 60), PANEL_FONT, 0.45, (200, 200, 200), 1)
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


def _edit_body_profile_form_dialog(resolved: ResolvedBodyProfile) -> BodyProfile | None:
    """Single modal with all segments; defaults match the resolved values shown on the dashboard."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    segment_keys = [
        ("shoulder_width", "Shoulder width (m)"),
        ("hip_width", "Hip width (m)"),
        ("torso", "Torso length (m)"),
        ("upper_arm", "Upper arm (m)"),
        ("forearm", "Forearm (m)"),
        ("thigh", "Thigh (m)"),
        ("shank", "Shank (m)"),
    ]

    root = tk.Tk()
    root.withdraw()
    top = tk.Toplevel(root)
    top.title("Body profile")
    top.attributes("-topmost", True)
    top.resizable(False, False)

    frm = ttk.Frame(top, padding=14)
    frm.grid(row=0, column=0, sticky="nsew")

    ttk.Label(
        frm,
        text="Edit saved measurements (meters)",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
    ttk.Label(
        frm,
        text="Values match what the dashboard is using now. Empty field clears that measurement.",
        font=("Segoe UI", 9),
        wraplength=400,
        justify="left",
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

    entries: dict[str, tk.Entry] = {}
    row = 2

    ttk.Label(frm, text="Height (m)", width=22, anchor="w").grid(
        row=row, column=0, sticky="w", pady=2
    )
    e_h = ttk.Entry(frm, width=16, font=("Segoe UI", 10))
    e_h.grid(row=row, column=1, sticky="e", pady=2)
    e_h.insert(0, f"{float(resolved.height_m):.3f}")
    entries["height_m"] = e_h
    row += 1

    for key, label in segment_keys:
        ttk.Label(frm, text=label, width=22, anchor="w").grid(
            row=row, column=0, sticky="w", pady=2
        )
        e = ttk.Entry(frm, width=16, font=("Segoe UI", 10))
        e.grid(row=row, column=1, sticky="e", pady=2)
        v = resolved.segments_m.get(key)
        if v is not None and float(v) > 0.0:
            e.insert(0, f"{float(v):.3f}")
        entries[key] = e
        row += 1

    out: list[BodyProfile | None] = [None]

    def parse_entry(widget: tk.Entry) -> float | None:
        s = widget.get().strip()
        if not s:
            return None
        return float(s)

    def on_ok() -> None:
        try:
            hv = parse_entry(entries["height_m"])
            if hv is not None and hv <= 0.0:
                raise ValueError("height must be positive")
            segs: dict[str, float] = {}
            for key, _lab in segment_keys:
                sv = parse_entry(entries[key])
                if sv is None:
                    continue
                if sv <= 0.0:
                    raise ValueError(f"{key} must be positive or empty")
                segs[key] = sv
            out[0] = BodyProfile(height_m=hv, segments_m=segs)
        except ValueError as ex:
            messagebox.showerror("Body profile", str(ex), parent=top)
            return
        top.destroy()
        root.destroy()

    def on_cancel() -> None:
        out[0] = None
        top.destroy()
        root.destroy()

    btn_row = ttk.Frame(frm)
    btn_row.grid(row=row, column=0, columnspan=2, sticky="e", pady=(12, 0))
    ttk.Button(btn_row, text="Cancel", command=on_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btn_row, text="Save", command=on_ok).grid(row=0, column=1)

    top.protocol("WM_DELETE_WINDOW", on_cancel)
    top.grab_set()
    root.wait_window(top)
    return out[0]


def _resolve_runtime_camera_setup(
    calibrations,
    last_setup,
    connected,
) -> tuple[list[str] | None, str | None, bool, dict[str, int], float | None, dict[str, int]]:
    """
    Restore runtime rig from last_camera_setup when available.

    Camera indices can drift on Windows between runs (same hardware, different
    numeric index). If a saved index is missing, auto-fill with currently
    connected extras so the app starts with a live rig.
    """
    connected_ids = [str(c.index) for c in connected]
    selected_ids: list[str] = []

    if last_setup and last_setup.selected_camera_ids:
        saved_ids: list[str] = []
        for cid in last_setup.selected_camera_ids:
            s = str(cid)
            if not is_local_camera_id(s):
                continue
            if s not in saved_ids:
                saved_ids.append(s)

        present = [cid for cid in saved_ids if cid in connected_ids]
        missing = [cid for cid in saved_ids if cid not in connected_ids]
        extras = [cid for cid in connected_ids if cid not in present]
        selected_ids = list(present)
        if missing and extras:
            n = min(len(missing), len(extras))
            replacements = extras[:n]
            selected_ids.extend(replacements)
            print(
                "Camera index drift detected. "
                f"Missing saved indices {missing}; substituted live indices {replacements}."
            )
        elif saved_ids and len(saved_ids) >= 2 and present and missing:
            # Probe often returns only one USB webcam while the second is still absent from
            # the index list. If we stop here, we silently drop the saved rig's other camera.
            selected_ids = list(saved_ids)
            print(
                "Multi-camera rig from last_camera_setup.json: opening saved indices "
                f"{saved_ids} (probe reported {sorted(connected_ids, key=int)}). "
                "Second camera may appear after USB settles; if open fails, check cables/ports."
            )

    if not selected_ids:
        for cid in connected_ids:
            if cid in calibrations and cid not in selected_ids:
                selected_ids.append(cid)
    if not selected_ids:
        selected_ids = list(connected_ids)

    if not selected_ids:
        return None, None, False, {}, None

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
    overlay_offsets_px = {}
    if last_setup and getattr(last_setup, "camera_overlay_offsets_px", None):
        overlay_offsets_px = {
            cid: int(last_setup.camera_overlay_offsets_px.get(cid, 0))
            for cid in selected_ids
            if cid in last_setup.camera_overlay_offsets_px
        }

    # 3D whenever two+ calibrated views include extrinsics (ignore saved use_triangulation=False).
    use_triangulation = len(selected_ids) >= 2 and all(
        calibrations.get(cid) and getattr(calibrations.get(cid), "extrinsics", None)
        for cid in selected_ids
    )
    baseline_m = last_setup.camera_baseline_m if last_setup else None
    return selected_ids, primary, use_triangulation, rotations, baseline_m, overlay_offsets_px


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
    camera_overlay_offsets_px: dict[str, int],
) -> None:
    save_last_setup(
        LastCameraSetup(
            selected_camera_ids=active_camera_ids,
            primary_camera_id=primary_camera_id,
            use_triangulation=use_triangulation,
            camera_rotations=camera_rotations,
            camera_baseline_m=baseline_m,
            camera_overlay_offsets_px=camera_overlay_offsets_px,
        )
    )


def _point_in_rect(x, y, rect):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _draw_palette_gallery(img, selected_palette_key, stage_bounds):
    sx0, sy0, sx1, sy1 = stage_bounds
    sw = max(1, sx1 - sx0)
    sh = max(1, sy1 - sy0)
    darken_full_image(img, PALETTE_GALLERY_SCRIM, 0.56)

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
    modal_x0 = sx0 + max(12, (sw - modal_w) // 2)
    modal_y0 = sy0 + max(12, (sh - modal_h) // 2)
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
        PANEL_FONT,
        0.76,
        PALETTE_GALLERY_TEXT,
        PANEL_TITLE_THICK,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        PALETTE_GALLERY_SUBTITLE,
        (modal_x0 + inner_pad, modal_y0 + 52),
        PANEL_FONT,
        0.46,
        PALETTE_GALLERY_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "ESC or click outside to close",
        (modal_x1 - 220, modal_y0 + 28),
        PANEL_FONT,
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
            PANEL_FONT,
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
            PANEL_FONT,
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

    calibrations = load_calibrations()
    last_setup = load_last_setup()
    body_profile = load_body_profile()
    active_camera_ids = None
    primary_camera_id = None
    use_triangulation = False
    camera_rotations: dict[str, int] = {}
    camera_overlay_offsets_px: dict[str, int] = {}
    baseline_m = last_setup.camera_baseline_m if last_setup else None

    max_probe_idx = 7
    if last_setup and last_setup.selected_camera_ids:
        try:
            max_saved = max(int(cid) for cid in last_setup.selected_camera_ids if str(cid).isdigit())
            max_probe_idx = max(max_probe_idx, max_saved + 5)
            if len(last_setup.selected_camera_ids) >= 2:
                max_probe_idx = max(max_probe_idx, 16)
        except ValueError:
            pass
    connected = detect_connected_cameras(max_index=max_probe_idx, timeout=1.2)
    if not connected and last_setup and last_setup.selected_camera_ids:
        # Fast probe can miss slow-enumerating devices on Windows; fall back to saved rig IDs.
        print(
            "Camera probe returned no devices; falling back to saved camera indices from "
            "last_camera_setup.json."
        )
        connected = [
            CameraInfo(index=int(cid), label=f"Saved camera {cid}")
            for cid in last_setup.selected_camera_ids
            if str(cid).isdigit()
        ]
    runtime_setup = _resolve_runtime_camera_setup(calibrations, last_setup, connected)
    if runtime_setup[0]:
        active_camera_ids, primary_camera_id, use_triangulation, camera_rotations, baseline_m, camera_overlay_offsets_px = (
            runtime_setup
        )
    # Do not save last_camera_setup on every launch: that used to overwrite a multi-cam
    # rig with a single detected camera and drop indices like 2 from the JSON.
    if active_camera_ids is None:
        connected_ids = [str(c.index) for c in connected if str(c.index).isdigit()]
        if connected_ids:
            active_camera_ids = [connected_ids[0]]
            primary_camera_id = connected_ids[0]
        else:
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

    def _runtime_calibration_aliases(
        selected_ids: list[str],
        saved_setup: LastCameraSetup | None,
        base_calibrations,
    ):
        """
        Build runtime calibration dict with index-drift aliasing.

        Example: saved rig [1,5], live rig remapped to [1,4], calibrations exist for [1,5].
        Alias 4 -> 5 at runtime so triangulation can continue without editing JSON.
        """
        out = dict(base_calibrations)
        alias_map: dict[str, str] = {}
        if not saved_setup or not saved_setup.selected_camera_ids:
            return out, alias_map
        saved = [str(cid) for cid in saved_setup.selected_camera_ids if str(cid).isdigit()]
        missing_saved = [cid for cid in saved if cid not in selected_ids and cid in base_calibrations]
        uncal_live = [cid for cid in selected_ids if cid not in out]

        # Primary path: alias from saved rig entries that disappeared.
        alias_sources = list(missing_saved)
        # Fallback path: if saved rig no longer contains old index IDs (e.g. user already saved [1,4]),
        # map any uncalibrated live IDs to remaining calibrated IDs not in the live rig.
        if not alias_sources and uncal_live:
            alias_sources = [
                cid for cid in base_calibrations.keys()
                if is_local_camera_id(str(cid)) and str(cid) not in selected_ids
            ]

        if alias_sources and len(alias_sources) == len(uncal_live):
            for live_cid, saved_cid in zip(
                sorted(uncal_live, key=int), sorted([str(x) for x in alias_sources], key=int)
            ):
                out[live_cid] = base_calibrations[saved_cid]
                alias_map[live_cid] = saved_cid
                print(f"Runtime calibration alias: camera {live_cid} -> saved calibration {saved_cid}")
        return out, alias_map

    runtime_calibrations, runtime_alias_map = _runtime_calibration_aliases(
        active_camera_ids, last_setup, calibrations
    )
    use_triangulation = len(active_camera_ids) >= 2 and all(
        runtime_calibrations.get(cid)
        and getattr(runtime_calibrations.get(cid), "extrinsics", None)
        for cid in active_camera_ids
    )
    if runtime_alias_map and use_triangulation:
        print(
            "Index drift alias active; using saved extrinsics for remapped camera indices "
            "for this run."
        )

    metric_scale, solved_baseline_m, baseline_pair = _compute_metric_scale(
        active_camera_ids,
        primary_camera_id,
        runtime_calibrations,
        baseline_m,
    )

    usb_read_order = [primary_camera_id] + [
        c for c in active_camera_ids if c != primary_camera_id
    ]
    landmarker: vision.PoseLandmarker | None = None
    try:
        multi_cap = MultiCameraReader.from_camera_ids(
            list(active_camera_ids),
            CAM_W,
            CAM_H,
            read_order=usb_read_order,
        )
    except RuntimeError as e:
        # Fast probe may miss slow-enumerating devices. Retry once with a slower probe
        # and re-resolve the runtime rig (including index-drift remap).
        print(f"Fast startup open failed: {e}")
        print("Retrying camera detection with slower probe...")
        connected_slow = detect_connected_cameras(max_index=max_probe_idx, timeout=3.0)
        runtime_setup2 = _resolve_runtime_camera_setup(calibrations, last_setup, connected_slow)
        if not runtime_setup2[0]:
            raise
        active_camera_ids, primary_camera_id, use_triangulation, camera_rotations, baseline_m, camera_overlay_offsets_px = (
            runtime_setup2
        )
        runtime_calibrations, runtime_alias_map = _runtime_calibration_aliases(
            active_camera_ids, last_setup, calibrations
        )
        use_triangulation = len(active_camera_ids) >= 2 and all(
            runtime_calibrations.get(cid)
            and getattr(runtime_calibrations.get(cid), "extrinsics", None)
            for cid in active_camera_ids
        )
        if runtime_alias_map and use_triangulation:
            print(
                "Index drift alias active; using saved extrinsics for remapped camera indices "
                "for this run."
            )
        metric_scale, solved_baseline_m, baseline_pair = _compute_metric_scale(
            active_camera_ids,
            primary_camera_id,
            runtime_calibrations,
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
    print("Camera source uses saved rig with auto-remap for missing indices.")
    if len(active_camera_ids) >= 2:
        print(
            f"Multi-cam: pose IMAGE mode + one model (low latency); "
            f"non-primary infer every {MULTICAM_NONPRIMARY_INFER_EVERY_N} tick(s), "
            f"secondary scale {MULTICAM_SECONDARY_INFER_INPUT_SCALE}, "
            f"USB extra drain +{MULTICAM_USB_EXTRA_DRAIN} (MULTICAM_* in config.py)."
        )

    # Single-camera: VIDEO mode (internal tracker). Multi-cam: IMAGE mode + detect() so one
    # model alternates streams without wrong VIDEO state; 2x VIDEO models was heavy and laggy.
    pose_image_mode = len(active_camera_ids) >= 2
    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                vision.RunningMode.IMAGE
                if pose_image_mode
                else vision.RunningMode.VIDEO
            ),
            num_poses=1,
            min_pose_detection_confidence=POSE_MIN_DETECTION_CONF,
            min_pose_presence_confidence=POSE_MIN_PRESENCE_CONF,
            min_tracking_confidence=POSE_MIN_TRACKING_CONF,
            output_segmentation_masks=False,
        )
    )
    mp_pose_ts = 0
    if pose_image_mode:
        print(
            "Pose: IMAGE mode for multi-cam — temporal smoothing is from your 2D One Euro filter."
        )

    # Runtime undistortion improves edge quality substantially (often where pose fails first).
    # Keep it disabled for remapped alias cameras, where intrinsics may belong to a different
    # physical device and would bend geometry/overlays.
    undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cid in active_camera_ids:
        if cid in runtime_alias_map:
            continue
        cal = runtime_calibrations.get(cid)
        if cal is None or getattr(cal, "intrinsics", None) is None:
            continue
        K = np.asarray(cal.intrinsics.K, dtype=np.float64)
        dist = np.asarray(cal.intrinsics.dist, dtype=np.float64)
        if K.shape != (3, 3) or dist.size == 0:
            continue
        newK, _roi = cv2.getOptimalNewCameraMatrix(K, dist, (CAM_W, CAM_H), 0.0)
        m1, m2 = cv2.initUndistortRectifyMap(
            K, dist, None, newK, (CAM_W, CAM_H), cv2.CV_32FC1
        )
        undistort_maps[cid] = (m1, m2)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO
        )
    except cv2.error:
        pass
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
    exports_gallery_open = False
    exports_scroll_row = 0
    exports_gallery_max_row = 0
    exports_hitboxes: list = []
    exports_modal_rect = None
    polar_exp_dir = Path(__file__).resolve().parent / "polar_exports"
    show_dual_cam_strip = len(active_camera_ids) >= 2
    display_frame_cache: dict[str, np.ndarray] = {}
    export_viewer_ctx: dict = {
        "open": False,
        "paths": [],
        "idx": 0,
        "img_rect": None,
    }
    # Dual-cam latency control: infer one camera per tick, reuse the other briefly.
    pose_cache: dict[str, tuple] = {}
    pose_cache_ts: dict[str, float] = {}
    pose_max_stale_s = 0.25
    lm_filters_2d: dict[str, LandmarkArrayFilter] = {}
    lm_filter_3d: LandmarkArrayFilter | None = None
    if SMOOTH_3D_ENABLED:
        lm_filter_3d = LandmarkArrayFilter(
            33,
            3,
            min_cutoff=SMOOTH_3D_MIN_CUTOFF,
            beta=SMOOTH_3D_BETA,
            d_cutoff=SMOOTH_3D_D_CUTOFF,
        )

    def _close_export_viewer() -> None:
        export_viewer_ctx["open"] = False
        export_viewer_ctx["paths"] = []
        export_viewer_ctx["idx"] = 0
        export_viewer_ctx["img_rect"] = None

    def mouse_cb(event, x, y, flags, param):
        nonlocal \
            palette_gallery_open, \
            selected_palette_key, \
            palette_modal_rect, \
            palette_hitboxes, \
            exports_gallery_open, \
            exports_scroll_row, \
            exports_gallery_max_row

        if export_viewer_ctx["open"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                r = export_viewer_ctx.get("img_rect")
                paths = export_viewer_ctx["paths"]
                if (
                    r is not None
                    and paths
                    and _point_in_rect(x, y, r)
                ):
                    x0, y0, x1, y1 = r
                    w = max(1, x1 - x0)
                    t = w // 3
                    if x < x0 + t:
                        export_viewer_ctx["idx"] = max(0, export_viewer_ctx["idx"] - 1)
                    elif x > x1 - t:
                        export_viewer_ctx["idx"] = min(
                            len(paths) - 1, export_viewer_ctx["idx"] + 1
                        )
                return
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
                return

        if exports_gallery_open:
            if event == cv2.EVENT_LBUTTONDOWN:
                for x0, y0, x1, y1, pth in exports_hitboxes:
                    if _point_in_rect(x, y, (x0, y0, x1, y1)):
                        fresh = list_polar_export_files(polar_exp_dir)
                        try:
                            idx = fresh.index(pth)
                        except ValueError:
                            fresh = [pth]
                            idx = 0
                        export_viewer_ctx["paths"] = fresh
                        export_viewer_ctx["idx"] = idx
                        export_viewer_ctx["open"] = True
                        export_viewer_ctx["img_rect"] = None
                        exports_gallery_open = False
                        return
                if exports_modal_rect is None or (
                    not _point_in_rect(x, y, exports_modal_rect)
                ):
                    exports_gallery_open = False
                return
            ev_wheel = getattr(cv2, "EVENT_MOUSEWHEEL", 10)
            if event == ev_wheel:
                delta = (flags >> 16) & 0xFFFF
                if delta >= 32768:
                    delta -= 65536
                if delta > 0:
                    exports_scroll_row = max(0, exports_scroll_row - 1)
                elif delta < 0:
                    exports_scroll_row = min(
                        exports_gallery_max_row, exports_scroll_row + 1
                    )
                return
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
                return

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
            # Drain queued camera buffers before each read so overlays use near-live frames.
            raw_batch = multi_cap.read_batch(
                drain_first=True,
                extra_drain=MULTICAM_USB_EXTRA_DRAIN
                if len(active_camera_ids) >= 2
                else 0,
            )
            for cid in active_camera_ids:
                if cid not in raw_batch:
                    continue
                frame = raw_batch[cid]
                if cid in undistort_maps:
                    m1, m2 = undistort_maps[cid]
                    frame = cv2.remap(frame, m1, m2, interpolation=cv2.INTER_LINEAR)
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
            t_now_s = time.time()
            # Primary: infer every tick. Non-primary: every N ticks when 2+ cams (USB + CPU).
            infer_ids = [cid for cid in active_camera_ids if cid in frames_by_id]
            infer_stride = max(1, int(MULTICAM_NONPRIMARY_INFER_EVERY_N))

            for cid in infer_ids:
                if cid not in frames_by_id:
                    continue
                if (
                    len(active_camera_ids) >= 2
                    and infer_stride > 1
                    and cid != primary_camera_id
                    and cid in pose_cache
                    and (frame_i % infer_stride) != 0
                ):
                    prev = pose_cache[cid]
                    if prev[1] is not None:
                        pose_cache_ts[cid] = t_now_s
                        continue
                frame = frames_by_id[cid]
                if cid == primary_camera_id:
                    eff_scale = float(PRIMARY_INFER_INPUT_SCALE)
                elif len(active_camera_ids) >= 2:
                    eff_scale = float(MULTICAM_SECONDARY_INFER_INPUT_SCALE)
                else:
                    eff_scale = float(INFER_INPUT_SCALE)
                if eff_scale < 0.999:
                    ih, iw = frame.shape[:2]
                    nw = max(160, int(round(iw * eff_scale)))
                    nh = max(120, int(round(ih * eff_scale)))
                    infer_frame = cv2.resize(
                        frame, (nw, nh), interpolation=cv2.INTER_AREA
                    )
                else:
                    infer_frame = frame
                inf_h, inf_w = infer_frame.shape[:2]
                rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                if pose_image_mode:
                    result = landmarker.detect(mp_img)
                else:
                    mp_pose_ts += 1
                    result = landmarker.detect_for_video(mp_img, mp_pose_ts)
                lm = (
                    result.pose_landmarks[0]
                    if (result.pose_landmarks and len(result.pose_landmarks) > 0)
                    else None
                )
                if lm is None:
                    filt = lm_filters_2d.get(cid)
                    if filt is not None:
                        filt.reset_all()
                h, w = frame.shape[:2]
                out = process_pose(lm, h, w, infer_h=inf_h, infer_w=inf_w)
                pts, vis, pts_norm, vis_arr, vals, pts_norm_snapshot, vis_snapshot = out
                if (
                    SMOOTH_2D_ENABLED
                    and pts_norm_snapshot is not None
                    and vis_snapshot is not None
                ):
                    filt = lm_filters_2d.get(cid)
                    if filt is None:
                        filt = LandmarkArrayFilter(
                            33,
                            2,
                            min_cutoff=SMOOTH_2D_MIN_CUTOFF,
                            beta=SMOOTH_2D_BETA,
                            d_cutoff=SMOOTH_2D_D_CUTOFF,
                        )
                        lm_filters_2d[cid] = filt
                    valid_2d = np.isfinite(vis_snapshot) & (vis_snapshot >= VIS_MIN)
                    pts_norm_f = filt.update(pts_norm_snapshot, t_now_s, update_mask=valid_2d)
                    pts_norm_snapshot = pts_norm_f.copy()
                    pts_norm = pts_norm_f.copy() if pts_norm is not None else pts_norm
                    if pts is not None:
                        pts = {
                            j: np.array(
                                [pts_norm_f[j, 0] * float(w), pts_norm_f[j, 1] * float(h)],
                                dtype=np.float32,
                            )
                            for j in range(pts_norm_f.shape[0])
                        }
                out = (pts, vis, pts_norm, vis_arr, vals, pts_norm_snapshot, vis_snapshot)
                cached = (cid, *out)
                pose_cache[cid] = cached
                pose_cache_ts[cid] = t_now_s

            for cid in active_camera_ids:
                cached = pose_cache.get(cid)
                ts = float(pose_cache_ts.get(cid, 0.0))
                if cached is None:
                    continue
                if (t_now_s - ts) > pose_max_stale_s:
                    continue
                per_cam_results.append(cached)
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
            primary_vals_2d = {k: np.nan for k in ANGLE_KEYS}
            pts_norm_snapshot = None
            vis_snapshot = None
            pts_3d_live = None
            vis_3d_live = None

            if use_triangulation and len(per_cam_results) >= 2:
                vals, pts_3d_live, vis_3d_live = process_multi_cam_poses(
                    per_cam_results,
                    runtime_calibrations,
                    metric_scale=metric_scale,
                    primary_camera_id=primary_camera_id,
                )
                for r in per_cam_results:
                    if r[0] == primary_camera_id:
                        # (cid, pts, vis, pts_norm, vis_arr, vals, pts_norm_snapshot, vis_snapshot)
                        primary_vals_2d = r[5]
                        break
                if (
                    SMOOTH_3D_ENABLED
                    and lm_filter_3d is not None
                    and pts_3d_live is not None
                    and vis_3d_live is not None
                ):
                    valid_3d = np.isfinite(vis_3d_live) & (vis_3d_live >= VIS_MIN)
                    pts_3d_live = lm_filter_3d.update(pts_3d_live, t_now_s, update_mask=valid_3d)
                    vals = compute_angles_3d(pts_3d_live, vis_3d_live)
                # If triangulation confidence drops, use best single-camera 2D angles
                # (secondary may still track when primary fails on one side of the frame).
                if (
                    pts_3d_live is None
                    or vis_3d_live is None
                    or int(np.sum(np.isfinite(vis_3d_live) & (vis_3d_live >= VIS_MIN))) < 10
                ):
                    vals = best_2d_angle_values(
                        per_cam_results, preferred_cam_id=primary_camera_id
                    )
                for r in per_cam_results:
                    if r[0] == primary_camera_id:
                        _, pts, vis, _, _, _, pts_norm_snapshot, vis_snapshot = r
                        if pts is not None:
                            dx = int(camera_overlay_offsets_px.get(primary_camera_id, 0))
                            if dx != 0:
                                pts = {k: (np.asarray(v, dtype=np.float32) + np.array([dx, 0], dtype=np.float32)) for k, v in pts.items()}
                            draw_skeleton_on_video(
                                video, pts, vis, show_skeleton, show_joints, show_vis
                            )
                        break
            else:
                vals = best_2d_angle_values(
                    per_cam_results, preferred_cam_id=primary_camera_id
                )
                for r in per_cam_results:
                    if r[0] == primary_camera_id:
                        _, pts, vis, _, _, _, pts_norm_snapshot, vis_snapshot = r
                        if pts is not None:
                            dx = int(camera_overlay_offsets_px.get(primary_camera_id, 0))
                            if dx != 0:
                                pts = {k: (np.asarray(v, dtype=np.float32) + np.array([dx, 0], dtype=np.float32)) for k, v in pts.items()}
                            draw_skeleton_on_video(
                                video, pts, vis, show_skeleton, show_joints, show_vis
                            )
                        break

            # Primary may have no pose while a secondary still tracks — polar/clip use best view.
            if pts_norm_snapshot is None and per_cam_results:
                best_vn = -1
                best_pns = None
                best_vs = None
                for r in per_cam_results:
                    _, _, _, _, _, _, pns, vs = r
                    if pns is None or vs is None:
                        continue
                    vn = int(np.sum(np.isfinite(vs) & (vs >= VIS_MIN)))
                    if vn > best_vn:
                        best_vn = vn
                        best_pns, best_vs = pns, vs
                if best_vn > 0 and best_pns is not None and best_vs is not None:
                    pts_norm_snapshot, vis_snapshot = best_pns, best_vs

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
                    44,
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
                    PANEL_FONT,
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
                    PANEL_FONT,
                    0.46,
                    TEXT_SECONDARY,
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
                    PANEL_FONT,
                    0.46,
                    TEXT_SECONDARY,
                    1,
                    cv2.LINE_AA,
                )
                y += 26

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
                        overlay_offsets_px=camera_overlay_offsets_px,
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
                        overlay_offsets_px=camera_overlay_offsets_px,
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

            stage_bounds = compute_video_stage_bounds(
                VIEW_W,
                VIEW_H,
                panel_w_eff=panel_w_eff,
                plot_enabled=plot_enabled,
                plot_w=PLOT_W,
                plot_pad=PLOT_PAD,
                video_pad=VIDEO_PAD,
            )

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
                age_bits = []
                for cid in active_camera_ids:
                    age = multi_cap.get_frame_age_ms(cid)
                    if age is not None:
                        age_bits.append(f"cam{cid}:{age:4.0f}ms")
                if age_bits:
                    line += " | " + " ".join(age_bits)
                sys.stdout.write("\r" + line + " " * 10)
                sys.stdout.flush()
                last_log = time.time()

            export_viewer_ctx["img_rect"] = None
            if exports_gallery_open:
                export_paths = list_polar_export_files(polar_exp_dir)
                exports_modal_rect, exports_hitboxes, exports_gallery_max_row = (
                    draw_exports_gallery(
                        dash,
                        stage_bounds=stage_bounds,
                        export_paths=export_paths,
                        scroll_row=exports_scroll_row,
                    )
                )
                exports_scroll_row = min(
                    exports_scroll_row, exports_gallery_max_row
                )
                palette_modal_rect = None
                palette_hitboxes = []
            elif palette_gallery_open and plot_enabled:
                exports_modal_rect = None
                exports_hitboxes = []
                palette_modal_rect, palette_hitboxes = _draw_palette_gallery(
                    dash, selected_palette_key, stage_bounds
                )
            else:
                exports_modal_rect = None
                exports_hitboxes = []
                palette_modal_rect = None
                palette_hitboxes = []

            if export_viewer_ctx["open"] and export_viewer_ctx["paths"]:
                paths_v = export_viewer_ctx["paths"]
                export_viewer_ctx["idx"] = max(
                    0, min(len(paths_v) - 1, export_viewer_ctx["idx"])
                )
                pv = paths_v[export_viewer_ctx["idx"]]
                darken_full_image(dash, PALETTE_GALLERY_SCRIM, 0.56)
                export_viewer_ctx["img_rect"] = draw_export_viewer_in_stage(
                    dash,
                    stage_bounds,
                    pv,
                    export_viewer_ctx["idx"],
                    len(paths_v),
                )

            cv2.imshow(WINDOW, dash)

            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF

            if export_viewer_ctx["open"] and export_viewer_ctx["paths"]:
                step_v = export_viewer_step_index(key_ex)
                if step_v is not None:
                    export_viewer_ctx["idx"] = max(
                        0,
                        min(
                            len(export_viewer_ctx["paths"]) - 1,
                            export_viewer_ctx["idx"] + step_v,
                        ),
                    )
                elif key == 27 or key == ord("q"):
                    _close_export_viewer()

            if not export_viewer_ctx["open"]:
                if key == ord("q"):
                    break
                elif key == 27:
                    palette_gallery_open = False
                    exports_gallery_open = False
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
                elif key in (ord("k"), ord("K")):
                    if len(active_camera_ids) >= 2:
                        ordered = sorted(active_camera_ids, key=int)
                        try:
                            ix = ordered.index(primary_camera_id)
                        except ValueError:
                            ix = -1
                        primary_camera_id = ordered[(ix + 1) % len(ordered)]
                        _save_runtime_setup(
                            active_camera_ids,
                            primary_camera_id,
                            use_triangulation,
                            camera_rotations,
                            baseline_m,
                            camera_overlay_offsets_px,
                        )
                        print(
                            f"Primary camera (main pane + full-res infer): "
                            f"{_short_label(primary_camera_id)}"
                        )
                elif key in (ord("a"), ord("A")):
                    show_polar = not show_polar
                    if not show_polar:
                        palette_gallery_open = False
                elif key in (ord("g"), ord("G")) and show_polar:
                    palette_gallery_open = not palette_gallery_open
                    if palette_gallery_open:
                        exports_gallery_open = False
                elif key in (ord("e"), ord("E")):
                    exports_gallery_open = not exports_gallery_open
                    if exports_gallery_open:
                        palette_gallery_open = False
                        exports_scroll_row = 0
                elif key in (ord("o"), ord("O")) and show_polar:
                    if seg_series is None or not seg_series:
                        print(
                            "Polar export: no segment in the current window (need more samples)."
                        )
                    else:
                        any_s = next(iter(seg_series.values()))
                        if int(any_s.shape[0]) <= 2:
                            print("Polar export: segment too short.")
                        else:
                            exp_dir = polar_exp_dir
                            if timeline_module.live_mode:
                                if seg_ts is not None and len(seg_ts) >= 2:
                                    wl = f"live {float(seg_ts[0]):.2f}-{float(seg_ts[-1]):.2f}s"
                                else:
                                    wl = "live"
                            else:
                                ws = float(timeline_module.pinned_start_t)
                                wl = f"review {ws:.2f}-{ws + SEG_SECONDS:.2f}s"
                            default_title = (
                                f"Angles (Polar)  ({wl})" if wl else "Angles (Polar)"
                            )
                            exp_dir.mkdir(parents=True, exist_ok=True)
                            choice = _polar_export_save_dialog(exp_dir, default_title)
                            if choice is None:
                                print("Polar export cancelled.")
                            else:
                                out_png, chart_title = choice
                                export_polar_plot_png(
                                    seg_series,
                                    selected_palette_key,
                                    out_png,
                                    title=chart_title,
                                )
                                print(f"Saved polar plot: {out_png}")
                                _open_file_default_app(out_png)
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
                            runtime_calibrations,
                            baseline_m,
                        )
                        _save_runtime_setup(
                            active_camera_ids,
                            primary_camera_id,
                            use_triangulation,
                            camera_rotations,
                            baseline_m,
                            camera_overlay_offsets_px,
                        )
                        if baseline_m is None:
                            print("Cleared camera baseline override.")
                        else:
                            print(f"Saved camera baseline: {baseline_m:.3f} m")
                elif key == ord("["):
                    camera_overlay_offsets_px[primary_camera_id] = int(
                        camera_overlay_offsets_px.get(primary_camera_id, 0) - 2
                    )
                    _save_runtime_setup(
                        active_camera_ids,
                        primary_camera_id,
                        use_triangulation,
                        camera_rotations,
                        baseline_m,
                        camera_overlay_offsets_px,
                    )
                    print(
                        f"{_short_label(primary_camera_id)} overlay x offset: {camera_overlay_offsets_px[primary_camera_id]} px"
                    )
                elif key == ord("]"):
                    camera_overlay_offsets_px[primary_camera_id] = int(
                        camera_overlay_offsets_px.get(primary_camera_id, 0) + 2
                    )
                    _save_runtime_setup(
                        active_camera_ids,
                        primary_camera_id,
                        use_triangulation,
                        camera_rotations,
                        baseline_m,
                        camera_overlay_offsets_px,
                    )
                    print(
                        f"{_short_label(primary_camera_id)} overlay x offset: {camera_overlay_offsets_px[primary_camera_id]} px"
                    )
                elif key == ord(";"):
                    secondary_ids = [c for c in active_camera_ids if c != primary_camera_id]
                    if secondary_ids:
                        sid = secondary_ids[0]
                        camera_overlay_offsets_px[sid] = int(
                            camera_overlay_offsets_px.get(sid, 0) - 2
                        )
                        _save_runtime_setup(
                            active_camera_ids,
                            primary_camera_id,
                            use_triangulation,
                            camera_rotations,
                            baseline_m,
                            camera_overlay_offsets_px,
                        )
                        print(
                            f"{_short_label(sid)} overlay x offset: {camera_overlay_offsets_px[sid]} px"
                        )
                elif key == ord("'"):
                    secondary_ids = [c for c in active_camera_ids if c != primary_camera_id]
                    if secondary_ids:
                        sid = secondary_ids[0]
                        camera_overlay_offsets_px[sid] = int(
                            camera_overlay_offsets_px.get(sid, 0) + 2
                        )
                        _save_runtime_setup(
                            active_camera_ids,
                            primary_camera_id,
                            use_triangulation,
                            camera_rotations,
                            baseline_m,
                            camera_overlay_offsets_px,
                        )
                        print(
                            f"{_short_label(sid)} overlay x offset: {camera_overlay_offsets_px[sid]} px"
                        )
                elif key in (ord("m"), ord("M")):
                    updated_profile = _edit_body_profile_form_dialog(resolved_body)
                    if updated_profile is not None:
                        body_profile = updated_profile
                        save_body_profile(body_profile)
                        print("Saved body profile.")

    finally:
        multi_cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        if SHOW_CONSOLE:
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
