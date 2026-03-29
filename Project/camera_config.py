"""Camera calibration storage and detection for multi-camera pose dashboard."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import time

import cv2
import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent
CALIBRATIONS_PATH = _PROJECT_DIR / "camera_calibrations.json"
LAST_SETUP_PATH = _PROJECT_DIR / "last_camera_setup.json"


# ---------------------------------------------------------------------------
# Data structures (JSON-serializable where possible)
# ---------------------------------------------------------------------------


@dataclass
class CameraInfo:
    """Detected local camera: OpenCV index plus optional label."""
    index: int
    label: str = ""


def is_local_camera_id(camera_id: str) -> bool:
    """True if camera_id is a non-negative integer index string (local / USB webcam)."""
    return camera_id.isdigit()


_INDEX_BACKENDS = []
for _b in ("CAP_DSHOW", "CAP_MSMF"):
    if hasattr(cv2, _b):
        _INDEX_BACKENDS.append(getattr(cv2, _b))
if not _INDEX_BACKENDS:
    _INDEX_BACKENDS.append(cv2.CAP_ANY)

# HD calibration opens: on Windows MSMF often negotiates 720p+ better than DSHOW+YUY2 (~800x600).
INTRINSICS_OPEN_BACKENDS: list[int] = []
if sys.platform == "win32":
    for _b in ("CAP_MSMF", "CAP_DSHOW"):
        if hasattr(cv2, _b):
            bv = getattr(cv2, _b)
            if bv not in INTRINSICS_OPEN_BACKENDS:
                INTRINSICS_OPEN_BACKENDS.append(bv)
else:
    INTRINSICS_OPEN_BACKENDS = list(_INDEX_BACKENDS)
if not INTRINSICS_OPEN_BACKENDS:
    INTRINSICS_OPEN_BACKENDS = list(_INDEX_BACKENDS)


def open_camera(
    camera_id: str,
    width: int = 1280,
    height: int = 720,
    *,
    prefer_mjpeg: bool = True,
    backends: list[int] | None = None,
) -> cv2.VideoCapture:
    """
    Open a VideoCapture for a local camera index (string like "0").
    Default backend order is DSHOW then MSMF; pass backends=INTRINSICS_OPEN_BACKENDS for HD calibration.

    MJPEG is requested for most modes; resolution is set before and after FOURCC so drivers
    do not stay stuck at ~800x600 YUY2.
    """
    idx = int(camera_id)
    order = backends if backends is not None else _INDEX_BACKENDS
    cap = cv2.VideoCapture()
    for backend in order:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            break
        cap.release()
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if prefer_mjpeg and max(width, height) >= 480:
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def drain_stale_frames(cap: cv2.VideoCapture, max_drain: int = 3) -> None:
    """
    Grab (but don't decode) up to max_drain queued frames to get closer to the live edge.
    Call before the "real" read().
    """
    for _ in range(max_drain):
        cap.grab()


import threading

# Serialize VideoCapture.read() across the process. Two threads calling read() on
# different USB webcams at the same time often corrupts or starves one stream on Windows.
_USB_READ_LOCK = threading.Lock()


class ThreadedCapture:
    """
    Continuously reads frames from a VideoCapture in a background thread.
    The main thread can call latest() at any time to get the most recent
    frame without blocking. Keeps only the newest frame (no queue buildup).

    USB webcams on the same controller often glitch if multiple devices read()
    at once; a process-wide lock serializes capture.read() across instances.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            with _USB_READ_LOCK:
                ok, frame = self.cap.read()
            if ok and frame is not None:
                t = time.time()
                with self._lock:
                    self._frame = frame
                    self._timestamp = t

    def latest(self):
        """Return (frame, wall_clock_timestamp) or (None, 0.0) if no frame yet."""
        with self._lock:
            return self._frame, self._timestamp

    def release(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


class MultiCameraReader:
    """
    Dashboard multi-USB capture: **main thread** reads every device back-to-back
    under the process USB lock.

    A background reader can race the slow MediaPipe loop and leave one camera
    stuck on bogus/black buffers while the other advances. Synchronous batch reads
    tie each displayed frame to one USB pass. On Windows, 2+ cameras open with
    MSMF-first backends (see INTRINSICS_OPEN_BACKENDS), which is usually more
    reliable than DSHOW for multiple USB webcams.
    """

    def __init__(
        self,
        caps_by_id: dict[str, cv2.VideoCapture],
        *,
        read_order: list[str] | None = None,
    ):
        if not caps_by_id:
            raise ValueError("MultiCameraReader needs at least one camera")
        self._caps = caps_by_id
        order = read_order if read_order is not None else list(caps_by_id.keys())
        self._read_order = [cid for cid in order if cid in caps_by_id]
        if len(self._read_order) != len(caps_by_id):
            raise ValueError("read_order must list each camera id exactly once")

    @classmethod
    def from_camera_ids(
        cls,
        camera_ids: list[str],
        width: int,
        height: int,
        *,
        read_order: list[str] | None = None,
    ) -> MultiCameraReader:
        caps: dict[str, cv2.VideoCapture] = {}
        multi = len(camera_ids) >= 2
        backends = INTRINSICS_OPEN_BACKENDS if multi and sys.platform == "win32" else None
        for cid in camera_ids:
            cap = open_camera(cid, width, height, backends=backends)
            if not cap.isOpened():
                for c in caps.values():
                    c.release()
                raise RuntimeError(f"Camera index {cid} not available (failed to open).")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps[cid] = cap
        inst = cls(caps, read_order=read_order or list(camera_ids))
        inst._warmup_reads()
        return inst

    def _warmup_reads(self) -> None:
        """Drop startup placeholder frames (often black) before the dashboard loop."""
        for _ in range(4):
            self.read_batch(drain_first=True)

    def read_batch(self, *, drain_first: bool = False) -> dict[str, np.ndarray]:
        """
        Read each camera once in read_order under the USB lock.
        Returns fresh copies (safe to use for the rest of the frame tick).
        """
        out: dict[str, np.ndarray] = {}
        with _USB_READ_LOCK:
            for cid in self._read_order:
                cap = self._caps[cid]
                if drain_first:
                    drain_stale_frames(cap, max_drain=3)
                ok, fr = cap.read()
                if ok and fr is not None:
                    out[cid] = fr.copy()
        return out

    def release(self) -> None:
        for cap in self._caps.values():
            cap.release()


@dataclass
class Intrinsics:
    """Camera intrinsics from OpenCV calibrateCamera."""
    K: np.ndarray  # 3x3 camera matrix
    dist: np.ndarray  # distortion coefficients
    image_size: tuple[int, int]  # (width, height)
    reproj_error: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "image_size": list(self.image_size),
            "reproj_error": self.reproj_error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Intrinsics:
        return cls(
            K=np.array(d["K"], dtype=np.float64),
            dist=np.array(d["dist"], dtype=np.float64),
            image_size=tuple(d["image_size"]),
            reproj_error=float(d.get("reproj_error", 0.0)),
        )


@dataclass
class Extrinsics:
    """Camera pose in world frame: R (3x3), t (3,) from solvePnP / stereo."""
    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # 3x1 translation (column vector convention in world coords)

    def to_dict(self) -> dict[str, Any]:
        return {
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Extrinsics:
        return cls(
            R=np.array(d["R"], dtype=np.float64),
            t=np.array(d["t"], dtype=np.float64),
        )


@dataclass
class Calibration:
    """Full calibration for one camera."""
    camera_id: str
    intrinsics: Intrinsics
    extrinsics: Extrinsics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "camera_id": self.camera_id,
            "intrinsics": self.intrinsics.to_dict(),
            "metadata": self.metadata,
        }
        if self.extrinsics is not None:
            out["extrinsics"] = self.extrinsics.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        ext = None
        if "extrinsics" in d and d["extrinsics"] is not None:
            ext = Extrinsics.from_dict(d["extrinsics"])
        return cls(
            camera_id=str(d["camera_id"]),
            intrinsics=Intrinsics.from_dict(d["intrinsics"]),
            extrinsics=ext,
            metadata=dict(d.get("metadata", {})),
        )


ROTATION_OPTIONS = [0, 90, 180, 270]


@dataclass
class LastCameraSetup:
    """Last-used camera selection for the dashboard."""
    selected_camera_ids: list[str]
    primary_camera_id: str
    use_triangulation: bool = True
    camera_rotations: dict[str, int] = field(default_factory=dict)  # cam_id -> degrees (0/90/180/270)
    camera_baseline_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_camera_ids": list(self.selected_camera_ids),
            "primary_camera_id": self.primary_camera_id,
            "use_triangulation": self.use_triangulation,
            "camera_rotations": dict(self.camera_rotations),
            "camera_baseline_m": self.camera_baseline_m,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LastCameraSetup:
        baseline_raw = d.get("camera_baseline_m")
        baseline_m = None
        if baseline_raw is not None:
            try:
                baseline_m = float(baseline_raw)
            except (TypeError, ValueError):
                baseline_m = None
        raw_selected = [str(x) for x in d.get("selected_camera_ids", [])]
        selected = [cid for cid in raw_selected if is_local_camera_id(cid)]
        rot_raw = dict(d.get("camera_rotations", {}))
        camera_rotations = {
            str(k): int(v)
            for k, v in rot_raw.items()
            if is_local_camera_id(str(k))
        }
        primary = str(d.get("primary_camera_id", ""))
        if not is_local_camera_id(primary) or primary not in selected:
            primary = selected[0] if selected else ""
        return cls(
            selected_camera_ids=selected,
            primary_camera_id=primary,
            use_triangulation=bool(d.get("use_triangulation", True)),
            camera_rotations=camera_rotations,
            camera_baseline_m=baseline_m,
        )


def apply_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a frame by 0, 90, 180, or 270 degrees."""
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Reject mostly-black "idle" frames from virtual webcams (e.g. DroidCam when the
# phone app is not streaming). Those devices still open as DirectShow indices.
_PLACEHOLDER_GRAY_CAP = 18
_PLACEHOLDER_DARK_FRAC_ALMOST_ALL = 0.97
_PLACEHOLDER_DARK_FRAC_MOSTLY = 0.88
_PLACEHOLDER_GRAY_MEAN_MAX = 32


def is_inactive_virtual_cam_frame(bgr: np.ndarray) -> bool:
    """
    True if the frame looks like an idle / splash screen (dominant near-black),
    not a live scene. Used to skip DroidCam-style placeholders during detection.
    """
    if bgr is None or bgr.size == 0:
        return True
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g_mean = float(gray.mean())
    dark_frac = float(np.mean(gray < _PLACEHOLDER_GRAY_CAP))
    if dark_frac > _PLACEHOLDER_DARK_FRAC_ALMOST_ALL:
        return True
    if dark_frac > _PLACEHOLDER_DARK_FRAC_MOSTLY and g_mean < _PLACEHOLDER_GRAY_MEAN_MAX:
        return True
    return False


_PROBE_CODE = r"""
import cv2, json, sys, numpy as np
T, DF0, DF1, GM = 18, 0.97, 0.88, 32
def _dead(bgr):
    if bgr is None or bgr.size == 0:
        return True
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gm = float(g.mean())
    df = float(np.mean(g < T))
    return df > DF0 or (df > DF1 and gm < GM)
idx = int(sys.argv[1])
backends = []
for b in ('CAP_DSHOW', 'CAP_MSMF'):
    if hasattr(cv2, b):
        backends.append((b, getattr(cv2, b)))
if not backends:
    backends.append(('ANY', cv2.CAP_ANY))
for bname, be in backends:
    cap = cv2.VideoCapture(idx, be)
    if not cap.isOpened():
        cap.release()
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Read a few frames; first can be stale
    avg = 0.0
    ok = False
    for _ in range(4):
        ok, f = cap.read()
        if ok and f is not None:
            avg = float(f.mean())
    cap.release()
    if not ok:
        continue
    # Reject pure-green frames (DroidCam MSMF bug: R=0 G=136 B=0 -> avg~45)
    # and pure-black frames (scanner devices)
    if avg < 2.0:
        continue
    # Check for green-only pattern (all green channel, no red/blue)
    if ok and f is not None:
        r_ch, g_ch, b_ch = f[:,:,2].mean(), f[:,:,1].mean(), f[:,:,0].mean()
        if g_ch > 50 and r_ch < 5 and b_ch < 5:
            continue
    if ok and f is not None and _dead(f):
        continue
    print(json.dumps({"i": idx, "w": w, "h": h, "be": bname, "avg": avg}))
    sys.exit(0)
sys.exit(1)
"""


def _probe_single_index(index: int, timeout: float = 8.0) -> CameraInfo | None:
    """
    Probe a single camera index in a subprocess to avoid hanging the caller.
    Rejects black/green-only frames and mostly-black virtual-cam splash screens.
    """
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE, str(index)],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0 and r.stdout.strip():
            info = json.loads(r.stdout.strip())
            be = info.get("be", "")
            return CameraInfo(
                index=info["i"],
                label=f"Camera {info['i']} ({info['w']}x{info['h']}, {be})",
            )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        pass
    return None


def detect_connected_cameras(max_index: int = 10) -> list[CameraInfo]:
    """
    Probe OpenCV indices 0..max_index-1 in isolated subprocesses.
    Each index gets a timeout so a hung device can't block detection.
    Rejects devices that produce only black or green frames, or mostly-black
    virtual-cam splash screens (e.g. DroidCam when not streaming).
    """
    result: list[CameraInfo] = []
    for i in range(max_index):
        info = _probe_single_index(i)
        if info is not None:
            result.append(info)
    return result


# ---------------------------------------------------------------------------
# Calibrations load/save
# ---------------------------------------------------------------------------


def load_calibrations(path: Path | str | None = None) -> dict[str, Calibration]:
    """Load calibration data from JSON. Returns dict keyed by camera_id."""
    p = Path(path) if path is not None else CALIBRATIONS_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    # Expect {"cameras": { "id": {...}, ... }} or legacy list
    if "cameras" in raw:
        data = raw["cameras"]
    else:
        data = raw if isinstance(raw, dict) else {}
    return {k: Calibration.from_dict(v) for k, v in data.items()}


def save_calibrations(calibrations: dict[str, Calibration], path: Path | str | None = None) -> None:
    """Save calibration data to JSON."""
    p = Path(path) if path is not None else CALIBRATIONS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {cid: cal.to_dict() for cid, cal in calibrations.items()}
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"cameras": data}, f, indent=2)


# ---------------------------------------------------------------------------
# Last setup load/save
# ---------------------------------------------------------------------------


def load_last_setup(path: Path | str | None = None) -> LastCameraSetup | None:
    """Load last camera setup. Returns None if file missing or invalid."""
    p = Path(path) if path is not None else LAST_SETUP_PATH
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        return LastCameraSetup.from_dict(raw)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_last_setup(setup: LastCameraSetup, path: Path | str | None = None) -> None:
    """Save last camera setup to JSON."""
    p = Path(path) if path is not None else LAST_SETUP_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(setup.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Projection matrix helpers (for triangulation)
# ---------------------------------------------------------------------------


def get_projection_matrix(cal: Calibration) -> np.ndarray:
    """
    Return 3x4 projection matrix P = K [R | t] in world frame.
    If extrinsics missing, assumes identity (camera is world origin).
    """
    K = cal.intrinsics.K
    if cal.extrinsics is None:
        return np.hstack([K, np.zeros((3, 1))]).astype(np.float64)
    R = cal.extrinsics.R
    t = cal.extrinsics.t
    t = t.reshape(3, 1) if t.ndim == 1 else t
    Rt = np.hstack([R, t])
    return (K @ Rt).astype(np.float64)


def get_camera_center_world(cal: Calibration) -> np.ndarray | None:
    """Return the camera center in world coordinates, or None if unavailable."""
    if cal.extrinsics is None:
        return None
    R = np.array(cal.extrinsics.R, dtype=np.float64)
    t = np.array(cal.extrinsics.t, dtype=np.float64).reshape(3, 1)
    center = -R.T @ t
    return center[:, 0]


def get_camera_baseline_world(cal_a: Calibration, cal_b: Calibration) -> float | None:
    """Return the distance between two camera centers in calibration world units."""
    c_a = get_camera_center_world(cal_a)
    c_b = get_camera_center_world(cal_b)
    if c_a is None or c_b is None:
        return None
    return float(np.linalg.norm(c_a - c_b))
