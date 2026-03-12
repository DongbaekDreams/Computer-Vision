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
    """Detected camera: index (int) or URL source (str), plus optional label."""
    index: int  # -1 for URL-based cameras
    label: str = ""
    source: str = ""  # URL string for IP cameras; empty for index-based


def is_url_source(camera_id: str) -> bool:
    """True if camera_id is a URL rather than an integer index."""
    return camera_id.startswith("http://") or camera_id.startswith("https://") or camera_id.startswith("rtsp://")


_INDEX_BACKENDS = []
for _b in ("CAP_DSHOW", "CAP_MSMF"):
    if hasattr(cv2, _b):
        _INDEX_BACKENDS.append(getattr(cv2, _b))
if not _INDEX_BACKENDS:
    _INDEX_BACKENDS.append(cv2.CAP_ANY)


def open_camera(camera_id: str, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """
    Open a VideoCapture for a camera_id that is either an integer index
    (as a string like "0") or a URL ("http://...", "rtsp://...").
    For URL cameras, minimizes internal buffering to reduce latency.
    For index cameras, tries DSHOW first (more reliable for real hardware),
    then falls back to MSMF.
    """
    if is_url_source(camera_id):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        idx = int(camera_id)
        cap = cv2.VideoCapture()
        for backend in _INDEX_BACKENDS:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                break
            cap.release()
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def drain_stale_frames(cap: cv2.VideoCapture, max_drain: int = 3) -> None:
    """
    Grab (but don't decode) up to max_drain queued frames from an IP camera
    to get closer to the live edge. Call before the "real" read().
    """
    for _ in range(max_drain):
        cap.grab()


import threading


class ThreadedCapture:
    """
    Continuously reads frames from a VideoCapture in a background thread.
    The main thread can call latest() at any time to get the most recent
    frame without blocking. Keeps only the newest frame (no queue buildup).
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_camera_ids": list(self.selected_camera_ids),
            "primary_camera_id": self.primary_camera_id,
            "use_triangulation": self.use_triangulation,
            "camera_rotations": dict(self.camera_rotations),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LastCameraSetup:
        return cls(
            selected_camera_ids=list(d.get("selected_camera_ids", [])),
            primary_camera_id=str(d.get("primary_camera_id", "")),
            use_triangulation=bool(d.get("use_triangulation", True)),
            camera_rotations=dict(d.get("camera_rotations", {})),
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


_PROBE_CODE = r"""
import cv2, json, sys, time
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
    print(json.dumps({"i": idx, "w": w, "h": h, "be": bname, "avg": avg}))
    sys.exit(0)
sys.exit(1)
"""


def _probe_single_index(index: int, timeout: float = 8.0) -> CameraInfo | None:
    """
    Probe a single camera index in a subprocess to avoid hanging the caller.
    Rejects devices that only produce black or green frames.
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


def detect_connected_cameras(max_index: int = 5) -> list[CameraInfo]:
    """
    Probe OpenCV indices 0..max_index-1 in isolated subprocesses.
    Each index gets a timeout so a hung device can't block detection.
    Rejects devices that produce only black or green frames.
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
