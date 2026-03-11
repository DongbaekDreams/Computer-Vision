"""Camera calibration storage and detection for multi-camera pose dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    """Detected camera: index and optional label from last setup."""
    index: int
    label: str = ""


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


@dataclass
class LastCameraSetup:
    """Last-used camera selection for the dashboard."""
    selected_camera_ids: list[str]
    primary_camera_id: str
    use_triangulation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_camera_ids": list(self.selected_camera_ids),
            "primary_camera_id": self.primary_camera_id,
            "use_triangulation": self.use_triangulation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LastCameraSetup:
        return cls(
            selected_camera_ids=list(d.get("selected_camera_ids", [])),
            primary_camera_id=str(d.get("primary_camera_id", "")),
            use_triangulation=bool(d.get("use_triangulation", True)),
        )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_connected_cameras(max_index: int = 10) -> list[CameraInfo]:
    """
    Probe OpenCV indices 0..max_index-1 and return list of cameras that open.
    Uses CAP_DSHOW on Windows for more reliable indexing.
    """
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    result: list[CameraInfo] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            result.append(CameraInfo(index=i, label=""))
            cap.release()
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
