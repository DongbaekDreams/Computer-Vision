"""Body segment profile persistence and estimation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from landmarks import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)

_PROJECT_DIR = Path(__file__).resolve().parent
BODY_PROFILE_PATH = _PROJECT_DIR / "body_profile.json"
DEFAULT_HEIGHT_M = 1.75

SEGMENT_RATIOS = {
    "shoulder_width": 0.259,
    "hip_width": 0.191,
    "torso": 0.288,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "thigh": 0.245,
    "shank": 0.246,
}


@dataclass
class BodyProfile:
    """User-entered body measurements stored in meters."""

    height_m: float | None = None
    segments_m: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "height_m": self.height_m,
            "segments_m": dict(self.segments_m),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BodyProfile":
        height_raw = raw.get("height_m")
        height_m = _positive_float(height_raw)

        segments_m: dict[str, float] = {}
        raw_segments = raw.get("segments_m", {})
        if isinstance(raw_segments, dict):
            for name, value in raw_segments.items():
                clean = _positive_float(value)
                if clean is not None:
                    segments_m[str(name)] = clean
        return cls(height_m=height_m, segments_m=segments_m)


@dataclass
class ResolvedBodyProfile:
    """Fully resolved profile with measured, live, or estimated values."""

    height_m: float
    height_source: str
    segments_m: dict[str, float]
    segment_sources: dict[str, str]


def _positive_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_f) or value_f <= 0.0:
        return None
    return value_f


def load_body_profile(path: Path | str | None = None) -> BodyProfile:
    """Load the stored body profile, or return an empty profile."""
    p = Path(path) if path is not None else BODY_PROFILE_PATH
    if not p.exists():
        return BodyProfile()
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return BodyProfile()
        return BodyProfile.from_dict(raw)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return BodyProfile()


def save_body_profile(profile: BodyProfile, path: Path | str | None = None) -> None:
    """Persist a body profile to disk."""
    p = Path(path) if path is not None else BODY_PROFILE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)


def estimate_height_m(profile: BodyProfile) -> tuple[float, str]:
    """Resolve a usable height estimate in meters."""
    height_m = _positive_float(profile.height_m)
    if height_m is not None:
        return height_m, "measured"

    inferred: list[float] = []
    for name, ratio in SEGMENT_RATIOS.items():
        seg = _positive_float(profile.segments_m.get(name))
        if seg is not None and ratio > 0.0:
            inferred.append(seg / ratio)
    if inferred:
        return float(np.mean(inferred)), "estimated from measurements"
    return DEFAULT_HEIGHT_M, "estimated default"


def _point_distance(pts_3d: np.ndarray | None, idx_a: int, idx_b: int) -> float | None:
    if pts_3d is None:
        return None
    if idx_a >= len(pts_3d) or idx_b >= len(pts_3d):
        return None
    a = np.asarray(pts_3d[idx_a], dtype=np.float64)
    b = np.asarray(pts_3d[idx_b], dtype=np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    dist = float(np.linalg.norm(a - b))
    return dist if np.isfinite(dist) and dist > 0.0 else None


def _mean_dist(pts_3d: np.ndarray | None, pairs: list[tuple[int, int]]) -> float | None:
    vals = [d for d in (_point_distance(pts_3d, a, b) for a, b in pairs) if d is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _live_segment_estimates(pts_3d: np.ndarray | None) -> dict[str, float]:
    if pts_3d is None:
        return {}
    mid_shoulder = _mean_dist(pts_3d, [(L_SHOULDER, R_SHOULDER)])
    mid_hip = _mean_dist(pts_3d, [(L_HIP, R_HIP)])

    torso = None
    if mid_shoulder is not None and mid_hip is not None:
        shoulder_center = (np.asarray(pts_3d[L_SHOULDER]) + np.asarray(pts_3d[R_SHOULDER])) * 0.5
        hip_center = (np.asarray(pts_3d[L_HIP]) + np.asarray(pts_3d[R_HIP])) * 0.5
        if np.all(np.isfinite(shoulder_center)) and np.all(np.isfinite(hip_center)):
            torso = float(np.linalg.norm(shoulder_center - hip_center))

    live: dict[str, float] = {}
    mapping = {
        "shoulder_width": mid_shoulder,
        "hip_width": mid_hip,
        "torso": torso,
        "upper_arm": _mean_dist(pts_3d, [(L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW)]),
        "forearm": _mean_dist(pts_3d, [(L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)]),
        "thigh": _mean_dist(pts_3d, [(L_HIP, L_KNEE), (R_HIP, R_KNEE)]),
        "shank": _mean_dist(pts_3d, [(L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE)]),
    }
    for name, value in mapping.items():
        clean = _positive_float(value)
        if clean is not None:
            live[name] = clean
    return live


def resolve_body_profile(
    profile: BodyProfile,
    pts_3d: np.ndarray | None = None,
) -> ResolvedBodyProfile:
    """Return measured, live, or estimated segment lengths in meters."""
    height_m, height_source = estimate_height_m(profile)
    estimated = {name: ratio * height_m for name, ratio in SEGMENT_RATIOS.items()}
    live = _live_segment_estimates(pts_3d)

    segments_m: dict[str, float] = {}
    segment_sources: dict[str, str] = {}
    for name in SEGMENT_RATIOS:
        measured = _positive_float(profile.segments_m.get(name))
        if measured is not None:
            segments_m[name] = measured
            segment_sources[name] = "measured"
            continue
        live_val = _positive_float(live.get(name))
        if live_val is not None:
            segments_m[name] = live_val
            segment_sources[name] = "live estimate"
            continue
        segments_m[name] = estimated[name]
        segment_sources[name] = "height estimate"

    return ResolvedBodyProfile(
        height_m=height_m,
        height_source=height_source,
        segments_m=segments_m,
        segment_sources=segment_sources,
    )
