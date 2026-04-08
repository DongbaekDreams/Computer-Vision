"""Low-overhead adaptive smoothing filters for landmarks."""

from __future__ import annotations

import math

import numpy as np


def _alpha_from_cutoff(dt_s: float, cutoff_hz: float) -> float:
    """Return EMA alpha from dt and cutoff frequency."""
    if dt_s <= 0.0 or cutoff_hz <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + (tau / dt_s))


class OneEuroFilter1D:
    """One Euro filter for scalar streams."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.03, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def update(self, x: float, t_s: float) -> float:
        x = float(x)
        t_s = float(t_s)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t_s
            self._dx_prev = 0.0
            return x

        dt = max(1e-6, t_s - self._t_prev)
        dx = (x - self._x_prev) / dt
        a_d = _alpha_from_cutoff(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a_x = _alpha_from_cutoff(dt, cutoff)
        x_hat = a_x * x + (1.0 - a_x) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t_s
        return x_hat


class LandmarkArrayFilter:
    """
    One Euro filter for landmark arrays.

    - Keeps one filter per [landmark, dimension].
    - update_mask lets us skip invalid points and preserve prior state.
    """

    def __init__(self, n_landmarks: int, dims: int, min_cutoff: float, beta: float, d_cutoff: float):
        self.n_landmarks = int(n_landmarks)
        self.dims = int(dims)
        self._filters = [
            [OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(self.dims)]
            for _ in range(self.n_landmarks)
        ]

    def reset_all(self) -> None:
        for row in self._filters:
            for f in row:
                f.reset()

    def reset_landmark(self, idx: int) -> None:
        if 0 <= idx < self.n_landmarks:
            for f in self._filters[idx]:
                f.reset()

    def update(
        self,
        values: np.ndarray,
        t_s: float,
        update_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != (self.n_landmarks, self.dims):
            raise ValueError(
                f"Expected shape {(self.n_landmarks, self.dims)}, got {tuple(arr.shape)}"
            )
        out = arr.copy()
        if update_mask is None:
            update_mask = np.ones((self.n_landmarks,), dtype=bool)
        else:
            update_mask = np.asarray(update_mask, dtype=bool).reshape(self.n_landmarks)

        for i in range(self.n_landmarks):
            if not update_mask[i]:
                continue
            for d in range(self.dims):
                v = float(arr[i, d])
                if not np.isfinite(v):
                    break
                out[i, d] = self._filters[i][d].update(v, t_s)
        return out
