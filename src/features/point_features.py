from __future__ import annotations

import numpy as np


def xyt_features(stroke: np.ndarray) -> np.ndarray:
    """
    Baseline features: use raw (x, y, t).

    Args:
        stroke: numpy array shape (N, 3) with columns [x, y, t]

    Returns:
        numpy array shape (N, 3)
    """
    return stroke[:, :3]


def xy_features(stroke: np.ndarray) -> np.ndarray:
    """
    Features using only (x, y) (time is ignored).

    Args:
        stroke: numpy array shape (N, 3) with columns [x, y, t]

    Returns:
        numpy array shape (N, 2)
    """
    return stroke[:, :2]


def point_features(stroke: np.ndarray) -> np.ndarray:
    """
    Slightly richer per-point features that preserve ordering:
    (x, y, dx, dy, d_angle)

    This often clusters handwriting trajectories better than raw (x,y,t).

    Args:
        stroke: numpy array shape (N, 3) with columns [x, y, t]

    Returns:
        numpy array shape (N, 5)
    """
    x = stroke[:, 0]
    y = stroke[:, 1]

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    angles = np.arctan2(dy, dx)
    d_angle = np.diff(angles, prepend=angles[0])

    return np.column_stack([x, y, dx, dy, d_angle])


