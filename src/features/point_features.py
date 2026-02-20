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
    x = stroke[:, 0]
    y = stroke[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)

    angles = np.arctan2(dy, dx)
    angles = np.unwrap(angles) 
    d_angle = np.gradient(angles)

    speed = np.sqrt(dx*dx + dy*dy)

    return np.column_stack([x, y, dx, dy, d_angle, speed])

