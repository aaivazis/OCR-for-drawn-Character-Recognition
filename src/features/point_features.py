from __future__ import annotations
import numpy as np

#returns all variables x,y,t of the array into a numpy array
def xyt_features(stroke: np.ndarray) -> np.ndarray:
    
    return stroke[:, :3]

#use if time is not needed
def xy_features(stroke: np.ndarray) -> np.ndarray:
    return stroke[:, :2]

#use when angle and speed of stroke are needed
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

