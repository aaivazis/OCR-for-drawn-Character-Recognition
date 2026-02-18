from src.preprocessing.normalize import normalize_coordinates
from src.preprocessing.resampling import resampling
from src.preprocessing.smoothing import smoothing
import numpy as np
from typing import List, Tuple
import json


def open_json(json_name: str):
    """Open a JSON file saved from the GUI and return its strokes list."""
    with open(json_name) as f:
        data = json.load(f)

    strokes = data["strokes"]
    return strokes


def extract(strokes: List[List[Tuple[float, float, float]]]) -> List[np.ndarray]:
    """
    Run preprocessing (normalize -> resampling -> smoothing) per stroke.

    Args:
        strokes: list of strokes, each stroke is a list of (x, y, t) tuples

    Returns:
        List of numpy arrays, one per stroke, each of shape (N, 3)
        with columns [x_processed, y_processed, t_processed].
        Stroke order is preserved and strokes are not mixed.
    """
    processed_strokes: List[np.ndarray] = []
    
    #find global min, max points so normalization remains same for all strokes
    all_raw_points = np.vstack([np.array(s) for s in strokes if s])
    x_min, x_max = all_raw_points[:, 0].min(), all_raw_points[:, 0].max()
    y_min, y_max = all_raw_points[:, 1].min(), all_raw_points[:, 1].max()

    for stroke in strokes:
        # stroke: list of (x, y, t)
        if not stroke:
            continue

        stroke_points = np.array(stroke)
        x = stroke_points[:, 0]
        y = stroke_points[:, 1]
        t = stroke_points[:, 2]

        # Normalization, resampling, smoothing per stroke
        x_n, y_n, t_n = normalize_coordinates(x, y, t, x_min, x_max, y_min, y_max)
        x_s, y_s, t_s = resampling(x_n, y_n, t_n)
        x_sm, y_sm, t_sm = smoothing(x_s, y_s, t_s)

        # Stack back to (N, 3) array of x,y,t 
        processed_strokes.append(np.column_stack((x_sm, y_sm, t_sm)))

    return processed_strokes