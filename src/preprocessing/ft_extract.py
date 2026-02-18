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
    normalized_strokes = normalize_coordinates(strokes)
    sampled_strokes = resampling(normalized_strokes, N_total = 64)
    smoothed_strokes = smoothing(sampled_strokes)
    
    return smoothed_strokes