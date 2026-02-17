from src.preprocessing.normalize import normalize_coordinates
from src.preprocessing.resampling import resampling
from src.preprocessing.smoothing import smoothing
import numpy as np
from typing import List, Tuple
import json

def open_json(json_name):
    with open(json_name) as f:
        data = json.load(f)

    strokes = data["strokes"]
    return strokes


def extract(strokes:List[List[Tuple[float, float, float]]]):
    all_points = np.array([point for stroke in strokes for point in stroke])
    x,y,t = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    
    return x, y, t