import numpy as np
from typing import List, Tuple
#fetch ta JSON
#access ta strokes 
#normalize me tupous


def normalize_coordinates(strokes:List[List[Tuple[float, float, float]]]) -> List[np.ndarray]:
    #find range to avoid division with 0
    normalized_strokes = []
    
    all_raw_points = np.vstack([np.array(s) for s in strokes if s])
    x_min, x_max = all_raw_points[:, 0].min(), all_raw_points[:, 0].max()
    y_min, y_max = all_raw_points[:, 1].min(), all_raw_points[:, 1].max()
    
    for stroke in strokes: 
        stroke_points = np.array(stroke)
        x = stroke_points[:, 0]
        y = stroke_points[:, 1]
        t = stroke_points[:, 2]
        
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
    
        x_norm = (x - x_min) / x_range
        y_norm = (y - y_min) / y_range
        
        normalized_strokes.append(np.column_stack((x_norm, y_norm, t)))
        
    return normalized_strokes