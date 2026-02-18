import numpy as np
from typing import List, Tuple
#fetch ta JSON
#access ta strokes 
#normalize me tupous


def normalize_coordinates(x, y, t, x_min, x_max, y_min, y_max):
    #avoid division with 0
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    
    return x_norm, y_norm, t