from typing import List
import numpy as np
from scipy.signal import savgol_filter

#use of Savgol filter to smoothe rough edges
def smoothing(strokes: List[np.array], window_length: int = 7, polyorder: int = 2)->List[np.array]:
    smoothed_list = []

    for stroke in strokes:
        n_points = len(stroke)
        if n_points < window_length:
            dynamic_window = n_points if n_points % 2 == 1 else n_points - 1
            if dynamic_window < 3:
                smoothed_list.append(stroke)
                continue
            w = dynamic_window
        else:
            w = window_length

        x_sm = savgol_filter(stroke[:, 0], window_length=w, polyorder=polyorder)
        y_sm = savgol_filter(stroke[:, 1], window_length=w, polyorder=polyorder)
        t_sm = stroke[:, 2] 

        smoothed_list.append(np.column_stack((x_sm, y_sm, t_sm)))

    return smoothed_list