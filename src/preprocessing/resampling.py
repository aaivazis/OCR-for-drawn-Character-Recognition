import numpy as np
from typing import List

def resampling(strokes: List[np.ndarray], N_total: int = 64) -> List[np.ndarray]:
    stroke_lengths = []
    for s in strokes:
        if len(s) < 2:
            stroke_lengths.append(0.0)
        else:
            dx = np.diff(s[:, 0])
            dy = np.diff(s[:, 1])
            stroke_lengths.append(np.sum(np.sqrt(dx**2 + dy**2)))
    
    total_char_length = sum(stroke_lengths)
    resampled_list = []
    
    if total_char_length == 0:
        n_per_stroke = N_total // len(strokes)
        for s in strokes:
            resampled_list.append(np.full((n_per_stroke, 3), s[0]))
        return resampled_list

    points_per_stroke = []
    for length in stroke_lengths:
        n = max(2, int(round((length / total_char_length) * N_total)))
        points_per_stroke.append(n)
        
    diff = N_total - sum(points_per_stroke)
    if diff != 0:
        points_per_stroke[np.argmax(stroke_lengths)] += diff

    for i, stroke in enumerate(strokes):
        n_samples = points_per_stroke[i]
        x, y, t = stroke[:, 0], stroke[:, 1], stroke[:, 2]
        
        dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        s_cumulative = np.concatenate(([0.0], np.cumsum(dists)))
        total_s = s_cumulative[-1]
        
        s_new = np.linspace(0.0, total_s, n_samples)
        
        x_new = np.interp(s_new, s_cumulative, x)
        y_new = np.interp(s_new, s_cumulative, y)
        t_new = np.interp(s_new, s_cumulative, t)
        
        resampled_list.append(np.column_stack((x_new, y_new, t_new)))
        
    return resampled_list