import numpy as np
from scipy.signal import savgol_filter

#use of Savgol filter to smoothe rough edges, better result in curvature estimation
def smoothing(x: np.ndarray, y: np.ndarray, t: np.ndarray, window_length: int = 7, polyorder: int = 2):
    n = len(x)
    if n < 3:
        return x, y, t

    # window must be odd and <= n
    w = min(window_length, n if n % 2 == 1 else n - 1)
    if w < 3:
        return x, y, t

    p = min(polyorder, w - 1)
    x_s = savgol_filter(x, window_length=w, polyorder=p)
    y_s = savgol_filter(y, window_length=w, polyorder=p)

    return x_s, y_s, t