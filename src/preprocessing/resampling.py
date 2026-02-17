import numpy as np
from typing import List, Tuple

def resampling(x, y, t, N=64):
    #arr = np.array([point for stroke in strokes for point in stroke])
    
    if len(x)==0 or len(y)==0:
        return x,y,t
    
    if len(x)==1 or len(y)==1 or N<=1:
        return np.full(N, x[0]), np.full(N, y[0]), np.linspace(t[0], t[0], N)
        
    L = stroke_length(x, y)
    Delta = L / (N-1)
    
    s = np.concatenate(([0.0], np.cumsum(L)))
    total = s[-1]

    if total == 0.0:
        return np.full(N, x[0]), np.full(N, y[0]), np.linspace(t[0], t[-1], N)    
    
    s_new = s_new = np.linspace(0.0, total, N)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    t_new = np.interp(s_new, s, t)
    
    return x_new, y_new, t_new
        


def stroke_length(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    
    return np.sqrt(dx**2 + dy**2)
