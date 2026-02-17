from src.preprocessing.normalize import normalize_coordinates
from src.preprocessing.resampling import resampling
from src.preprocessing.smoothing import smoothing
from src.preprocessing.ft_extract import (extract, open_json)
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
import json

folder = Path("data/raw")

for file in folder.glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)
        strokes = data["strokes"]
        x, y, t = extract(strokes)

        print(f"Processed {file.name}")
        x_n, y_n, t_n = normalize_coordinates(x,y,t)
        x_s, y_s, t_s = resampling(x_n, y_n, t_n)
        x_sm, y_sm, t_sm = smoothing(x_s, y_s, t_s)
        
    
        plt.figure(figsize=(6,6))

        plt.subplot(2,2,1)
        plt.plot(x, y, 'o-', label="Raw", alpha=0.4)
        plt.title("Raw Character")

        plt.subplot(2,2,2)
        plt.plot(x_s, y_s, 'o-r', label="Resampled", alpha=0.7)
        plt.title("After Sampling")

        plt.subplot(2,2,3)
        plt.plot(x_sm, y_sm, 'o-g', label="Smoothed", linewidth=2)
        plt.title("After Smoothing")

        plt.legend()
        plt.show()


