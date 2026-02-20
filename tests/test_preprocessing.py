from src.preprocessing.ft_extract import extract_all_from_csv
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import math

def plot_preprocessed_characters(results, num_samples=9, cols=3):
    samples_to_plot = results[:num_samples]
    n = len(samples_to_plot)
    
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if n > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i in range(rows * cols):
        ax = axes_flat[i]
        
        if i < n:
            char, strokes = samples_to_plot[i]
            
            for stroke in strokes:
                # stroke: [x, y, t]
                x = stroke[:, 0]
                y = stroke[:, 1]
                ax.plot(x, y, marker='o', markersize=2, linestyle='-', linewidth=1)
            
            ax.set_title(f"Sample {i+1}: {char}")
            ax.set_aspect('equal')
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.invert_yaxis()
            ax.grid(True, linestyle='--', alpha=0.3)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


    

results = extract_all_from_csv("data/training/cap_a_cleaned.csv")
plot_preprocessed_characters(results, num_samples=60, cols=6)


