from src.preprocessing.ft_extract import extract
import matplotlib.pyplot as plt
from pathlib import Path
import json


folder = Path("data/raw")

for file in folder.glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)
        strokes = data["strokes"]

    processed_strokes = extract(strokes)

    # New figure per character
    plt.figure(figsize=(6, 6))

    # Plot each stroke separately, preserving order
    for idx, stroke_arr in enumerate(processed_strokes):
        if stroke_arr.shape[0] == 0:
            continue
        xp = stroke_arr[:, 0]
        yp = stroke_arr[:, 1]
        plt.plot(xp, yp, "o-", label=f"stroke {idx}", alpha=0.7)

    plt.title(f"Char from {file.name}")
    plt.axis("equal")
    plt.legend()
    plt.show()
