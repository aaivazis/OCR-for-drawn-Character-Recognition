from src.preprocessing.normalize import normalize_coordinates
from src.preprocessing.resampling import resampling
from src.preprocessing.smoothing import smoothing
import numpy as np
import pandas as pd
from typing import List, Tuple
import json


def extract(strokes: List[List[Tuple[float, float, float]]]) -> List[np.ndarray]:
    """Run full preprocessing pipeline for a single character's strokes."""
    normalized_strokes = normalize_coordinates(strokes)
    sampled_strokes = resampling(normalized_strokes, N_total=64)
    smoothed_strokes = smoothing(sampled_strokes)
    return smoothed_strokes


def extract_from_csv(csv_name: str):
    #extracts JSONs from csv, converts them to a dataframe
    df = pd.read_csv(csv_name)
    df = df.dropna(subset=["char_data"])

    characters = df["char"].tolist()
    json_strings = df["char_data"].tolist()

    return characters, json_strings


def extract_all_from_csv(csv_name: str):
    #extracts all strokes of all letters from the dataframe
    characters, json_strings = extract_from_csv(csv_name)

    results = []
    for char, json_str in zip(characters, json_strings):
        data = json.loads(json_str)
        strokes = data["strokes"]  # same structure as GUI JSON: List[List[(x,y,t)]]
        processed_strokes = extract(strokes)
        results.append((char, processed_strokes))

    return results







