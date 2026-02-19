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
    """
    Open a training CSV and return character labels and raw JSON strings.

    Args:
        csv_name: path to CSV (e.g. 'data/training/cap_a.csv').

    Returns:
        characters: list of characters (e.g. ['Α', 'Α', ...])
        json_strings: list of JSON strings from 'char_data' column.
    """
    df = pd.read_csv(csv_name)
    df = df.dropna(subset=["char_data"])

    characters = df["char"].tolist()
    json_strings = df["char_data"].tolist()

    return characters, json_strings


def extract_all_from_csv(csv_name: str):
    """
    High-level helper:
    - uses extract_from_csv to read the CSV,
    - parses the embedded JSON for each row,
    - extracts the 'strokes' list for each instance,
    - feeds those strokes into extract() using the same input type.

    Args:
        csv_name: path to training CSV.

    Returns:
        List of tuples (char, processed_strokes) where:
        - char is the character label from the CSV,
        - processed_strokes is the list returned by extract(strokes)
            for that character (list of per-stroke arrays).
    """
    characters, json_strings = extract_from_csv(csv_name)

    results = []
    for char, json_str in zip(characters, json_strings):
        data = json.loads(json_str)
        strokes = data["strokes"]  # same structure as GUI JSON: List[List[(x,y,t)]]
        processed_strokes = extract(strokes)
        results.append((char, processed_strokes))

    return results







