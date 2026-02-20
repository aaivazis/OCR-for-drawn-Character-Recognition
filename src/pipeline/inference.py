from __future__ import annotations
import json
from typing import List, Tuple
import numpy as np
from src.algorithms.kmeans import discretize_character
from src.features.point_features import point_features
from src.models.hmm_models import load_global_codebook, load_hmms_for_inference
from src.preprocessing.ft_extract import extract

#inference pipeline that takes a json path and returns the top n scores
def pipeline_raw_json_to_top_scores(
    json_path: str,
    parameters_dir: str = "parameters",
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    #opens the json file and loads the data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    strokes = data.get("strokes", [])
    if not strokes:
        return []

    processed_strokes = extract(strokes)

    codebook = load_global_codebook(parameters_dir=parameters_dir)
    hmms = load_hmms_for_inference(parameters_dir=parameters_dir)
    if not hmms:
        return []

    #gets the separator token
    try:
        sep_token = int(codebook.named_steps["kmeans"].n_clusters)
    except Exception:
        sep_token = 12

    disc = discretize_character(
        processed_strokes,
        codebook,
        feature_fn=point_features,
        sep_token=sep_token,
    )
    if disc.flat_codes is None or len(disc.flat_codes) == 0:
        return []

    #reshapes the flat codes into a numpy array
    X = disc.flat_codes.astype(np.int32, copy=False).reshape(-1, 1)

    #list to store the scores
    scores: List[Tuple[str, float]] = []
    #scores the character against the hmm model
    for char, hmm_model in hmms.items():
        score = float(hmm_model.score(X))
        scores.append((char, score))

    #sorts the scores in descending order
    scores.sort(key=lambda item: item[1], reverse=True)
    #returns the top n scores
    return scores[:top_n]

