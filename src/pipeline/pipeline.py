from src.algorithms.kmeans import (discretize_character, build_codebook)
from src.preprocessing.ft_extract import extract_all_from_csv
from src.features.point_features import point_features
import numpy as np
import pandas as pd


def pipeline_to_discrete(csv:str, K=12):
    SEP = K
    results = extract_all_from_csv(csv)
    
    model = build_codebook(results, n_clusters=K, feature_fn=point_features, random_state=0)  
    discrete_dataset = []

    for char, processed_strokes in results:
        disc = discretize_character(
        processed_strokes,
        model,
        feature_fn=point_features,
        sep_token=SEP
        )
        discrete_dataset.append((char, disc))

    return discrete_dataset