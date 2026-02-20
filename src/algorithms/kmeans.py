from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#callable that takes a numpy array and returns a numpy array
FeatureFn = Callable[[np.ndarray], np.ndarray]

#dataclass used to return two forms of the same data
#- stroke_codes: list of 1D arrays; each array is the sequence of cluster IDs
# for that stroke, preserving point order.
# - flat_codes: optional single 1D array with separator tokens inserted between strokes.
@dataclass(frozen=True)
class DiscretizationResult:
    stroke_codes: List[np.ndarray]
    flat_codes: Optional[np.ndarray] = None

#dataclass used to return the scaler and kmeans model
@dataclass(frozen=True)
class Codebook:
    scaler: StandardScaler
    kmeans: KMeans

#iterates through the strokes and yields the features
def _iter_stroke_features(
    results: Sequence[Tuple[object, Sequence[np.ndarray]]],
    feature_fn: FeatureFn,
) -> Iterable[np.ndarray]:
    for _char, strokes in results:
        for stroke in strokes:
            if stroke is None or len(stroke) == 0:
                continue
            feats = feature_fn(stroke)
            if feats is None or len(feats) == 0:
                continue
            yield feats

#builds the codebook
def build_codebook(
    results: Sequence[Tuple[object, Sequence[np.ndarray]]],
    n_clusters: int = 64,
    feature_fn: Optional[FeatureFn] = None,
    random_state: int = 0,
) -> Pipeline:
    #if no feature function is provided, use the default one
    if feature_fn is None:
        feature_fn = lambda s: s[:, :3]

    feats_list = list(_iter_stroke_features(results, feature_fn))
    if not feats_list:
        raise ValueError("No stroke data found to build KMeans codebook.")

    X = np.vstack(feats_list)

    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
    )
    model.fit(X)
    return model

#discretizes the strokes
def discretize_strokes(
    strokes: Sequence[np.ndarray],
    model: Pipeline,
    feature_fn: Optional[FeatureFn] = None,
) -> List[np.ndarray]:
    #if no feature function is provided, use the default one
    if feature_fn is None:
        feature_fn = lambda s: s[:, :3]

    #list to store the stroke codes
    stroke_codes: List[np.ndarray] = []
    for stroke in strokes:
        if stroke is None or len(stroke) == 0:
            continue
        X = feature_fn(stroke)
        codes = model.predict(X)  # scaling happens inside the pipeline
        stroke_codes.append(codes.astype(np.int32, copy=False))
    return stroke_codes

#flattens the stroke codes with separators
def flatten_with_separators(
    stroke_codes: Sequence[np.ndarray],
    sep_token: int,
) -> np.ndarray:
    #if no stroke codes are provided, return an empty array
    if not stroke_codes:
        return np.array([], dtype=np.int32)

    parts: List[np.ndarray] = []
    for i, seq in enumerate(stroke_codes):
        parts.append(seq)
        if i < len(stroke_codes) - 1:
            parts.append(np.array([sep_token], dtype=seq.dtype))
    return np.concatenate(parts)

#discretizes the character
def discretize_character(
    strokes: Sequence[np.ndarray],
    model: Pipeline,
    feature_fn: Optional[FeatureFn] = None,
    sep_token: Optional[int] = None,
) -> DiscretizationResult:
    #discretizes the strokes
    stroke_codes = discretize_strokes(strokes, model, feature_fn=feature_fn)
    #if a separator token is provided, flatten the stroke codes with separators
    flat_codes = None
    if sep_token is not None:
        flat_codes = flatten_with_separators(stroke_codes, sep_token=sep_token)
    return DiscretizationResult(stroke_codes=stroke_codes, flat_codes=flat_codes)

