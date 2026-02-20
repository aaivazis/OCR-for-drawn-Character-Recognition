from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


FeatureFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DiscretizationResult:
    """
    Holds discretized sequences for one character.

    - stroke_codes: list of 1D arrays; each array is the sequence of cluster IDs
    for that stroke, preserving point order.
    - flat_codes: optional single 1D array with separator tokens inserted between strokes.
    """

    stroke_codes: List[np.ndarray]
    flat_codes: Optional[np.ndarray] = None
    
@dataclass(frozen=True)
class Codebook:
    scaler: StandardScaler
    kmeans: KMeans


def _iter_stroke_features(
    results: Sequence[Tuple[object, Sequence[np.ndarray]]],
    feature_fn: FeatureFn,
) -> Iterable[np.ndarray]:
    """Yield feature arrays for every stroke in every (char, strokes) item."""
    for _char, strokes in results:
        for stroke in strokes:
            if stroke is None or len(stroke) == 0:
                continue
            feats = feature_fn(stroke)
            if feats is None or len(feats) == 0:
                continue
            yield feats


def build_codebook(
    results: Sequence[Tuple[object, Sequence[np.ndarray]]],
    n_clusters: int = 64,
    feature_fn: Optional[FeatureFn] = None,
    random_state: int = 0,
) -> Pipeline:
    """
    Fit a single codebook (StandardScaler + KMeans) over all points.
    Returns a sklearn Pipeline so scaling is applied consistently at predict time too.
    """
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


def discretize_strokes(
    strokes: Sequence[np.ndarray],
    model: Pipeline,
    feature_fn: Optional[FeatureFn] = None,
) -> List[np.ndarray]:
    """
    Discretize each stroke into an ordered sequence of cluster IDs
    using the same scaling + kmeans model.
    """
    if feature_fn is None:
        feature_fn = lambda s: s[:, :3]

    stroke_codes: List[np.ndarray] = []
    for stroke in strokes:
        if stroke is None or len(stroke) == 0:
            continue
        X = feature_fn(stroke)
        codes = model.predict(X)  # scaling happens inside the pipeline
        stroke_codes.append(codes.astype(np.int32, copy=False))
    return stroke_codes


def flatten_with_separators(
    stroke_codes: Sequence[np.ndarray],
    sep_token: int,
) -> np.ndarray:
    """
    Concatenate stroke code sequences into one sequence and insert sep_token between strokes.
    """
    if not stroke_codes:
        return np.array([], dtype=np.int32)

    parts: List[np.ndarray] = []
    for i, seq in enumerate(stroke_codes):
        parts.append(seq)
        if i < len(stroke_codes) - 1:
            parts.append(np.array([sep_token], dtype=seq.dtype))
    return np.concatenate(parts)


def discretize_character(
    strokes: Sequence[np.ndarray],
    model: Pipeline,
    feature_fn: Optional[FeatureFn] = None,
    sep_token: Optional[int] = None,
) -> DiscretizationResult:
    stroke_codes = discretize_strokes(strokes, model, feature_fn=feature_fn)
    flat_codes = None
    if sep_token is not None:
        flat_codes = flatten_with_separators(stroke_codes, sep_token=sep_token)
    return DiscretizationResult(stroke_codes=stroke_codes, flat_codes=flat_codes)

