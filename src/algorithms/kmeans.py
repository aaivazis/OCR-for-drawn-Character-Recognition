from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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
) -> KMeans:
    """
    Fit a single KMeans codebook over all points (across all characters).

    Args:
        results: output of extract_all_from_csv: List[(char, processed_strokes)]
                where processed_strokes is List[np.ndarray] of shape (Ni, 3).
        n_clusters: number of discrete symbols to learn.
        feature_fn: maps a stroke (Ni,3) -> features (Ni,F). Defaults to x,y,t.
        random_state: for reproducibility.
    """
    if feature_fn is None:
        feature_fn = lambda s: s[:, :3]

    feats_list = list(_iter_stroke_features(results, feature_fn))
    if not feats_list:
        raise ValueError("No stroke data found to build KMeans codebook.")

    X = np.vstack(feats_list)

    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    km.fit(X)
    return km


def discretize_strokes(
    strokes: Sequence[np.ndarray],
    kmeans: KMeans,
    feature_fn: Optional[FeatureFn] = None,
) -> List[np.ndarray]:
    """
    Discretize each stroke into an ordered sequence of cluster IDs.
    """
    if feature_fn is None:
        feature_fn = lambda s: s[:, :3]

    stroke_codes: List[np.ndarray] = []
    for stroke in strokes:
        if stroke is None or len(stroke) == 0:
            continue
        X = feature_fn(stroke)
        codes = kmeans.predict(X)
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
    kmeans: KMeans,
    feature_fn: Optional[FeatureFn] = None,
    sep_token: Optional[int] = None,
) -> DiscretizationResult:
    """
    Discretize a single character's strokes, optionally returning a flat sequence with separators.
    """
    stroke_codes = discretize_strokes(strokes, kmeans, feature_fn=feature_fn)
    flat_codes = None
    if sep_token is not None:
        flat_codes = flatten_with_separators(stroke_codes, sep_token=sep_token)
    return DiscretizationResult(stroke_codes=stroke_codes, flat_codes=flat_codes)

