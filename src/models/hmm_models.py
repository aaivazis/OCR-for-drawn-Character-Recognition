from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from hmmlearn.hmm import CategoricalHMM
from sklearn.pipeline import Pipeline

from src.algorithms.kmeans import DiscretizationResult, build_codebook, discretize_character
from src.features.point_features import point_features
from src.preprocessing.ft_extract import extract_all_from_csv
from src.pipeline.pipeline import pipeline_to_discrete_with_codebook


#dataclass that stores the codebook and hmm model
@dataclass(frozen=True)
class CharModel:
    codebook: Pipeline
    hmm: CategoricalHMM


#converts the character to a filename key
def _char_to_filename_key(char: str) -> str:
    codepoint = f"U{ord(char):04X}"
    return f"{codepoint}_{char}"


#extracts the hmm parameters
def extract_hmm_parameters(hmm_model: CategoricalHMM) -> Dict[str, np.ndarray | int]:
    #returns the hmm parameters
    return {
        "n_components": int(hmm_model.n_components),
        "n_features": int(hmm_model.n_features),
        "startprob_": np.asarray(hmm_model.startprob_),
        "transmat_": np.asarray(hmm_model.transmat_),
        "emissionprob_": np.asarray(hmm_model.emissionprob_),
    }

#saves the hmm parameters
def save_hmm_parameters(
    models: Mapping[str, CharModel],
    parameters_dir: str = "parameters",
) -> Dict[str, str]:
    #creates the output directory
    out_dir = Path(parameters_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #list to store the saved paths
    saved_paths: Dict[str, str] = {}
    #saves the hmm parameters for each character
    for char, char_model in models.items():
        out_path = out_dir / f"hmm_{_char_to_filename_key(char)}.joblib"
        payload = extract_hmm_parameters(char_model.hmm)
        payload["char"] = char
        joblib.dump(payload, out_path)
        saved_paths[char] = str(out_path)

    return saved_paths

#saves the global codebook
def save_global_codebook(
    codebook: Pipeline,
    parameters_dir: str = "parameters",
) -> str:
    out_dir = Path(parameters_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "global_codebook.joblib"
    joblib.dump(codebook, out_path)
    return str(out_path)

#loads the global codebook
def load_global_codebook(parameters_dir: str = "parameters") -> Pipeline:
    codebook_path = Path(parameters_dir) / "global_codebook.joblib"
    if not codebook_path.exists():
        raise FileNotFoundError(f"Global codebook not found: {codebook_path}")
    return joblib.load(codebook_path)

#loads the hmm parameters
def load_hmm_parameters(parameters_dir: str = "parameters") -> Dict[str, Dict[str, np.ndarray | int | str]]:
    #creates the input directory
    in_dir = Path(parameters_dir)
    if not in_dir.exists():
        return {}
    #list to store the loaded parameters
    loaded: Dict[str, Dict[str, np.ndarray | int | str]] = {}
    for file_path in sorted(in_dir.glob("hmm_*.joblib")):
        payload = joblib.load(file_path)
        char = payload.get("char")
        if not isinstance(char, str):
            continue
        loaded[char] = payload

    return loaded

#loads the hmm parameters
def hmm_from_parameters(payload: Mapping[str, np.ndarray | int | str]) -> CategoricalHMM:
    model = CategoricalHMM(
        n_components=int(payload["n_components"]),
        n_features=int(payload["n_features"]),
        init_params="",
        params="",
    )
    model.startprob_ = np.asarray(payload["startprob_"], dtype=float)
    model.transmat_ = np.asarray(payload["transmat_"], dtype=float)
    model.emissionprob_ = np.asarray(payload["emissionprob_"], dtype=float)
    return model

#loads the hmms for inference
def load_hmms_for_inference(parameters_dir: str = "parameters") -> Dict[str, CategoricalHMM]:
    params_by_char = load_hmm_parameters(parameters_dir=parameters_dir)
    return {char: hmm_from_parameters(payload) for char, payload in params_by_char.items()}

#converts the discrete dataset to hmm training data
def _to_hmm_training_data(
    discrete_dataset: Iterable[Tuple[str, DiscretizationResult]],
) -> Tuple[np.ndarray, List[int]]:
    sequences: List[np.ndarray] = []
    lengths: List[int] = []

    for _char, disc in discrete_dataset:
        if disc.flat_codes is None or len(disc.flat_codes) == 0:
            continue
        seq = disc.flat_codes.astype(np.int32, copy=False).reshape(-1, 1)
        sequences.append(seq)
        lengths.append(seq.shape[0])

    if not sequences:
        raise ValueError("No non-empty flat_codes were produced for HMM training.")

    X = np.vstack(sequences)
    return X, lengths

#trains a character model from a csv file
def train_char_model_from_csv(
    csv_path: str,
    K: int = 12,
    S: int = 6,
    n_iter: int = 100,
    tol: float = 1e-2,
    random_state: int = 0,
) -> CharModel:
    #discretizes the dataset
    discrete_dataset, codebook = pipeline_to_discrete_with_codebook(csv_path, K=K)
    #converts the discrete dataset to hmm training data
    X, lengths = _to_hmm_training_data(discrete_dataset)
    
    #trains the hmm model
    hmm_model = CategoricalHMM(
        n_components=S,
        n_features=K + 1,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )
    hmm_model.fit(X, lengths)

    return CharModel(codebook=codebook, hmm=hmm_model)

#trains a character model from a discrete dataset
def train_char_model_from_discrete_dataset(
    discrete_dataset: Sequence[Tuple[str, DiscretizationResult]],
    codebook: Pipeline,
    K: int = 12,
    S: int = 6,
    n_iter: int = 100,
    tol: float = 1e-2,
    random_state: int = 0,
) -> CharModel:
    #converts the discrete dataset to hmm training data
    X, lengths = _to_hmm_training_data(discrete_dataset)
    #trains the hmm model
    hmm_model = CategoricalHMM(
        n_components=S,
        n_features=K + 1,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )
    hmm_model.fit(X, lengths)
    return CharModel(codebook=codebook, hmm=hmm_model)

#trains a character model from a csv map
def train_models_from_csv_map(
    csv_by_char: Mapping[str, str],
    K: int = 12,
    S: int = 6,
    n_iter: int = 100,
    tol: float = 1e-2,
    random_state: int = 0,
    strict_files: bool = True,
    persist_parameters: bool = True,
    parameters_dir: str = "parameters",
) -> Dict[str, CharModel]:
    #list to store the models
    models: Dict[str, CharModel] = {}
    #list to store the available items
    available_items: List[Tuple[str, str]] = []
    all_results: List[Tuple[object, Sequence[np.ndarray]]] = []
    #iterates through the csv map
    for char, csv_path in csv_by_char.items():
        #if the file does not exist and strict files is true, raise an error
        if strict_files and not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found for '{char}': {csv_path}")
        #if the file does not exist, continue
        if not Path(csv_path).exists():
            continue
        available_items.append((char, csv_path))
        all_results.extend(extract_all_from_csv(csv_path))

    #if there are no available items, return the models
    if not available_items:
        return models

    #Build one global codebook from all available characters.
    global_codebook = build_codebook(
        all_results,
        n_clusters=K,
        feature_fn=point_features,
        random_state=random_state,
    )

    for char, csv_path in available_items:
        results = extract_all_from_csv(csv_path)
        discrete_dataset: List[Tuple[str, DiscretizationResult]] = []
        for sample_char, processed_strokes in results:
            disc = discretize_character(
                processed_strokes,
                global_codebook,
                feature_fn=point_features,
                sep_token=K,
            )
            discrete_dataset.append((sample_char, disc))

        models[char] = train_char_model_from_discrete_dataset(
            discrete_dataset=discrete_dataset,
            codebook=global_codebook,
            K=K,
            S=S,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )
    #if persist parameters is true and there are models, save the global codebook and hmm parameters
    if persist_parameters and models:
        save_global_codebook(codebook=global_codebook, parameters_dir=parameters_dir)
        save_hmm_parameters(models=models, parameters_dir=parameters_dir)

    return models

#maps the greek vowel csv files
def greek_vowel_csv_map(training_dir: str = "data/training") -> Dict[str, str]:
    #returns the csv map
    return {
        "α": f"{training_dir}/lower_a_cleaned.csv",
        "ε": f"{training_dir}/lower_e_cleaned.csv",
        "η": f"{training_dir}/lower_htta_cleaned.csv",
        "ι": f"{training_dir}/lower_i_cleaned.csv",
        "ο": f"{training_dir}/lower_omikron_cleaned.csv",
        "υ": f"{training_dir}/lower_ypsilon_cleaned.csv",
        "ω": f"{training_dir}/lower_omega_cleaned.csv",
        "Α": f"{training_dir}/cap_a_cleaned.csv",
        "Ε": f"{training_dir}/cap_e_cleaned.csv",
        "Η": f"{training_dir}/cap_htta_cleaned.csv",
        "Ι": f"{training_dir}/cap_i_cleaned.csv",
        "Ο": f"{training_dir}/cap_omikron_cleaned.csv",
        "Υ": f"{training_dir}/cap_ypsilon_cleaned.csv",
        "Ω": f"{training_dir}/cap_omega_cleaned.csv",
    }

#maps the greek vowel csv candidates (checks fo different names and if they exist then uses them to train)
def greek_vowel_csv_candidates_map(training_dir: str = "data/training") -> Dict[str, List[str]]:
    #returns the csv candidates map
    return {
        "α": [f"{training_dir}/lower_a_cleaned.csv"],
        "ε": [f"{training_dir}/lower_e_cleaned.csv"],
        "η": [f"{training_dir}/lower_eta_cleaned.csv", f"{training_dir}/lower_htta_cleaned.csv"],
        "ι": [f"{training_dir}/lower_i_cleaned.csv"],
        "ο": [f"{training_dir}/lower_omikron_cleaned.csv", f"{training_dir}/lower_o_cleaned.csv"],
        "υ": [f"{training_dir}/lower_ypsilon_cleaned.csv"],
        "ω": [f"{training_dir}/lower_omega_cleaned.csv"],
        "Α": [f"{training_dir}/cap_a_cleaned.csv"],
        "Ε": [f"{training_dir}/cap_e_cleaned.csv"],
        "Η": [f"{training_dir}/cap_eta_cleaned.csv", f"{training_dir}/cap_htta_cleaned.csv"],
        "Ι": [f"{training_dir}/cap_i_cleaned.csv"],
        "Ο": [f"{training_dir}/cap_omikron_cleaned.csv", f"{training_dir}/cap_o_cleaned.csv"],
        "Υ": [f"{training_dir}/cap_ypsilon_cleaned.csv"],
        "Ω": [f"{training_dir}/cap_omega_cleaned.csv"],
    }


def resolve_existing_csv_map(
    csv_candidates_by_char: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    resolved: Dict[str, str] = {}
    missing: Dict[str, List[str]] = {}

    for char, candidates in csv_candidates_by_char.items():
        hit: Optional[str] = None
        for candidate in candidates:
            if Path(candidate).exists():
                hit = candidate
                break

        if hit is None:
            missing[char] = list(candidates)
            continue
        resolved[char] = hit

    return resolved, missing

#trains the greek vowel models
def train_greek_vowel_models(
    training_dir: str = "data/training",
    K: int = 12,
    S: int = 6,
    n_iter: int = 100,
    tol: float = 1e-2,
    random_state: int = 0,
    strict_files: bool = True,
    persist_parameters: bool = True,
    parameters_dir: str = "parameters",
) -> Dict[str, CharModel]:
    csv_map = greek_vowel_csv_map(training_dir=training_dir)
    return train_models_from_csv_map(
        csv_by_char=csv_map,
        K=K,
        S=S,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        strict_files=strict_files,
        persist_parameters=persist_parameters,
        parameters_dir=parameters_dir,
    )

#trains the greek vowel models resolved (if some files dont exist it proceeds anyway)
def train_greek_vowel_models_resolved(
    training_dir: str = "data/training",
    K: int = 12,
    S: int = 6,
    n_iter: int = 100,
    tol: float = 1e-2,
    random_state: int = 0,
    strict_files: bool = True,
    persist_parameters: bool = True,
    parameters_dir: str = "parameters",
) -> Tuple[Dict[str, CharModel], Dict[str, List[str]]]:
    csv_map, missing = resolve_existing_csv_map(
        greek_vowel_csv_candidates_map(training_dir=training_dir)
    )
    models = train_models_from_csv_map(
        csv_by_char=csv_map,
        K=K,
        S=S,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        strict_files=strict_files,
        persist_parameters=persist_parameters,
        parameters_dir=parameters_dir,
    )
    return models, missing

