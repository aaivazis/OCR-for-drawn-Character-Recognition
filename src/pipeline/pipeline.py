from src.algorithms.kmeans import discretize_character, build_codebook
from src.preprocessing.ft_extract import extract_all_from_csv
from src.features.point_features import point_features

#pipeline from the processed data to discretization
def pipeline_to_discrete_with_codebook(csv: str, K: int = 12):
    SEP = K
    results = extract_all_from_csv(csv)

    #using kmeans algorithm make the clusters
    model = build_codebook(results, n_clusters=K, feature_fn=point_features, random_state=0)
    discrete_dataset = []

    #match each point to the nearest cluster, makes data from continuous (x,y,t) to discrete O0...011
    for char, processed_strokes in results:
        disc = discretize_character(
            processed_strokes,
            model,
            feature_fn=point_features,
            sep_token=SEP,
        )
        #return discrete points again respecting point and stroke order
        discrete_dataset.append((char, disc))

    return discrete_dataset, model

#use only if model codebook is already constructed
def pipeline_to_discrete(csv: str, K: int = 12):
    discrete_dataset, _model = pipeline_to_discrete_with_codebook(csv, K=K)
    return discrete_dataset

#pipeline from discretization to feeding data to the 13 HMM models
def pipeline_to_hmm_models(
    training_dir: str = "data/training",
    K: int = 12,
    S: int = 6,
    n_iter: int = 100,
    tol: float = 1e-2,
    random_state: int = 0,
    strict_files: bool = True,
    persist_parameters: bool = True,
    parameters_dir: str = "parameters",
):
    from src.models.hmm_models import train_greek_vowel_models

    #returns result of 
    return train_greek_vowel_models(
        training_dir=training_dir,
        K=K,
        S=S,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        strict_files=strict_files,
        persist_parameters=persist_parameters,
        parameters_dir=parameters_dir,
    )

#resolved pipeline
def pipeline_to_hmm_models_resolved(
    training_dir: str = "data/training",
    K: int = 12,
    S: int = 6,
    n_iter: int = 100,
    tol: float = 1e-2,
    random_state: int = 0,
    strict_files: bool = True,
    persist_parameters: bool = True,
    parameters_dir: str = "parameters",
):
    from src.models.hmm_models import train_greek_vowel_models_resolved

    return train_greek_vowel_models_resolved(
        training_dir=training_dir,
        K=K,
        S=S,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        strict_files=strict_files,
        persist_parameters=persist_parameters,
        parameters_dir=parameters_dir,
    )

#pipeline gui uses to send the raw data to be processed and discretized
def pipeline_raw_json_to_top3(
    json_path: str,
    parameters_dir: str = "parameters",
):
    from src.pipeline.inference import pipeline_raw_json_to_top_scores

    return pipeline_raw_json_to_top_scores(
        json_path=json_path,
        parameters_dir=parameters_dir,
        top_n=3,
    )