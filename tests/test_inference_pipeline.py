from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.inference import pipeline_raw_json_to_top_scores
from src.pipeline.pipeline import pipeline_to_hmm_models_resolved, pipeline_raw_json_to_top3


# Ensure persisted parameters (including global codebook) exist.
pipeline_to_hmm_models_resolved(
    training_dir="data/training",
    K=12,
    S=6,
    n_iter=5,
    strict_files=False,
    persist_parameters=True,
    parameters_dir="parameters",
)

raw_files = sorted(Path("data/raw").glob("letter_*.json"))
if not raw_files:
    print("No raw files found.")
else:
    top_scores = pipeline_raw_json_to_top_scores(
        str(raw_files[-1]),
        parameters_dir="parameters",
        top_n=3,
    )
    top3 = pipeline_raw_json_to_top3(
        str(raw_files[-1]),
        parameters_dir="parameters",
    )
    print("raw file:", raw_files[-1])
    print("top 3:", top_scores)
    print("top3 wrapper:", top3)

