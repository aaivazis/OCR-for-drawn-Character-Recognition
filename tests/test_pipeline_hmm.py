from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.pipeline import pipeline_to_hmm_models_resolved
from src.models.hmm_models import load_hmm_parameters


models, missing = pipeline_to_hmm_models_resolved(
    training_dir="data/training",
    K=12,
    S=6,
    n_iter=5,
    strict_files=False,
)
print("trained models:", len(models))
print("keys:", sorted(models.keys()))
print("missing:", missing)
codebook_ids = {id(model.codebook) for model in models.values()}
print("shared global codebook:", len(codebook_ids) == 1)
loaded = load_hmm_parameters("parameters")
print("loaded parameter files:", len(loaded))
print("saved/loaded match:", len(loaded) == len(models))

