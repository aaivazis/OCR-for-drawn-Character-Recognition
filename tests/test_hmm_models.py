from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.hmm_models import train_greek_vowel_models_resolved


models, missing = train_greek_vowel_models_resolved(K=12, S=6, n_iter=5, strict_files=False)
print("trained models:", len(models))
print("keys:", sorted(models.keys()))
print("missing:", missing)
codebook_ids = {id(model.codebook) for model in models.values()}
print("shared global codebook:", len(codebook_ids) == 1)

