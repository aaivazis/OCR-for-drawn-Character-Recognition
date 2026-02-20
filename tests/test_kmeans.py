import matplotlib.pyplot as plt
from src.preprocessing.ft_extract import extract_all_from_csv
from src.algorithms.kmeans import build_codebook, discretize_character
from src.features.point_features import point_features

results = extract_all_from_csv("data/training/lower_a_cleaned.csv")

K = 12
SEP = K  # separator token that marks stroke boundaries in the flat sequence
model = build_codebook(results, n_clusters=K, feature_fn=point_features, random_state=0)
for char, processed_strokes in results:

    disc = discretize_character(
        processed_strokes,
        model,
        feature_fn=point_features,
        sep_token=SEP
    )

print(f"Discrete strokes: {len(disc.stroke_codes)}")
print("First stroke codes (first 25):", disc.stroke_codes[0][:64] if disc.stroke_codes else [])
print("Flat sequence length:", len(disc.flat_codes) if disc.flat_codes is not None else 0)
print("Separator token:", SEP)

# Visualize the flat discrete sequence as a simple index plot
if disc.flat_codes is not None and len(disc.flat_codes) > 0:
    plt.figure(figsize=(10, 2.5))
    plt.plot(disc.flat_codes, ".", markersize=3)
    plt.title(f"Discrete code sequence for '{char}' (SEP={SEP})")
    plt.xlabel("sequence index")
    plt.ylabel("code id")
    plt.show()
