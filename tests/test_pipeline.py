from src.pipeline.pipeline import pipeline_to_discrete


ds = pipeline_to_discrete("data/training/lower_a_cleaned.csv", K=12)
flat = ds[0][1].flat_codes
print("Count SEP:", (flat == 12).sum())
print("Max code:", flat.max())
print("Min code:", flat.min())

