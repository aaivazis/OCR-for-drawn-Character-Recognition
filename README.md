# OCR Greek Vowels

Handwritten Greek vowel recognition using:
- preprocessing (normalize/resample/smooth),
- global KMeans discretization,
- per-character HMM scoring.

## Setup

### 1) Create virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## Train Models (Global Codebook + HMMs)

This command rebuilds:
- `parameters/global_codebook.joblib`
- `parameters/hmm_*.joblib`

```powershell
.\venv\Scripts\python.exe -c "from src.pipeline.pipeline import pipeline_to_hmm_models_resolved; models, missing = pipeline_to_hmm_models_resolved(training_dir='data/training', K=12, S=6, n_iter=100, strict_files=False, persist_parameters=True, parameters_dir='parameters'); print('trained:', len(models)); print('missing:', missing)"
```

## Run GUI

```powershell
.\venv\Scripts\python.exe -m src.app.gui
```

When you press **Send** in the GUI:
1. strokes are saved to `data/raw`,
2. the latest JSON is preprocessed and discretized,
3. persisted HMMs score it,
4. top-3 predictions are shown with normalized probabilities.

## Optional Smoke Tests

```powershell
.\venv\Scripts\python.exe .\tests\test_pipeline_hmm.py
.\venv\Scripts\python.exe .\tests\test_inference_pipeline.py
```

