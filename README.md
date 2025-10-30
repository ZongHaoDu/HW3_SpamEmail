# HW3_SpamEmail

A small reproducible homework project that trains a spam/ham text classifier (logistic regression) and provides
CLI and Streamlit UI for inference and evaluation.

This repository includes:
 - A minimal preprocessing and training pipeline (`src/`)
 - A Streamlit demo app (`streamlit_app/`) for single-message inference and evaluation visualizations
 - Tests and a smoke training test (`tests/`)
 - OpenSpec change proposals and project metadata (`openspec/`)

This README explains how to set up the environment, produce a demo model, run training, and launch the Streamlit app.

## Quickstart (recommended)

1. Create and activate a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bat
pip install -r requirements.txt
```

3. Create a demo model for the Streamlit app (fast):

```bat
python scripts\bootstrap_model.py
```

4. Run the Streamlit demo:

```bat
streamlit run streamlit_app\app.py
```

The bootstrap script trains a tiny synthetic model for demo purposes and writes `models/logistic_baseline.joblib` and
`models/vectorizer.joblib` so the Streamlit app can load them immediately.

## Training on the real dataset

The project includes a training script that downloads a public SMS spam dataset and trains a logistic regression baseline.

```bat
python -m src.cli train --data-url "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
```

This will save artifacts to `models/logistic_baseline.joblib` and `models/vectorizer.joblib` by default.

## Inspecting a trained model (CLI)

After training you can inspect model metadata and top features with:

```bat
python -m src.cli inspect
```

## Tests

Run unit tests and the smoke training test with pytest:

```bat
pytest -q
```

## Project layout

 - `src/` — preprocessing, training, CLI
 - `streamlit_app/` — Streamlit demo and README
 - `scripts/bootstrap_model.py` — create a small demo model for offline demos
 - `models/` — saved model artifacts (not checked in by default)
 - `tests/` — unit and smoke tests
 - `openspec/` — project metadata and change proposals

## Notes and assumptions

 - The code is written for Python 3.10+ and uses scikit-learn for modeling.
 - The public dataset used in Phase 1 is the SMS spam CSV linked above; the project assumes a two-column CSV (label, text) without a header by default, but `src.preprocess.load_data` tries to handle header/no-header variants.
 - The bootstrap model is intentionally tiny and intended only for demos — do not use it for evaluation.

## Contributing

Follow the OpenSpec workflow in `openspec/AGENTS.md` when proposing features or changes. Create a change proposal under `openspec/changes/` and follow the tasks checklist before implementation.

## Contact

Repo owner: `ZongHaoDu`

```
# HW3_SpamEmail