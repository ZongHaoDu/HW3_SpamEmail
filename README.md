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
# HW3_SpamEmail

This repository contains a reproducible homework project that trains a spam/ham text classifier (logistic regression) and provides a polished Streamlit UI for inference and evaluation.

Key features
- Training pipeline (CLI) that downloads a public SMS spam dataset and trains a logistic regression baseline.
- Streamlit multi-page app (Home / Train / Predict / Evaluate / About) with interactive Plotly charts.
- Quick bootstrap script to create a tiny demo model for offline demos.
- Tests and a smoke training test to validate end-to-end flow.
- OpenSpec proposals and project metadata in `openspec/`.

Prerequisites
- Python 3.10+ recommended
- Windows (commands below use cmd.exe syntax)

Quick start
1. Create and activate a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bat
pip install -r requirements.txt
```

3. Create a demo model (fast) so the Streamlit app has something to load:

```bat
python scripts\bootstrap_model.py
```

4. Run the Streamlit app:

```bat
streamlit run streamlit_app\app.py
```

CLI: training and inspection
- Train on the public SMS spam CSV (downloads and trains):

```bat
python -m src.cli train --data-url "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
```

- Inspect a trained model (prints top features):

```bat
python -m src.cli inspect
```

Streamlit app — features summary
- Home: quick actions (bootstrap demo model, train on public CSV, inspect model) and example messages
- Train: bootstrap the demo model, run full training, or upload a pre-trained model/vectorizer
- Predict: single-message inference with example messages and prediction probabilities
- Evaluate: run evaluation on the public CSV or an uploaded CSV and view interactive plots including:
	- Confusion Matrix (interactive)
	- ROC curve (interactive with AUC)
	- Precision-Recall curve (interactive with AUC)
	- Top Tokens (corpus-level counts)
	- Top Tokens by Class (tokens most indicative of spam vs ham using model coefficients)
	- Threshold sweep for precision / recall / F1 and suggested best threshold by F1
- About: project summary and links

Model management
- Use the Train page to upload a model/vectorizer (joblib files). The sidebar exposes a "Download model" button when a model exists.
- The bootstrap script `scripts/bootstrap_model.py` creates a tiny demo model saved to `models/logistic_baseline.joblib` and `models/vectorizer.joblib`.

Testing

Run tests with pytest:

```bat
pytest -q
```

Project layout

- `src/` — preprocessing, training, CLI
- `streamlit_app/` — Streamlit UI and README
- `scripts/` — helper scripts (bootstrap_model.py)
- `models/` — saved model artifacts (not checked in by default)
- `tests/` — unit and smoke tests
- `openspec/` — project metadata and change proposals

Notes and next steps
- The Streamlit app uses Plotly for interactive visualizations. If you prefer static charts or a different style, I can switch to Matplotlib/Seaborn.
- For production or deployment, consider packaging the project (pyproject.toml) and installing with `pip install -e .` so `src` is importable without modifying sys.path.
- If you want the Top Tokens by Class computed from per-class TF-IDF averages instead of model coefficients, I can add a toggle in the Evaluate page.

Contact

Repo owner: `ZongHaoDu`
