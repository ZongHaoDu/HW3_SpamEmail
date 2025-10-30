## Why

This change introduces a reproducible spam/ham classification workflow and a minimal demo UI so students and maintainers can:
- Experiment with baseline machine learning models for text classification.
- Reproduce training and evaluation steps locally.
- Inspect preprocessing steps, model metrics, and predictions via CLI and a Streamlit UI.

The repo currently lacks an implemented model, reproducible training flow, and an interactive demo. This proposal provides a phased plan to add those capabilities.

## What Changes

- Phase 1 — Baseline model
  - Ingest dataset from the provided public CSV URL.
  - Implement minimal preprocessing pipeline (cleaning, tokenization, TF-IDF).
  - Train and evaluate a logistic regression classifier as the baseline.
  - Provide a small CLI script to run training, evaluation, and save the model artifact (joblib).
  - Add unit tests for preprocessing and a smoke integration test that trains on a small sample.

- Phase 2 — Streamlit demo & expanded preprocessing
  - Build a Streamlit application for interactive inference and result visualization.
  - Add richer preprocessing options and toggle controls (stopwords, n-grams, TF vs TF-IDF).
  - Add visualizations: confusion matrix, ROC/PR curves, metric history, and intermediate outputs (token counts, sample transformed features).
  - Add a simple CLI mode for batch inference using a saved model.

## Data Source

Phase1 uses this public dataset (SMS spam) as the initial data source:

https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

Notes / assumption: the CSV has two columns per row: the label (`spam` or `ham`) and the message text, and it does not include a header row. If the schema differs, we'll update the ingestion code and docs.

## Deliverables

- `src/data/` — ingestion and small-sample download script
- `src/preprocess.py` — preprocessing utilities with unit tests
- `src/train.py` — script to train logistic regression baseline and save `models/logistic_baseline.joblib`
- `src/cli.py` — simple CLI to run training/eval and batch inference
- `streamlit_app/` — Streamlit app for interactive inference and visualizations
- `tests/` — pytest tests: preprocessing unit tests and a smoke e2e test
- `README.md` updates with run instructions

## Acceptance Criteria

- Phase 1
  - There is a reproducible training script that can be run locally and produces a saved model artifact.
  - Tests pass: preprocessing unit tests and one smoke train-eval test run.
  - Baseline logistic regression runs and outputs metrics (precision, recall, F1, accuracy). Example target: F1 > 0.85 on the dataset is optional; the key is reproducibility and correct pipeline.

- Phase 2
  - Streamlit app runs and allows entering a message to get a prediction and display metrics and visualizations.
  - Visualizations include confusion matrix and ROC/PR curves for evaluation set.

## Impact

- Files added or modified:
  - `src/` (new): preprocessing, training, CLI
  - `streamlit_app/` (new): Streamlit UI
  - `tests/` (new): unit and smoke tests
  - `README.md` (modified): add run instructions
  - `openspec/changes/add-spam-classifier-baseline/*` (this change)

- No breaking API changes expected; this is an additive feature.

## Validation

- Run unit tests: `pytest -q`.
- Run smoke training: `python src/train.py --data-url <URL> --output models/logistic_baseline.joblib` and confirm artifact created.
- Run Streamlit locally: `streamlit run streamlit_app/app.py` and verify the UI loads and can make predictions.

## Risks and Mitigations

- Risk: Dataset schema or availability changes.
  - Mitigation: include a small sample dataset in `data/sample/` for tests and provide downloader that can handle minor schema variants (header/no-header, different delimiter).

- Risk: Student machines may have varying Python versions or missing packages.
  - Mitigation: include a `requirements.txt` with pinned minimum versions and recommend using a venv.

## Timeline / Phases

- Phase 1 (baseline) — ~1–2 days
  - Ingest dataset, implement preprocessing, train logistic regression baseline, add tests, add CLI.

- Phase 2 (UI & enrichment) — ~2–3 days
  - Streamlit app, richer preprocessing options, visualizations, and documentation.

## Contacts

- Owner: `ZongHaoDu` (repo owner). Add TA or collaborator if needed.

## Next Steps

1. Review this proposal and confirm the dataset schema and acceptance targets.
2. Approve and I will scaffold the implementation files and open a feature branch `feature/add-spam-classifier-baseline` with incremental commits.
