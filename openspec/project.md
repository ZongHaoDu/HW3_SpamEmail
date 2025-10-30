# Project Context
```markdown
# Project Context

## Purpose
This repository contains the code and artifacts for "HW3_SpamEmail": a homework project to build, evaluate, and deliver a spam/ham email classifier. The goals are:
- Explore feature engineering and text preprocessing for email data
- Train and evaluate one or more baseline classifiers (e.g., Naive Bayes, logistic regression)
- Provide reproducible training/evaluation steps and a saved model artifact for inference
- Deliver clear specs, tests, and minimal CLI/notebook examples for reproducibility

## Tech Stack
- Python 3.10+ (recommend 3.11)
- Data: pandas, numpy
- ML: scikit-learn (primary), optionally xgboost/lightgbm for experiments
- Text: scikit-learn feature extraction (CountVectorizer / TfidfVectorizer), nltk or spaCy (optional)
- Serialization: joblib
- Development: Jupyter / JupyterLab for analysis
- Testing / linting: pytest, black, isort, flake8
- Packaging / tooling: pip (requirements.txt) or virtualenv/venv

## Project Conventions

### Code Style
- Use idiomatic Python 3.10+ with type hints where helpful.
- Format code using `black` and `isort` before committing.
- Keep functions small and pure where possible; prefer explicit names over abbreviations.

### Architecture Patterns
- Small, single-repo structure with the following logical layout:
	- `data/` — raw and processed datasets (not checked in, add READMEs)
	- `notebooks/` — exploratory notebooks (analysis and experiments)
	- `src/` — application code (preprocessing, models, training, inference)
	- `tests/` — unit and integration tests
	- `models/` — serialized model artifacts (not checked in; place in artifacts storage)
- Simple pipeline: ingest → preprocess → featurize → train → evaluate → save

### Testing Strategy
- Unit tests for preprocessing, feature extraction, and metric calculations (pytest).
- A small smoke/integration test that trains a tiny model on a tiny sample dataset to ensure end-to-end flow.
- Tests should run quickly (<60s) locally. Use parametrized tests for model metrics.

### Git Workflow
- Branch from `main` using feature branches: `feature/<short-desc>` or `fix/<short-desc>`.
- Open PRs against `main` with descriptive titles, link to the change proposal when relevant.
- Commit messages follow: `<type>(scope): short description` where type ∈ {feat, fix, chore, docs, test, refactor}.

## Domain Context
- Binary classification task: spam vs. ham emails.
- Typical datasets have heavy class imbalance — use precision/recall/F1 and PR curves.
- Common preprocessing: tokenization, lowercasing, stopword removal, n-grams, TF-IDF weighting.
- Privacy: email text may contain PII; treat data as sensitive and avoid checking real data into repo.

## Important Constraints
- Keep the code runnable on a student laptop: avoid heavy GPU/cluster dependencies.
- Training should be reproducible with a single requirements file and a small dataset sample.
- Avoid including large datasets or model binaries in the repo; provide pointers or scripts to download them.

## External Dependencies
- scikit-learn, pandas, numpy, joblib
- Optional: nltk or spaCy for advanced preprocessing
- Optional experiment tracking: mlflow or Weights & Biases (not required for HW)

## Assumptions
- I assumed this is a Python-based ML homework project (name: `HW3_SpamEmail`). If your project uses another stack (e.g., Node.js, Java), tell me and I will adapt the file.
- I assumed typical ML tooling (scikit-learn, pandas). Confirm any specific libraries or CI you want to document.

## Owner / Contacts
- Repo owner: `ZongHaoDu` (GitHub user from repo context). Add email or team contact if needed.

```
