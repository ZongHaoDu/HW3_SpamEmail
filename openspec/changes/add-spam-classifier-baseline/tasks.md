## 1. Scaffolding & Proposal
- [ ] 1.1 Review and approve `openspec/changes/add-spam-classifier-baseline/proposal.md` (owner: repo maintainer / student)

## 2. Phase 1 — Baseline Implementation
- [ ] 2.1 Create `src/data/downloader.py` to fetch and cache the CSV dataset from the URL
- [ ] 2.2 Implement `src/preprocess.py` with functions:
  - `load_data()` — handle header/no-header variants
  - `clean_text()` — basic cleaning
  - `build_vectorizer()` — TF-IDF / CountVectorizer
- [ ] 2.3 Implement `src/train.py` that trains LogisticRegression and saves `models/logistic_baseline.joblib`
- [ ] 2.4 Add unit tests for preprocessing (`tests/test_preprocess.py`) and a smoke e2e test (`tests/test_smoke_train.py`)
- [ ] 2.5 Update `README.md` with quick start and commands

## 3. Phase 2 — Streamlit & Visualization
- [ ] 3.1 Scaffold `streamlit_app/app.py` with routes/pages for:
  - single-text prediction
  - dataset evaluation & visualizations
- [ ] 3.2 Add visualization helpers (confusion matrix, ROC/PR, sample tokens)
- [ ] 3.3 Add CLI batch inference in `src/cli.py`

## 4. Validation & CI
- [ ] 4.1 Add `requirements.txt` and test run instructions
- [ ] 4.2 (Optional) Add GitHub Actions workflow to run `pytest` and `flake8` on PRs

## 5. Finalize
- [ ] 5.1 Mark tasks complete and archive the change in `openspec/changes/archive/` once merged
