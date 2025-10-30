# Streamlit demo for HW3_SpamEmail

This folder contains a minimal Streamlit application demonstrating single-message inference for the logistic regression baseline.

How to run

1. Ensure dependencies are installed (see root `requirements.txt`).
2. Train the baseline model using the CLI or training script, which will create `models/logistic_baseline.joblib` and `models/vectorizer.joblib`:

```bat
python -m src.cli train
```

3. Run the Streamlit app:

```bat
streamlit run streamlit_app/app.py
```

Notes
- The app includes a button to trigger training via the existing `src.train` code. This is a convenience for demos but may take a minute.
- For richer evaluation and visualizations, run the training script to produce metrics and use a dedicated notebook or CLI tooling.
