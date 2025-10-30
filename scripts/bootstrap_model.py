"""Bootstrap a minimal trained model artifact for the Streamlit demo.

This trains a tiny logistic regression on an embedded sample and writes:
- models/logistic_baseline.joblib
- models/vectorizer.joblib

Run this locally (no network required):
    python scripts\bootstrap_model.py
"""
from pathlib import Path
import sys
import pathlib
import joblib

# Ensure repo root is on sys.path so `from src import ...` works when running the script directly
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import clean_text, get_vectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


MODEL_PATH = Path("models/logistic_baseline.joblib")
VEC_PATH = Path("models/vectorizer.joblib")


def main():
    # tiny built-in dataset (label,text)
    samples = [
        ("ham", "Hey, are we still meeting tomorrow?"),
        ("spam", "Win money now!!! Click here to claim your prize"),
        ("ham", "Don't forget the meeting notes"),
        ("spam", "Lowest prices on meds, buy now"),
        ("ham", "Can you review my PR?"),
        ("spam", "Congratulations, you have won a free ticket"),
    ]

    labels, texts = zip(*samples)
    texts = [clean_text(t) for t in texts]
    y = np.array([1 if l == "spam" else 0 for l in labels])

    vec = get_vectorizer(ngram_range=(1, 2), max_features=2000)
    X = vec.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    VEC_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vec, VEC_PATH)

    print(f"Wrote model to: {MODEL_PATH.resolve()}")
    print(f"Wrote vectorizer to: {VEC_PATH.resolve()}")


if __name__ == "__main__":
    main()
