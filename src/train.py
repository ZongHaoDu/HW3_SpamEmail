import argparse
import os
from typing import Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

from src.data.downloader import ensure_data
from src.preprocess import load_data, preprocess_series, encode_labels, get_vectorizer


def train_and_evaluate(
    data_source: str,
    output_model: str = "models/logistic_baseline.joblib",
    output_vectorizer: str = "models/vectorizer.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[LogisticRegression, object]:
    path = data_source
    if data_source.startswith("http"):
        path = ensure_data(data_source, out_path="data/raw.csv")

    df = load_data(path)
    texts = list(preprocess_series(df["text"]))
    y = encode_labels(df["label"]).values

    vectorizer = get_vectorizer()
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Ensure output dirs
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    os.makedirs(os.path.dirname(output_vectorizer), exist_ok=True)

    joblib.dump(model, output_model)
    joblib.dump(vectorizer, output_vectorizer)

    print("Training complete. Metrics:")
    print(classification_report(y_test, y_pred, digits=4))
    return model, metrics


def cli():
    parser = argparse.ArgumentParser(description="Train logistic regression spam classifier")
    parser.add_argument("--data-url", default=(
        "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    ))
    parser.add_argument("--output-model", default="models/logistic_baseline.joblib")
    parser.add_argument("--output-vectorizer", default="models/vectorizer.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    model, metrics = train_and_evaluate(
        args.data_url, args.output_model, args.output_vectorizer, test_size=args.test_size, random_state=args.random_state
    )
    print("Saved model to:", args.output_model)
    print("Saved vectorizer to:", args.output_vectorizer)


if __name__ == "__main__":
    cli()
