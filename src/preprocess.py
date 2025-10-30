import os
import re
from typing import Iterable, Tuple

import pandas as pd


def load_data(path_or_url: str):
    """Load dataset from a local path or URL.

    Expected format: CSV with two columns (label, text). Handles header/no-header.
    Returns a DataFrame with columns ['label', 'text'].
    """
    # Try reading with no header first (as dataset is 'no_header')
    try:
        df = pd.read_csv(path_or_url, header=None, encoding="latin-1")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["label", "text"]
            return df
    except Exception:
        pass

    # Fallback: try with header
    df = pd.read_csv(path_or_url, encoding="latin-1")
    if "label" in df.columns and "text" in df.columns:
        return df[["label", "text"]]
    # Try first two columns
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    return df


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # remove non-word characters (keep basic punctuation removed)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_series(texts: Iterable[str]):
    for t in texts:
        yield clean_text(t)


def encode_labels(labels):
    # map common variants to 0/1
    mapping = {"ham": 0, "spam": 1}
    encoded = []
    for l in labels:
        s = str(l).strip().lower()
        encoded.append(mapping.get(s, 1 if s == "spam" else 0))
    return pd.Series(encoded)


def get_vectorizer(**kwargs):
    from sklearn.feature_extraction.text import TfidfVectorizer

    defaults = dict(ngram_range=(1, 2), max_features=10000)
    defaults.update(kwargs)
    return TfidfVectorizer(**defaults)
