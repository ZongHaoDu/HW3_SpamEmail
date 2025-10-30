import os
import re
from typing import Iterable, Tuple, List, Dict, Any

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


def tokenize(text: str) -> List[str]:
    """Simple tokenizer that splits on whitespace after cleaning."""
    return clean_text(text).split()


def top_tokens_from_corpus(vectorizer, texts: Iterable[str], top_n: int = 20) -> List[tuple]:
    """Return top_n tokens by corpus frequency using the provided vectorizer.

    This fits the vectorizer on the provided texts and computes summed token counts.
    Returns list of (token, count) sorted descending.
    """
    X = vectorizer.fit_transform(texts)
    # sum over rows to get token counts
    import numpy as _np

    counts = _np.asarray(X.sum(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()
    pairs = list(zip(feature_names, counts))
    pairs.sort(key=lambda x: -x[1])
    return pairs[:top_n]


def vocab_top_tokens(vectorizer, top_n: int = 20) -> List[tuple]:
    """If the vectorizer is already fitted, return tokens sorted by vocabulary order.

    This is a lightweight helper that returns token names (no counts).
    """
    try:
        feature_names = vectorizer.get_feature_names_out()
        return [(f, i) for i, f in enumerate(feature_names[:top_n])]
    except Exception:
        return []
