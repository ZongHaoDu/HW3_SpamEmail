from src.preprocess import clean_text, get_vectorizer


def test_clean_text():
    assert clean_text("Hello, WORLD!!") == "hello world"
    assert clean_text(None) == ""


def test_vectorizer_shapes():
    texts = ["this is spam", "hello ham", "buy now spam offer"]
    vec = get_vectorizer(ngram_range=(1, 1), max_features=10)
    X = vec.fit_transform(texts)
    assert X.shape[0] == len(texts)
