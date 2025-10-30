import os
import tempfile

from src.train import train_and_evaluate


def test_smoke_train_creates_model():
    # small sample dataset
    data = "ham,Hello there\nspam,Buy cheap products now\nham,How are you?\nspam,Win a prize"
    with tempfile.TemporaryDirectory() as td:
        data_path = os.path.join(td, "sample.csv")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(data)

        model_path = os.path.join(td, "model.joblib")
        vec_path = os.path.join(td, "vec.joblib")

        model, metrics = train_and_evaluate(data_path, output_model=model_path, output_vectorizer=vec_path, test_size=0.5)
        assert os.path.exists(model_path)
        assert os.path.exists(vec_path)
        assert "f1" in metrics
