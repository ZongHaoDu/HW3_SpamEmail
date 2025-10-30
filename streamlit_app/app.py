import os
import sys
import pathlib
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
)

from pathlib import Path
# Ensure repo root is on sys.path so `from src import ...` works when Streamlit runs this file
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import load_data, preprocess_series, encode_labels


MODEL_PATH = Path("models/logistic_baseline.joblib")
VEC_PATH = Path("models/vectorizer.joblib")


def load_model_and_vectorizer():
    if MODEL_PATH.exists() and VEC_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            vec = joblib.load(VEC_PATH)
            return model, vec
        except Exception as e:
            st.error(f"Failed to load model/vectorizer: {e}")
    return None, None


def main():
    st.title("HW3 — Spam Classifier (Streamlit demo)")

    st.markdown(
        "This minimal demo lets you enter a message and get a prediction from the logistic regression baseline (if trained)."
    )

    model, vec = load_model_and_vectorizer()

    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area("Message", height=150, value="Win a prize! Click now to claim.")
    with col2:
        st.markdown("### Actions")
        if st.button("Predict"):
            if model is None or vec is None:
                st.warning("Model not found. Please train the baseline model first via the CLI or press 'Train model' below.")
            else:
                X = vec.transform([text])
                pred = model.predict(X)[0]
                label = "spam" if int(pred) == 1 else "ham"
                st.success(f"Prediction: {label}")
                try:
                    probs = model.predict_proba(X)[0]
                    st.write({"ham_prob": float(probs[0]), "spam_prob": float(probs[1])})
                except Exception:
                    pass

    st.markdown("---")

    st.header("Model status")
    if model is None:
        st.info("No trained model found at `models/logistic_baseline.joblib`.")
        if st.button("Train model (runs train script)"):
            with st.spinner("Training baseline model — this may take a moment..."):
                # Import here to avoid long imports when the app loads and model exists
                try:
                    from src.train import train_and_evaluate

                    train_and_evaluate(
                        data_source=(
                            "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
                        ),
                        output_model=str(MODEL_PATH),
                        output_vectorizer=str(VEC_PATH),
                    )
                    st.success("Training finished. Reload the page to use the model.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
    else:
        st.success(f"Loaded model from {MODEL_PATH}")

    st.markdown("---")
    st.markdown("---")
    st.header("Evaluation")
    st.caption(
        "Run a quick evaluation using the original dataset (downloads if needed) and display confusion matrix, ROC, and PR curves."
    )

    if model is None:
        st.info("No model available for evaluation. Train a model first.")
    else:
        if st.button("Evaluate model"):
            with st.spinner("Loading data and running evaluation..."):
                try:
                    DATA_URL = (
                        "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
                    )
                    df = load_data(DATA_URL)
                    texts = list(preprocess_series(df["text"]))
                    y_true = encode_labels(df["label"]).values

                    X = vec.transform(texts)
                    try:
                        y_score = model.predict_proba(X)[:, 1]
                    except Exception:
                        y_score = None
                    y_pred = model.predict(X)

                    # Confusion matrix
                    cm = confusion_matrix(y_true, y_pred)
                    fig1, ax1 = plt.subplots()
                    im = ax1.imshow(cm, cmap="Blues")
                    ax1.set_title("Confusion Matrix")
                    ax1.set_xlabel("Predicted")
                    ax1.set_ylabel("Actual")
                    ax1.set_xticks([0, 1])
                    ax1.set_yticks([0, 1])
                    ax1.set_xticklabels(["ham", "spam"])
                    ax1.set_yticklabels(["ham", "spam"])
                    for (i, j), val in np.ndenumerate(cm):
                        ax1.text(j, i, int(val), ha="center", va="center", color="black")

                    st.pyplot(fig1)

                    # ROC curve
                    if y_score is not None and len(set(y_true)) > 1:
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        roc_auc = roc_auc_score(y_true, y_score)
                        fig2, ax2 = plt.subplots()
                        ax2.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
                        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
                        ax2.set_xlabel("False Positive Rate")
                        ax2.set_ylabel("True Positive Rate")
                        ax2.set_title("ROC Curve")
                        ax2.legend(loc="lower right")
                        st.pyplot(fig2)

                        # PR curve
                        precision, recall, _ = precision_recall_curve(y_true, y_score)
                        pr_auc = auc(recall, precision)
                        fig3, ax3 = plt.subplots()
                        ax3.plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})")
                        ax3.set_xlabel("Recall")
                        ax3.set_ylabel("Precision")
                        ax3.set_title("Precision-Recall Curve")
                        ax3.legend(loc="lower left")
                        st.pyplot(fig3)
                    else:
                        st.info("Cannot compute ROC/PR curves: model does not provide probability estimates or labels are single-class.")

                    # Summary metrics
                    try:
                        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

                        summary = {
                            "accuracy": float(accuracy_score(y_true, y_pred)),
                            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                        }
                        st.write("### Summary metrics")
                        st.json(summary)
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    st.markdown("---")
    st.caption("This is a minimal demo. For richer evaluation, run the training script and use the CLI to produce evaluation artifacts (confusion matrix, ROC curve).")


if __name__ == "__main__":
    main()
