import os
import streamlit as st
import joblib

from pathlib import Path


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
    st.caption("This is a minimal demo. For richer evaluation, run the training script and use the CLI to produce evaluation artifacts (confusion matrix, ROC curve).")


if __name__ == "__main__":
    main()
