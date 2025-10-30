import argparse
import os
import sys

from src import train


def main(argv=None):
    parser = argparse.ArgumentParser(description="CLI for HW3_SpamEmail tasks")
    sub = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train", help="Train the baseline model")
    train_p.add_argument("--data-url", default=(
        "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    ))
    train_p.add_argument("--output-model", default="models/logistic_baseline.joblib")
    train_p.add_argument("--output-vectorizer", default="models/vectorizer.joblib")
    train_p.add_argument("--test-size", type=float, default=0.2)
    train_p.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args(argv)
    if args.cmd == "train":
        train.train_and_evaluate(
            data_source=args.data_url,
            output_model=args.output_model,
            output_vectorizer=args.output_vectorizer,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    elif args.cmd == "inspect":
        # Load model and vectorizer and print simple metadata
        import joblib

        try:
            model = joblib.load(args.output_model)
            vec = joblib.load(args.output_vectorizer)
            print(f"Loaded model: {args.output_model}")
            print(f"Vectorizer features: {len(vec.get_feature_names_out())}")
            # print top features if available
            try:
                coef = model.coef_.ravel()
                feature_names = vec.get_feature_names_out()
                pairs = list(zip(feature_names, coef))
                pairs.sort(key=lambda x: -x[1])
                print("Top positive features:")
                for f, c in pairs[:20]:
                    print(f, c)
                print("Top negative features:")
                for f, c in pairs[-20:]:
                    print(f, c)
            except Exception as e:
                print("Could not extract top features:", e)
        except Exception as e:
            print("Failed to load model/vectorizer:", e)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
