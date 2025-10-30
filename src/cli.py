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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
