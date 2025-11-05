from __future__ import annotations

import argparse
import os
from typing import Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.data import DatasetPaths, ensure_dataset, load_xy
from ml.features import make_vectorizer
from ml.metrics import (
    compute_basic_metrics,
    save_metrics_csv,
    plot_confusion_matrix,
    plot_roc_pr,
    save_threshold_sweep,
)
from ml.models import get_model


def parse_ngram(s: str) -> Tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--ngram must be like '1,2'")
    return int(parts[0]), int(parts[1])


def main():
    p = argparse.ArgumentParser(description="Train spam classifier")
    # Prefer user's local dataset by default
    p.add_argument("--data", default=os.path.join("dataset", "sms_spam_no_header.csv"))
    p.add_argument("--model", default="svm", choices=["lr", "nb", "svm"], help="Model type (default: svm)")
    p.add_argument("--vec", default="tfidf", choices=["tfidf", "bow"], help="Vectorizer: tfidf or bow (default: tfidf)")
    p.add_argument("--model-out", required=True, help="Path to save trained model (joblib)")
    p.add_argument("--out-dir", default="metrics", help="Where to save metrics and plots")
    p.add_argument("--test-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ngram", type=parse_ngram, default=(1, 1), help="n-gram range like '1,2'")
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--min-df", type=int, default=1)
    p.add_argument("--stop-words", choices=["none", "english"], default="none")
    p.add_argument("--token-pattern", type=str, default=r"(?u)\\b\\w+\\b", help="Regex token pattern for vectorizer")
    args = p.parse_args()

    # Load dataset directly from provided path (no auto-download)
    X, y = load_xy(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed, stratify=y
    )

    vec = make_vectorizer(
        kind=args.vec,
        ngram=args.ngram,
        max_features=args.max_features,
        min_df=args.min_df,
        stop_words=(None if args.stop_words == "none" else "english"),
        token_pattern=args.token_pattern,
    )
    model = get_model(args.model)
    pipe = Pipeline([("vec", vec), ("model", model)])

    try:
        pipe.fit(X_train, y_train)
    except ValueError as e:
        if "empty vocabulary" in str(e).lower():
            # Fallback: relax vectorizer settings
            from ml.features import make_vectorizer as _mv
            vec_fallback = _mv(
                kind=args.vec,
                ngram=args.ngram,
                max_features=None,
                min_df=1,
                stop_words=None,
                token_pattern=r"(?u)\b\w\w+\b",
            )
            pipe = Pipeline([("vec", vec_fallback), ("model", model)])
            pipe.fit(X_train, y_train)
        else:
            raise

    y_pred = pipe.predict(X_test)
    # Score for ROC/PR: use decision_function if available, else predict_proba
    if hasattr(pipe.named_steps["model"], "decision_function"):
        y_score = pipe.decision_function(X_test)
    elif hasattr(pipe.named_steps["model"], "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred  # fallback

    metrics = compute_basic_metrics(y_test, y_pred)
    print("accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}".format(**metrics))

    os.makedirs(args.out_dir, exist_ok=True)
    save_metrics_csv(metrics, args.out_dir)
    plot_confusion_matrix(y_test, y_pred, args.out_dir)
    try:
        # Some models may not provide meaningful scores; guard plotting
        y_score_arr = np.asarray(y_score)
        if y_score_arr.ndim == 1:
            plot_roc_pr(y_test, y_score_arr, args.out_dir)
            save_threshold_sweep(y_test, y_score_arr, args.out_dir)
    except Exception:
        pass

    # Save model pipeline
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f"[ok] Model saved to {args.model_out}. Metrics in {args.out_dir}")


if __name__ == "__main__":
    main()
