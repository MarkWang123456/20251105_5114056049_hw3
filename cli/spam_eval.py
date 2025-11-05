from __future__ import annotations

import argparse
import os

import joblib
import numpy as np

from ml.data import load_xy
from ml.metrics import compute_basic_metrics, save_metrics_csv, plot_confusion_matrix, plot_roc_pr, save_threshold_sweep


def main():
    p = argparse.ArgumentParser(description="Evaluate a saved spam model")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", default="metrics")
    args = p.parse_args()

    pipe = joblib.load(args.model)
    X, y = load_xy(args.data)

    y_pred = pipe.predict(X)
    if hasattr(pipe.named_steps["model"], "decision_function"):
        y_score = pipe.decision_function(X)
    elif hasattr(pipe.named_steps["model"], "predict_proba"):
        y_score = pipe.predict_proba(X)[:, 1]
    else:
        y_score = y_pred

    metrics = compute_basic_metrics(y, y_pred)
    print("accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}".format(**metrics))

    os.makedirs(args.out_dir, exist_ok=True)
    save_metrics_csv(metrics, args.out_dir)
    plot_confusion_matrix(y, y_pred, args.out_dir)
    try:
        y_score_arr = np.asarray(y_score)
        if y_score_arr.ndim == 1:
            plot_roc_pr(y, y_score_arr, args.out_dir)
            save_threshold_sweep(y, y_score_arr, args.out_dir)
    except Exception:
        pass

    print(f"[ok] Metrics exported to {args.out_dir}")


if __name__ == "__main__":
    main()
