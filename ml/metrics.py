from __future__ import annotations

import os
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def compute_basic_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_metrics_csv(metrics: Dict[str, float], out_dir: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(path, index=False)
    return path


def plot_confusion_matrix(y_true, y_pred, out_dir: Optional[str] = None) -> Optional[str]:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xlabel="Predicted", ylabel="True")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    path = None
    if out_dir:
        ensure_dir(out_dir)
        path = os.path.join(out_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_pr(y_true, y_score, out_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend()
    path_roc = None
    if out_dir:
        ensure_dir(out_dir)
        path_roc = os.path.join(out_dir, "roc.png")
        fig1.savefig(path_roc, dpi=150)
    plt.close(fig1)

    # PR
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = auc(r, p)
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.plot(r, p, label=f"AP={ap:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    path_pr = None
    if out_dir:
        ensure_dir(out_dir)
        path_pr = os.path.join(out_dir, "pr.png")
        fig2.savefig(path_pr, dpi=150)
    plt.close(fig2)

    return path_roc, path_pr


def threshold_sweep(y_true, y_score, thresholds: Optional[np.ndarray] = None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if thresholds is None:
        lo, hi = float(np.min(y_score)), float(np.max(y_score))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            thresholds = np.linspace(0.0, 1.0, 51)
        else:
            thresholds = np.linspace(lo, hi, 51)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        tpr = rec
        fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
        rows.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tpr": float(tpr),
            "fpr": float(fpr),
        })
    return pd.DataFrame(rows)


def save_threshold_sweep(y_true, y_score, out_dir: str) -> Tuple[str, Optional[str]]:
    ensure_dir(out_dir)
    df = threshold_sweep(y_true, y_score)
    csv_path = os.path.join(out_dir, "threshold_sweep.csv")
    df.to_csv(csv_path, index=False)

    # Plot precision/recall/F1 vs threshold
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["threshold"], df["precision"], label="precision")
    ax.plot(df["threshold"], df["recall"], label="recall")
    ax.plot(df["threshold"], df["f1"], label="f1")
    ax.set_xlabel("threshold")
    ax.legend()
    png_path = os.path.join(out_dir, "threshold_sweep.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return csv_path, png_path
