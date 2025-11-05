from __future__ import annotations

from typing import Literal

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


ModelKind = Literal["lr", "nb", "svm"]


def get_model(kind: ModelKind):
    if kind == "lr":
        return LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced")
    if kind == "nb":
        return MultinomialNB()
    if kind == "svm":
        return LinearSVC(class_weight="balanced")
    raise ValueError(f"Unknown model kind: {kind}")

