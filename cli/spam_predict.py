from __future__ import annotations

import argparse
import json
import os
from typing import List

import joblib


def load_texts_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description="Predict spam/ham for input text(s)")
    p.add_argument("--model", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text")
    group.add_argument("--file", help="File with one text per line")
    args = p.parse_args()

    pipe = joblib.load(args.model)

    texts: List[str]
    if args.text is not None:
        texts = [args.text]
    else:
        texts = load_texts_from_file(args.file)

    preds = pipe.predict(texts)

    # Score if available
    scores = None
    if hasattr(pipe.named_steps["model"], "decision_function"):
        scores = pipe.decision_function(texts)
    elif hasattr(pipe.named_steps["model"], "predict_proba"):
        scores = pipe.predict_proba(texts)[:, 1]

    for i, t in enumerate(texts):
        item = {"text": t, "label": int(preds[i])}
        if scores is not None:
            item["score"] = float(scores[i]) if hasattr(scores, "__len__") else float(scores)
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()

