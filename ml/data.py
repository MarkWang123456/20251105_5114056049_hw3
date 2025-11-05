from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd


DEFAULT_DATA_DIR = "data"
DEFAULT_DATA_FILE = "sms_spam_no_header.csv"


@dataclass
class DatasetPaths:
    path: str = os.path.join(DEFAULT_DATA_DIR, DEFAULT_DATA_FILE)
    urls: Tuple[str, ...] = ()  # Optional default URLs; can be provided via args/env


def ensure_dirs(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def try_download(target_path: str, urls: List[str]) -> bool:
    if not urls:
        return False
    ensure_dirs(target_path)
    for url in urls:
        try:
            urllib.request.urlretrieve(url, target_path)
            return True
        except Exception:
            continue
    return False


def ensure_dataset(paths: DatasetPaths, quiet: bool = False) -> str:
    """
    Ensure dataset exists locally; if missing, attempt to download from URLs or
    environment variable SPAM_DATA_URL. Returns absolute path where the dataset resides.
    """
    local_path = paths.path
    if os.path.exists(local_path):
        return local_path

    env_url = os.environ.get("SPAM_DATA_URL")
    url_list = list(paths.urls)
    if env_url:
        url_list.insert(0, env_url)

    if url_list and try_download(local_path, url_list):
        return local_path

    if not quiet:
        print(
            f"[info] Dataset not found at '{local_path}'. "
            "Set SPAM_DATA_URL or provide --data-url to enable auto-download. "
            "Alternatively, place the file manually. Expected name: 'sms_spam_no_header.csv'"
        )
    return local_path


_URL_RE = re.compile(r"https?://\S+")
_NUM_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _NUM_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_label(v) -> int:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"spam", "1"}:
            return 1
        if s in {"ham", "0"}:
            return 0
    try:
        iv = int(v)
        return 1 if iv == 1 else 0
    except Exception:
        return 0


def load_csv(path: str, has_header: Optional[bool] = None) -> pd.DataFrame:
    """
    Load dataset as DataFrame with columns [label, text]. If has_header is None,
    attempt to infer header presence; supports Packt's no-header CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if has_header is None:
        # Heuristic: read first line and infer whether it looks like a header
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        tokens = first.split(",")
        first_token = tokens[0].strip().lower() if tokens else ""
        looks_header = len(tokens) >= 2 and (first_token not in {"label", "spam", "ham", "0", "1"})
        has_header = looks_header

    if has_header:
        df = pd.read_csv(path)
        # Try to normalize column names
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        if "label" not in df.columns or "text" not in df.columns:
            # Try common variants
            label_col = next((c for c in cols if c in {"label", "category", "target"}), None)
            text_col = next((c for c in cols if c in {"text", "message", "sms", "content"}), None)
            if not label_col or not text_col:
                raise ValueError("CSV must contain columns 'label' and 'text' (or recognizable variants)")
            df = df[[label_col, text_col]].copy()
            df.columns = ["label", "text"]
    else:
        df = pd.read_csv(path, header=None, names=["label", "text"], encoding="utf-8", encoding_errors="ignore")

    # Normalize
    df["label"] = df["label"].map(normalize_label)
    df["text"] = df["text"].astype(str).map(clean_text)
    return df[["label", "text"]]


def load_xy(path: str) -> Tuple[pd.Series, pd.Series]:
    df = load_csv(path)
    return df["text"], df["label"]
