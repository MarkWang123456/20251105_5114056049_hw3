from __future__ import annotations

from typing import Tuple, Optional
from typing_extensions import Literal

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


VectorizerKind = Literal["tfidf", "bow"]


def make_vectorizer(
    *,
    kind: VectorizerKind = "tfidf",
    ngram: Tuple[int, int] = (1, 1),
    max_features: Optional[int] = None,
    min_df: int = 1,
    stop_words: Optional[str] = None,
    token_pattern: Optional[str] = r"(?u)\b\w+\b",
):
    common_kwargs = dict(
        ngram_range=ngram,
        max_features=max_features,
        min_df=min_df,
        stop_words=stop_words,
        lowercase=False,  # already lower-cased in cleaning
        token_pattern=token_pattern,
    )
    if kind == "tfidf":
        return TfidfVectorizer(**common_kwargs)
    if kind == "bow":
        return CountVectorizer(**common_kwargs)
    raise ValueError(f"Unknown vectorizer kind: {kind}")
