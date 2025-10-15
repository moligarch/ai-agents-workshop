"""
Section 4 — Embedding backends (TF‑IDF baseline, optional SBERT)

OVERVIEW
--------
Provides a small abstraction for embeddings so we can switch between a
**TF‑IDF** baseline (portable, no downloads) and an optional
**SentenceTransformer** backend when installed.

API
---
- `BaseEmbedding`: defines `fit_transform(texts)` and `transform(texts)`.
- `TfidfEmbedding`: scikit‑learn TF‑IDF; stores the vectorizer in the index for reuse.
- `SbertEmbedding`: lazy‑imports `sentence_transformers.SentenceTransformer`.

NOTES
-----
- We operate on **strings of chunks** (already split). For TF‑IDF this is ideal.
- Cosine similarity is used downstream; outputs are `numpy.ndarray`.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaseEmbedding:
    def fit_transform(self, texts: List[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def transform(self, texts: List[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class TfidfEmbedding(BaseEmbedding):
    def __init__(self, *, ngram_range=(1, 2), max_features: Optional[int] = 50_000):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).astype("float32").toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).astype("float32").toarray()


class SbertEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Install it or use --emb tfidf"
            ) from e
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype("float32")

    def transform(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype("float32")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity wrapper (row-wise)."""
    return cosine_similarity(a, b)
