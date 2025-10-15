"""
Section 4 — Retriever (cosine similarity top‑K search)

OVERVIEW
--------
Given a saved index (vectors + chunks), run a cosine‑similarity search to return
the top‑K most relevant chunks for a user query. Supports both TF‑IDF (needs
stored vectorizer to transform the query) and SBERT (model encodes query).

FUNCTIONS
---------
- `query_to_vector(index, text)` → np.ndarray (1, D)
- `search(index, query, k=4)` → list[(score: float, chunk: str, idx: int)]
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from embeddings import TfidfEmbedding, SbertEmbedding, cosine_sim


def query_to_vector(index, text: str) -> np.ndarray:
    meta = index.get("meta", {})
    emb = meta.get("emb")
    if emb == "tfidf":
        tfidf = index.get("tfidf_vectorizer")
        if tfidf is None:
            raise RuntimeError("TF‑IDF vectorizer missing from index")
        vec = tfidf.transform([text]).astype("float32").toarray()
        return vec
    elif emb == "sbert":
        # lazy construct SBERT encoder (index stores no model to keep pickle light)
        enc = SbertEmbedding()
        return enc.transform([text])
    else:
        raise ValueError(f"Unknown embedding backend: {emb}")


def search(index, query: str, k: int = 4) -> List[Tuple[float, str, int]]:
    vectors = index["vectors"]  # (N, D)
    qv = query_to_vector(index, query)  # (1, D)
    sims = cosine_sim(qv, vectors)[0]  # (N,)
    order = np.argsort(-sims)
    results: List[Tuple[float, str, int]] = []
    for i in order[: max(1, k)]:
        results.append((float(sims[i]), index["chunks"][i], int(i)))
    return results
