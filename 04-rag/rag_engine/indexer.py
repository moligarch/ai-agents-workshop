"""
Section 4 — Indexer (PDF/text ingestion → chunk → embed → persist)

OVERVIEW
--------
Reads source documents (PDF or plain text), splits into language‑aware chunks,
computes embeddings (TF‑IDF by default), and persists an index as a **pickle**
file. Designed to be deterministic and portable for workshops.

INDEX FORMAT (pickled dict)
---------------------------
{
  'meta': {
      'lang': 'en'|'fa',
      'emb': 'tfidf'|'sbert',
      'chunk_size': int,
      'overlap': int,
  },
  'chunks': [str, ...],         # chunk texts in order
  'vectors': np.ndarray,        # shape (N, D)
  'tfidf_vectorizer': object | None  # only for tfidf (to transform queries)
}

FUNCTIONS
---------
- `ingest_pdf(path)` → text
- `ingest_text_file(path)` → text
- `build_index(text, lang, emb_name, chunk_size, overlap)` → index dict
- `save_index(index, path)` / `load_index(path)`
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import os
import pickle

import numpy as np
from pypdf import PdfReader

from chunking import chunk_text
from embeddings import TfidfEmbedding, SbertEmbedding


def ingest_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            texts.append(t)
    return "\n".join(texts)


def ingest_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _select_embedder(name: str):
    n = (name or "tfidf").lower()
    if n == "tfidf":
        return "tfidf", TfidfEmbedding()
    if n == "sbert":
        return "sbert", SbertEmbedding()
    raise ValueError(f"Unknown embedding backend: {name}")


def build_index(text: str, *, lang: str = "en", emb_name: str = "tfidf", chunk_size: int = 600, overlap: int = 120) -> Dict[str, Any]:
    text = (text or "").strip()
    sentences = chunk_text(text, lang=lang, chunk_size=chunk_size, overlap=overlap)
    chunks = sentences  # already grouped

    emb_key, embedder = _select_embedder(emb_name)
    vectors = embedder.fit_transform(chunks)

    index: Dict[str, Any] = {
        "meta": {
            "lang": lang,
            "emb": emb_key,
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
        "chunks": chunks,
        "vectors": vectors,
        "tfidf_vectorizer": getattr(embedder, "vectorizer", None),
    }
    return index


def save_index(index: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_index(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
