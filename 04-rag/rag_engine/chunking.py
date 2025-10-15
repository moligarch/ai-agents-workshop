"""
Section 4 — Chunking utilities (language-aware sentence grouping)

OVERVIEW
--------
Provides deterministic, dependency-light text chunking for RAG. We split text
into sentences, then group sentences into chunks of approximately N **words**
with an optional **overlap**. Persian (fa) tokenization uses `hazm` if
available; otherwise we fall back to a naive splitter.

KEY FUNCTIONS
-------------
- `split_sentences(text, lang)` → list[str]
- `make_chunks(sentences, chunk_size, overlap)` → list[str]
- `chunk_text(text, lang, chunk_size, overlap)` → convenience wrapper

DESIGN NOTES
------------
- Word counts are computed by a simple whitespace split for portability.
- For reproducibility in tests, no randomness is used.
"""

from __future__ import annotations
from typing import List
import re

# Optional Persian pipeline
try:  # pragma: no cover (import paths vary in CI)
    from hazm import Normalizer, sent_tokenize
    _HAZM_OK = True
except Exception:  # pragma: no cover
    Normalizer = None  # type: ignore
    sent_tokenize = None  # type: ignore
    _HAZM_OK = False


_EN_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, lang: str) -> List[str]:
    """Split `text` into sentences using simple, language-aware rules.

    - en: regex split on end punctuation + whitespace
    - fa: hazm Normalizer + sent_tokenize if available; else fallback to newline/period split
    """
    text = (text or "").strip()
    if not text:
        return []

    if (lang or "").lower().startswith("fa"):
        if _HAZM_OK:
            norm = Normalizer()  # type: ignore
            t = norm.normalize(text)
            return [s.strip() for s in sent_tokenize(t) if s.strip()]  # type: ignore
        # fallback: split on Persian/Latin periods and newlines
        parts = re.split(r"[\n\r.\u06D4]+\s*", text)
        return [p.strip() for p in parts if p.strip()]

    # default: English-ish splitting
    parts = _EN_SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(sentences: List[str], chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Group sentences into word-bounded chunks.

    Args:
        sentences: list of sentence strings
        chunk_size: approx max words per chunk
        overlap: number of words overlapping between successive chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    chunks: List[str] = []
    window: List[str] = []
    count = 0

    def _words(s: str) -> int:
        return len(s.split())

    for s in sentences:
        w = _words(s)
        if count + w <= chunk_size:
            window.append(s)
            count += w
            continue
        # flush current window
        if window:
            chunks.append(" ".join(window).strip())
        # start new window with overlap from previous
        if overlap > 0 and chunks:
            tail = chunks[-1].split()
            window = [" ".join(tail[-overlap:])] if tail else []
            count = _words(window[0]) if window else 0
        else:
            window, count = [], 0
        # add current sentence
        if w > chunk_size:  # very long single sentence; split by words
            words = s.split()
            for i in range(0, len(words), chunk_size):
                seg = " ".join(words[i : i + chunk_size])
                if seg:
                    chunks.append(seg)
            window, count = [], 0
        else:
            window.append(s)
            count += w

    if window:
        chunks.append(" ".join(window).strip())

    return chunks


def chunk_text(text: str, lang: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Convenience: sentences → chunks in one call."""
    return make_chunks(split_sentences(text, lang), chunk_size, overlap)
