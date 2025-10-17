"""
Language-dispatching facade for text chunking.

This module keeps the public function `chunk_text(...)` stable while routing
to language-specific implementations in `rag_engine/en/chunking.py` and
`rag_engine/fa/chunking.py`.

Why this split?
- Cleaner structure: shared code stays at package root; per-language logic
  lives in dedicated subpackages.
- Optional deps: Persian pipeline can import `hazm` without making it a hard
  dependency for English users.
"""

from __future__ import annotations
from typing import List

from en.chunking import chunk_text as _chunk_en

try:
    from fa.chunking import chunk_text as _chunk_fa  # type: ignore
except Exception:  # pragma: no cover
    _chunk_fa = None  # type: ignore


def chunk_text(
    text: str,
    *,
    lang: str = "en",
    chunk_size: int = 600,
    overlap: int = 120,
) -> List[str]:
    """
    Split text into sentence-aware, word-bounded chunks.

    Args:
        text: Source text (UTF-8 string).
        lang: 'en' or 'fa'. Defaults to 'en' on unknown values.
        chunk_size: Target words per chunk.
        overlap: Words overlapped between consecutive chunks.

    Returns:
        List[str]: chunked passages.
    """
    l = (lang or "en").strip().lower()
    if l.startswith("fa") and _chunk_fa is not None:
        return _chunk_fa(text, chunk_size=chunk_size, overlap=overlap)
    # Fallback to English implementation if fa not available
    return _chunk_en(text, chunk_size=chunk_size, overlap=overlap)
