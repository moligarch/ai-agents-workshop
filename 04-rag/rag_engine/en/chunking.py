"""
English chunking: sentence split → word-bounded grouping with overlap.

This keeps dependencies minimal and works well for workshop-scale corpora.
"""

from __future__ import annotations
from typing import List

import re


_SENT_END = re.compile(r"([.!?])(\s+|$)")


def _sentence_split_en(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # naive sentence split, robust enough for docs without heavy NLP deps
    sents, start = [], 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        s = text[start:end].strip()
        if s:
            sents.append(s)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sents.append(tail)
    return sents


def _group_words(sents: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    word_count = 0

    for s in sents:
        w = s.split()
        if not w:
            continue
        if word_count + len(w) > chunk_size and buf:
            chunks.append(" ".join(buf))
            # overlap: keep the last `overlap` words
            tail = " ".join(" ".join(buf).split()[-overlap:]) if overlap > 0 else ""
            buf = [tail] if tail else []
            word_count = len(tail.split()) if tail else 0
        buf.append(s)
        word_count += len(w)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_text(text: str, *, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    sents = _sentence_split_en(text)
    return _group_words(sents, chunk_size, overlap)
