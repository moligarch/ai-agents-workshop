"""
Persian chunking: uses hazm if installed for better sentence segmentation.
Falls back to a simple punctuation-based splitter if hazm is unavailable.

Why optional?
- Keeps the core package light and portable.
- Hazm + its NumPy pin can be installed only when Persian quality matters.
"""

from __future__ import annotations
from typing import List
import re

# Try hazm (optional)
try:  # pragma: no cover
    from hazm import Normalizer, SentenceTokenizer  # type: ignore
    _HAZM = True
except Exception:  # pragma: no cover
    _HAZM = False

# Simple Persian/Arabic sentence enders fallback
_FA_SENT_END = re.compile(r"([\.!\؟\!])(\s+|$)")


def _sentence_split_fa(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if _HAZM:
        norm = Normalizer()
        t = norm.normalize(text)
        tok = SentenceTokenizer()
        sents = tok.tokenize(t)
        return [s.strip() for s in sents if s.strip()]
    # fallback splitter
    sents, start = [], 0
    for m in _FA_SENT_END.finditer(text):
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
            # overlap
            tail = " ".join(" ".join(buf).split()[-overlap:]) if overlap > 0 else ""
            buf = [tail] if tail else []
            word_count = len(tail.split()) if tail else 0
        buf.append(s)
        word_count += len(w)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_text(text: str, *, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    # Slightly smaller defaults for FA due to script morphology/spacing
    sents = _sentence_split_fa(text)
    return _group_words(sents, chunk_size, overlap)
