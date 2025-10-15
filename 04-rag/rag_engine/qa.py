"""
Section 4 — QA: offline extractive + optional OpenAI synthesis

OVERVIEW
--------
Two answer modes:
1) **Offline extractive** (default) — returns the top chunks with a short
   heuristic summary.
2) **LLM synthesis** (optional) — if `OPENAI_API_KEY` is set or passed via env,
   call an OpenAI-compatible chat model to synthesize a grounded answer. Router
   base URLs are supported.

PUBLIC API
----------
- `answer_offline(question, retrieved)` → str
- `answer_with_llm(question, retrieved, model, base_url)` → str
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import os
import textwrap

from dotenv import load_dotenv
load_dotenv()

try:  # lazy import; only used if --llm
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from prompts import build_context, rag_prompt


def _first_lines(s: str, n: int = 2) -> str:
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return " ".join(lines[:n])


def answer_offline(question: str, retrieved: List[Tuple[float, str, int]]) -> str:
    if not retrieved:
        return "I couldn't find anything relevant in the index."
    # Simple heuristic: echo top passages with a gist line
    bullets = []
    for score, chunk, idx in retrieved:
        bullets.append(f"• [chunk:{idx} score:{score:.3f}] {_first_lines(chunk)}")
    return "\n".join([f"Offline summary for: {question}"] + bullets)


def answer_with_llm(question: str, retrieved: List[Tuple[float, str, int]], *, model: Optional[str] = None, base_url: Optional[str] = None) -> str:
    if not retrieved:
        return "I couldn't find anything relevant in the index."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        # fall back gracefully
        return answer_offline(question, retrieved)

    model = model or os.getenv("MODEL", "gpt-4o-mini")
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    ctx = build_context(retrieved)
    prompt = rag_prompt(question, ctx)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""
