"""
Section 4 — QA: offline extractive + optional OpenAI synthesis

Two answer modes:
1) Offline extractive summary (default).
2) LLM synthesis (optional, OpenAI-compatible).

Citations control (via CLI --citations):
- inline: keep [chunk:ID] inline.
- refs  : remove inline tags; append a 'Sources:' list with IDs and previews.
- none  : strip all [chunk:ID] tags.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import os
import re

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


def _preview(text: str, n: int = 160) -> str:
    t = " ".join(text.split())
    return (t[: n] + "…") if len(t) > n else t


def _strip_inline_tags(text: str) -> str:
    return re.sub(r"\[chunk:\d+\]", "", text)


def _append_sources_block(text: str, retrieved: List[Tuple[float, str, int]]) -> str:
    lines = ["", "Sources:"]
    # unique order by appearance (based on retrieved order)
    seen = set()
    for _, chunk, idx in retrieved:
        if idx in seen:
            continue
        seen.add(idx)
        lines.append(f"- [{idx}] {_preview(chunk, 140)}")
    return text.rstrip() + "\n" + "\n".join(lines)


def answer_offline(
    question: str,
    retrieved: List[Tuple[float, str, int]],
    *,
    citations: str = "inline",
    verbose: bool = False,
) -> str:
    if not retrieved:
        return "I couldn't find anything relevant in the index."
    if verbose:
        print(f"[rag][qa] offline mode; chunks={len(retrieved)} question='{_preview(question, 80)}' citations={citations}")

    if citations == "inline":
        bullets = [f"• [chunk:{idx} score:{score:.3f}] {_first_lines(chunk)}" for score, chunk, idx in retrieved]
        return "\n".join([f"Offline summary for: {question}"] + bullets)

    if citations == "refs":
        bullets = [f"• {_first_lines(chunk)}" for _, chunk, _ in retrieved]
        body = "\n".join([f"Offline summary for: {question}"] + bullets)
        return _append_sources_block(body, retrieved)

    # citations == "none"
    bullets = [f"• {_first_lines(chunk)}" for _, chunk, _ in retrieved]
    return "\n".join([f"Offline summary for: {question}"] + bullets)


def answer_with_llm(
    question: str,
    retrieved: List[Tuple[float, str, int]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    citations: str = "inline",
    verbose: bool = False,
) -> str:
    if not retrieved:
        return "I couldn't find anything relevant in the index."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        if verbose:
            print("[rag][qa] LLM unavailable (no API key or SDK); falling back to offline summary")
        return answer_offline(question, retrieved, citations=citations, verbose=verbose)

    model = model or os.getenv("MODEL", "gpt-4o-mini")
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    ctx = build_context(retrieved)
    prompt = rag_prompt(question, ctx, citations=citations)

    if verbose:
        print(f"[rag][qa] LLM synthesis model={model} base_url={base_url or 'default'} citations={citations}")
        print(f"[rag][qa] context_len chars={len(ctx)} chunks={len(retrieved)}")
        print(f"[rag][qa] prompt preview:\n{_preview(prompt, 400)}\n---")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""

    # Post-process to enforce the citations mode
    if citations == "none":
        content = _strip_inline_tags(content)
    elif citations == "refs":
        content = _strip_inline_tags(content)
        content = _append_sources_block(content, retrieved)

    if verbose:
        print("[rag][qa] LLM response received.")
    return content