"""
Section 4 — Prompt templates for RAG synthesis

OVERVIEW
--------
Contains compact, explicit prompts for grounding answers in retrieved chunks.
We keep the template minimal and include **verbatim citations** (chunk ids) so
participants can see what supported the answer.
"""

from __future__ import annotations
from typing import List, Tuple


def build_context(retrieved: List[Tuple[float, str, int]]) -> str:
    parts = []
    for score, chunk, idx in retrieved:
        parts.append(f"[chunk:{idx} score:{score:.3f}]\n{chunk}\n")
    return "\n".join(parts)


def rag_prompt(question: str, context_blocks: str) -> str:
    return (
        "You are a careful assistant. Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"  # honesty
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_blocks}\n\n"
        "INSTRUCTIONS:\n"
        "- Be concise (<= 5 sentences).\n"
        "- Cite supporting chunk ids inline like [chunk:12].\n"
        "- If uncertain, say you don't know.\n"
    )
