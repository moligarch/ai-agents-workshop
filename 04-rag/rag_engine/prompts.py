"""
Section 4 — Prompt builders for RAG

- build_context(retrieved): format retrieved chunks with [chunk:ID] labels.
- rag_prompt(question, context, citations): instructs model on how to cite.
"""
from __future__ import annotations
from typing import List, Tuple


def build_context(retrieved: List[Tuple[float, str, int]]) -> str:
    """
    Build a context block with explicit [chunk:ID] labels so the model
    can cite if needed. We keep the chunk text faithful.
    """
    lines = []
    for _, chunk, idx in retrieved:
        lines.append(f"[chunk:{idx}]\n{chunk}\n")
    return "\n".join(lines)


def rag_prompt(question: str, context: str, *, citations: str = "inline") -> str:
    """
    Construct a compact instruction prompt. 'citations' controls how the
    model should present sources in the **final** answer:
      - 'inline' : cite using [chunk:ID] inline when making claims.
      - 'refs'   : DO NOT include [chunk:ID] inline; after the answer,
                   add a 'Sources:' list with the chunk IDs you used.
      - 'none'   : DO NOT include any [chunk:ID] tags in the final answer.
    """
    if citations not in ("inline", "refs", "none"):
        citations = "inline"

    cite_clause = {
        "inline": "Cite sources inline using the labels like [chunk:ID] when making factual claims.",
        "refs":   "Do NOT include [chunk:ID] inline. After the answer, add a 'Sources:' list that names the chunk IDs you used.",
        "none":   "Do NOT include any [chunk:ID] tags or source IDs in the final answer.",
    }[citations]

    return f"""You are answering a question using ONLY the provided context.
If the answer is not contained in the context, say you don't know.
Be concise and clear. {cite_clause}

Context:
{context}

Question:
{question}

Answer:"""
