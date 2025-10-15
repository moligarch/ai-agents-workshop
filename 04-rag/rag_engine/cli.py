"""
Section 4 — CLI for the portable RAG engine

OVERVIEW
--------
Provides two subcommands:
- `index` — ingest PDF or text file, build embeddings, and persist `index.pkl`.
- `query` — load an index and answer a question (offline summary or `--llm`).

Router/base URL and model are supported for LLM synthesis (OpenAI‑compatible).
All Python files include top‑file docstrings for workshop clarity.
"""

from __future__ import annotations
import argparse
import os

from indexer import ingest_pdf, ingest_text_file, build_index, save_index, load_index
from retriever import search
from qa import answer_offline, answer_with_llm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    pi = sub.add_parser("index", help="Build and save an index from a PDF or text file")
    src = pi.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Path to a PDF file to ingest")
    src.add_argument("--text", help="Path to a plain‑text file to ingest (UTF‑8)")
    pi.add_argument("--lang", default="en", help="Language hint: en | fa")
    pi.add_argument("--out", required=True, help="Output index path (e.g., ./index.pkl)")
    pi.add_argument("--emb", default="tfidf", choices=["tfidf", "sbert"], help="Embedding backend")
    pi.add_argument("--chunk-size", type=int, default=600)
    pi.add_argument("--chunk-overlap", type=int, default=120)

    # query
    pq = sub.add_parser("query", help="Load an index and answer a question")
    pq.add_argument("--index", required=True, help="Path to a saved index .pkl")
    pq.add_argument("--q", required=True, help="User question")
    pq.add_argument("--top-k", type=int, default=4)
    pq.add_argument("--llm", action="store_true", help="Use OpenAI LLM synthesis if API key available")
    pq.add_argument("--model", default=None, help="Model override for LLM mode")
    pq.add_argument("--base-url", default=None, help="OpenAI‑compatible base URL (router)")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "index":
        text = ingest_pdf(args.pdf) if args.pdf else ingest_text_file(args.text)
        if not text.strip():
            print("[rag] Warning: source text is empty; PDF may be image‑based.")
        index = build_index(text, lang=args.lang, emb_name=args.emb, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
        save_index(index, args.out)
        print(f"[rag] Indexed {len(index['chunks'])} chunks → {args.out}")
        return 0

    if args.cmd == "query":
        index = load_index(args.index)
        hits = search(index, args.q, k=args.top_k)
        if args.llm:
            print(answer_with_llm(args.q, hits, model=args.model, base_url=args.base_url))
        else:
            print(answer_offline(args.q, hits))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
