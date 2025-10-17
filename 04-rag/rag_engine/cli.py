"""
Section 4 — CLI for the portable RAG engine

OVERVIEW
--------
Provides two subcommands:
- `index` — ingest PDF or text file, build embeddings, and persist `index.pkl`.
- `query` — load an index and answer a question (offline summary or `--llm`).

Router/base URL and model are supported for LLM synthesis (OpenAI-compatible).

VERBOSE MODE
------------
Use `--verbose` to print step-by-step traces for both indexing and querying.

CITATIONS MODE
--------------
`--citations` controls how citations appear in the final answer:
- inline  : keep [chunk:ID] inline in sentences (default)
- refs    : DO NOT show inline tags; append a "Sources:" section listing IDs
- none    : remove citations altogether from the final answer
"""
from __future__ import annotations
import argparse
import os

from indexer import ingest_pdf, ingest_text_file, build_index, save_index, load_index
from retriever import search
from qa import answer_offline, answer_with_llm


def _looks_like_pdf(path: str) -> bool:
    if path.lower().endswith(".pdf"):
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(5)
        return head.startswith(b"%PDF-")
    except Exception:
        return False


def _preview(text: str, n: int = 120) -> str:
    t = " ".join(text.split())
    return (t[: n] + "…") if len(t) > n else t


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    pi = sub.add_parser("index", help="Build and save an index from a PDF or text file")
    src = pi.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Path to a PDF file to ingest")
    src.add_argument("--text", help="Path to a plain-text file to ingest (UTF-8 by default)")
    pi.add_argument("--encoding", default="utf-8", help="Text file encoding for --text (default: utf-8)")
    pi.add_argument("--lang", default="en", help="Language hint: en | fa")
    pi.add_argument("--out", required=True, help="Output index path (e.g., ./index.pkl)")
    pi.add_argument("--emb", default="tfidf", choices=["tfidf", "sbert"], help="Embedding backend")
    pi.add_argument("--chunk-size", type=int, default=600)
    pi.add_argument("--chunk-overlap", type=int, default=120)
    pi.add_argument("--verbose", action="store_true", help="Print step-by-step indexing trace")

    # query
    pq = sub.add_parser("query", help="Load an index and answer a question")
    pq.add_argument("--index", required=True, help="Path to a saved index .pkl")
    pq.add_argument("--q", required=True, help="User question")
    pq.add_argument("--top-k", type=int, default=4)
    pq.add_argument("--llm", action="store_true", help="Use OpenAI LLM synthesis if API key available")
    pq.add_argument("--model", default=None, help="Model override for LLM mode")
    pq.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (router)")
    pq.add_argument("--citations", default="refs", choices=["inline", "refs", "none"],
                    help="How to render citations: inline | refs | none (default: refs)")
    pq.add_argument("--verbose", action="store_true", help="Print step-by-step query/answer trace")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "index":
        # Ingest source with helpful auto-detection
        if args.pdf:
            if args.verbose:
                print(f"[rag][index] source=PDF path={args.pdf}")
            text = ingest_pdf(args.pdf)
        else:
            if _looks_like_pdf(args.text):
                print("[rag] Notice: --text points to a PDF; switching to PDF ingestion.")
                if args.verbose:
                    print(f"[rag][index] source=PDF path={args.text}")
                text = ingest_pdf(args.text)
            else:
                if args.verbose:
                    print(f"[rag][index] source=TEXT path={args.text} encoding={args.encoding}")
                try:
                    text = ingest_text_file(args.text, encoding=args.encoding)
                except UnicodeDecodeError:
                    print(
                        f"[rag] Decode error for {args.text} with encoding '{args.encoding}'. "
                        "Use --encoding to set the correct charset (e.g., latin-1), or use --pdf if this is a PDF."
                    )
                    return 2
                except ValueError as e:
                    print(f"[rag] {e}")
                    return 2

        if not text.strip():
            print("[rag] Warning: source text is empty; PDF may be image-based.")
        else:
            if args.verbose:
                chars = len(text)
                words = len(text.split())
                print(f"[rag][index] lang={args.lang} emb={args.emb} chunk={args.chunk_size}/{args.chunk_overlap}")
                print(f"[rag][index] source_size chars={chars} words≈{words} preview='{_preview(text)}'")

        index = build_index(
            text,
            lang=args.lang,
            emb_name=args.emb,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        if args.verbose:
            n = len(index["chunks"])
            vecs = index["vectors"]
            shape = getattr(vecs, "shape", None)
            print(f"[rag][index] chunks={n} vectors_shape={shape}")
            if n:
                print(f"[rag][index] first_chunk preview='{_preview(index['chunks'][0])}'")

        save_index(index, args.out)
        print(f"[rag] Indexed {len(index['chunks'])} chunks → {args.out}")
        return 0

    if args.cmd == "query":
        if args.verbose:
            print(f"[rag][query] index={args.index}")
        index = load_index(args.index)
        meta = index.get("meta", {})
        if args.verbose:
            print(f"[rag][query] meta lang={meta.get('lang')} emb={meta.get('emb')} "
                  f"chunk={meta.get('chunk_size')}/{meta.get('overlap')} total_chunks={len(index.get('chunks', []))}")
            print(f"[rag][query] question='{args.q}' top_k={args.top_k} citations={args.citations}")

        hits = search(index, args.q, k=args.top_k)
        if args.verbose:
            print("[rag][query] hits (ranked):")
            for rank, (score, chunk, idx) in enumerate(hits, 1):
                print(f"  #{rank} idx={idx} score={score:.3f} preview='{_preview(chunk)}'")

        if args.llm:
            if args.verbose:
                print(f"[rag][query] mode=LLM model={args.model or os.getenv('MODEL', 'gpt-4o-mini')} "
                      f"base_url={args.base_url or os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE')}")
            print(answer_with_llm(args.q, hits, model=args.model, base_url=args.base_url,
                                  citations=args.citations, verbose=args.verbose))
        else:
            if args.verbose:
                print("[rag][query] mode=OFFLINE")
            print(answer_offline(args.q, hits, citations=args.citations, verbose=args.verbose))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
