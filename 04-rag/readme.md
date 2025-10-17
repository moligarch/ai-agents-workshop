# Section 4 — Grounding Agents with RAG (PDF + Persian hazm Mini-Module)

In this section you’ll build a **portable RAG engine** that can index PDFs and answer questions grounded in retrieved chunks. It’s designed for workshops: zero external services required by default (TF-IDF embeddings), with optional OpenAI synthesis and an optional **Persian (fa)** pipeline using **hazm**. Now includes **`--verbose` tracing** and a **clean bilingual structure**.

---

## Learning objectives

By the end, participants can:

* Explain the **RAG loop**: chunk → embed → store → retrieve → synthesize.
* Implement a compact, dependency-light RAG stack (TF-IDF baseline, optional SBERT).
* Ingest **English PDFs** and run **Persian** text through **hazm** normalizers.
* Run fully **offline tests**, with optional LLM synthesis for nicer answers.
* Use **`--verbose`** to trace indexing and querying end-to-end.

---

## Repository layout (this section)

```
ai-agents-workshop/
  04-rag/
    README.md                     # ← main overview (this file)
    rag_engine/
      __init__.py
      cli.py                     # index/query entrypoint (+ --verbose)
      indexer.py                 # ingest PDF/text, build & save index
      chunking.py                # language dispatcher facade (routes to en/fa)
      embeddings.py              # TF-IDF baseline (+ optional SBERT)
      retriever.py               # cosine top-K
      qa.py                      # offline summary or LLM synthesis
      prompts.py                 # compact prompt builder for LLM mode
      requirements.txt
      .env.example
      en/
        __init__.py
        chunking.py              # English sentence split + chunk grouping
        README.md                # English how-to
      fa/
        __init__.py
        chunking.py              # Persian sentence split; hazm if available
        README.md                # راهنمای فارسی
      tests/
        test_chunking.py
        test_index_retrieve.py
        test_qa_offline.py
        test_hazm_optional.py
```

> **Why this layout?** Shared logic stays at `rag_engine/`. Language-specific chunking moves into `rag_engine/en/` and `rag_engine/fa/`. The top-level `chunking.py` is a **dispatcher** that calls the right implementation based on `--lang`.

---

## Quick start (all one-liners)

### 0) Install

```bash
cd ai-agents-workshop/04-rag/rag_engine && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cp .env.example .env
```

> Put `OPENAI_API_KEY` in `.env` only if you want LLM synthesis. Default TF-IDF path is offline and lightweight. (You can switch to `--emb sbert` if `sentence-transformers` is installed.)

### 1) Index a PDF (English)

```bash
python cli.py index --pdf ./samples/handbook.pdf --lang en --out ./index.pkl --chunk-size 500 --chunk-overlap 120 --emb tfidf --verbose
```

### 2) Ask a question (offline synthesis)

```bash
python cli.py query --index ./index.pkl --q "What is React?" --top-k 4 --verbose
```

### 3) Ask a question (LLM synthesis — optional)

```bash
python cli.py query --index ./index.pkl --q "What is React?" --top-k 4 --llm --model gpt-4o --base-url https://api.metisai.ir/openai/v1 --citations refs --verbose
```

> `.env` may also include `OPENAI_BASE_URL` to omit `--base-url`.

### 4) Persian mini-module (hazm)

```bash
python cli.py index --pdf ./samples/fa_handbook.pdf --lang fa --out ./fa_index.pkl --chunk-size 400 --chunk-overlap 80 --emb tfidf --verbose
```

```bash
python cli.py query --index ./fa_index.pkl --q "answer this question and give result in structured format: عوامل موثر بر تأثیر بازاریابی گوشه ای بر وفاداری مشتریان در صنایع ورزشی با استفاده از تکنیکهای هوش مصنوعی چیست؟" --top-k 4 --llm --model gpt-4o-mini --base-url https://api.metisai.ir/openai/v1 --citations refs --verbose
```

---

## What `--verbose` prints

**Indexing:**

* Source detection (PDF/TEXT) and path; `--encoding` for text inputs
* Language hint, embedding backend, chunk/overlap settings
* Source size (chars/words), preview of raw text
* Final chunk count and vector shape; preview of first chunk

**Querying:**

* Index metadata (lang, emb, chunk config, total chunks)
* **Ranked top-K** results with cosine scores and short previews
* Answer mode: **OFFLINE** or **LLM**; if LLM: model & base URL
* Prompt preview and context stats (chars, chunk count) before LLM call

---

## How it works

1. **Chunking** — language-aware sentence splitting then word-bounded grouping with overlap:

   * `rag_engine/en/chunking.py` (English, no heavy deps)
   * `rag_engine/fa/chunking.py` (Persian; uses **hazm** if available; graceful fallback otherwise)
   * `rag_engine/chunking.py` routes based on `--lang` (dispatcher)
2. **Embeddings** — `embeddings.py` provides **TF-IDF** baseline; optional **SBERT** if installed.
3. **Index** — `indexer.py` reads PDF/text, chunkifies, computes embeddings, and persists an **index pickle** with vectors + metadata.
4. **Retriever** — `retriever.py` performs cosine similarity search (top-K) over vectors, returning `(score, chunk, idx)` tuples.
5. **QA** — `qa.py` composes an answer: either a simple **extractive summary** (offline) or an **LLM-synthesized** answer grounded in retrieved chunks. Router/base-URL supported.
6. **CLI** — `cli.py` is the user-facing entrypoint with **PDF auto-detect** for `--text`, `--encoding` for text files, and **`--verbose`** tracing for instruction & diagnostics.

---

## Design choices

* **Portable first**: TF-IDF, `pypdf`, scikit-learn; no GPU or big downloads.
* **Deterministic tests**: unit tests run offline; external I/O mocked where needed.
* **LLM optional**: controlled via `--llm`. In LLM mode, prompts cite chunk IDs.
* **Persian support**: if `hazm` exists we use it; otherwise we fall back gracefully.
* **Bilingual code layout**: shared core at root; language specifics in `en/` and `fa/`.

---

## Troubleshooting

* **PDF text empty** → PDF is likely scanned; OCR upstream or pass `--text` (with `--encoding` if needed).
* **LLM call fails** → set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if using a router), or omit `--llm`.
* **`hazm` missing** → install it (`pip install hazm`) or stay with English/ASCII tokenization.
* **Similarity poor on small corpora** → reduce `--chunk-size`, increase `--top-k`, or switch `--emb sbert` if you can download models.

---

## Exercises

1. Add citation rendering (e.g., show page numbers from PDF extraction).
2. Add a **rerank** step using `sklearn`’s `TfidfVectorizer` on the retrieved set only.
3. Wrap this RAG as a **LangChain tool** and plug it into the Section 3 agent.