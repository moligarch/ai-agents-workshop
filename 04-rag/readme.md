# Section 4 — Grounding Agents with RAG (PDF + Persian hazm Mini-Module)

In this section you’ll build a **portable RAG engine** that can index PDFs and answer questions grounded in retrieved chunks. It’s designed for workshops: zero external services required by default (TF-IDF embeddings), with optional OpenAI synthesis and an optional **Persian (fa)** pipeline using **hazm**.

---

## Learning objectives

By the end, participants can:

* Explain the **RAG loop**: chunk → embed → store → retrieve → synthesize.
* Implement a compact, dependency-light RAG stack (TF-IDF baseline, optional SBERT).
* Ingest **English PDFs** and run **Persian** text through **hazm** normalizers.
* Run fully **offline tests**, with optional LLM synthesis for nicer answers.

---

## Repository layout (this section)

```
ai-agents-workshop/
  04-rag/
    README.md
    rag_engine/
      __init__.py
      cli.py
      indexer.py
      chunking.py
      embeddings.py
      retriever.py
      qa.py
      prompts.py
      requirements.txt
      .env.example
      tests/
        test_chunking.py
        test_index_retrieve.py
        test_qa_offline.py
        test_hazm_optional.py
```

---

## Quick start

### 0) Install

```bash
cd ai-agents-workshop/04-rag/rag_engine
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # (optional) put OPENAI_API_KEY if you want LLM synthesis
```

> The default path uses **TF-IDF** embeddings → no heavy model downloads, offline friendly. You can switch to `--emb sbert` if `sentence-transformers` is installed.

### 1) Index a PDF (English)

```bash
python cli.py index \
  --pdf ./samples/handbook.pdf \
  --lang en \
  --out ./index.pkl \
  --chunk-size 600 --chunk-overlap 120 \
  --emb tfidf
```

### 2) Ask a question (offline synthesis)

```bash
python cli.py query --index ./index.pkl \
  --q "What are the company holidays?" \
  --top-k 4
```

### 3) Ask a question (LLM synthesis — optional)

```bash
# .env must contain OPENAI_API_KEY (and optionally OPENAI_BASE_URL for routers)
python cli.py query --index ./index.pkl \
  --q "What are the company holidays?" \
  --top-k 4 \
  --llm --model gpt-4o-mini --base-url https://api.metisai.ir/openai/v1
```

### 4) Persian mini-module (hazm)

```bash
python cli.py index --pdf ./sample/fa1.pdf --lang fa --out ./fa_index.pkl --chunk-size 400 --chunk-overlap 80 --emb tfidf

python cli.py query --index ./fa_index.pkl   --q "answer this question and give result in structured format: عوامل موثر بر تأثیر بازاریابی گوشه ای بر وفاداری مشتریان در صنایع ورزشی با استفاده از تکنیکهای هوش مصنوعی چیست؟"   --top-k 4  --llm
```

---

## How it works

1. **Chunking** (`chunking.py`) — language-aware sentence splitting with grouping into fixed-size token buckets (words). Persian pipeline uses **hazm Normalizer + sent_tokenize** when available.
2. **Embeddings** (`embeddings.py`) — TF-IDF baseline (portable). Optional `SentenceTransformer` if installed. Both expose a common interface for `fit_transform()` and `transform()`.
3. **Index** (`indexer.py`) — reads PDF or text, chunkifies, computes embeddings, and persists an **index pickle** with vectors + metadata.
4. **Retriever** (`retriever.py`) — cosine similarity search (top-K) over vectors, returning chunks with scores.
5. **QA** (`qa.py`) — composes an answer: either a simple extractive summary (offline) or an LLM-synthesized answer grounded in retrieved chunks. Router/base-URL is supported for OpenAI-compatible endpoints.

---

## Design choices

* **Portable first**: TF-IDF, `pypdf`, scikit-learn; no GPU or big downloads.
* **Deterministic tests**: unit tests run offline, mock I/O when needed.
* **LLM optional**: controlled via `--llm`. When enabled, we cite chunk boundaries in the prompt.
* **Persian support**: if `hazm` available we use it; otherwise we fall back gracefully.

---

## Troubleshooting

* PDF text empty → some PDFs are image-based; add OCR upstream or pass `--text`.
* LLM call fails → set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if using a router), or omit `--llm`.
* `hazm` missing → install it (`pip install hazm`) or stick to English/ASCII tokenization.
* Similarity poor on small corpora → reduce `--chunk-size`, increase `--top-k`, or switch `--emb sbert` if you can download models.

---

## Exercises

1. Add citation rendering (highlight source page numbers).
2. Add a **Rerank** step using `sklearn`’s `TfidfVectorizer` on the retrieved set.
3. Plug this RAG into your Section 3 LangChain agent as a custom tool.