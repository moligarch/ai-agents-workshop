# RAG Engine — English Track
This subfolder hosts **English-facing docs**. Core code remains in `rag_engine/`. Language-specific chunking lives in `rag_engine/en/chunking.py`.

## Quick start (one-liners)

**Index a PDF with verbose trace**

```bash
python cli.py index --pdf ./samples/handbook.pdf --lang en --out ./index.pkl --chunk-size 600 --chunk-overlap 120 --emb tfidf --verbose
```

**Ask a question (offline)**

```bash
python cli.py query --index ./index.pkl --q "What is React?" --top-k 4 --verbose
```

**Ask a question (LLM via OpenAI-compatible router)**

```bash
python cli.py query --index ./index.pkl --q "What is React?" --top-k 4 --llm --model gpt-4o-mini --base-url https://api.metisai.ir/openai/v1 --verbose
```

## How language dispatch works

* `rag_engine/chunking.py` routes calls to `rag_engine/en/chunking.py` when `--lang en` (or default).
* Persian uses `rag_engine/fa/chunking.py`. Shared modules remain in the package root.