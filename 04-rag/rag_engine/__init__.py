"""
Section 4 — Portable RAG Engine (package init)

This package implements a compact Retrieval-Augmented Generation pipeline with:
- Chunking (language-aware; optional hazm for Persian)
- Embeddings (TF-IDF baseline, optional SentenceTransformer)
- Indexing & persistence (pickle)
- Cosine similarity retrieval
- Offline extractive QA or optional OpenAI synthesis with router support

Each module contains a detailed top-of-file docstring.
"""
