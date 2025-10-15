from __future__ import annotations

import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

from indexer import build_index
from retriever import search


def test_index_and_search_tfidf():
    text = (
        "The handbook covers vacation policy and benefits.\n"
        "Holidays include Nowruz and other national days.\n"
        "Employees may request leave via the HR portal.\n"
    )
    idx = build_index(text, lang="en", emb_name="tfidf", chunk_size=8, overlap=2)
    hits = search(idx, "What are the holidays?", k=2)
    assert len(hits) >= 1
    # Ensure the top hit references Holidays/Nowruz
    top = hits[0][1].lower()
    assert "holiday" in top or "nowruz" in top
