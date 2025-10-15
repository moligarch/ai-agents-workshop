from __future__ import annotations

import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

from chunking import split_sentences, make_chunks, chunk_text


def test_split_sentences_en_basic():
    t = "Hello world. This is a test! New line? OK."
    s = split_sentences(t, lang="en")
    assert len(s) == 4
    assert s[0].startswith("Hello world")


def test_make_chunks_overlap():
    sents = [f"s{i}" for i in range(10)]
    chunks = make_chunks(sents, chunk_size=3, overlap=1)
    # chunks are word-bounded; our sents are single words -> sizes predictable
    assert len(chunks) >= 3


def test_chunk_text_pipeline():
    t = "One two three. Four five six seven. Eight nine."
    chunks = chunk_text(t, lang="en", chunk_size=4, overlap=2)
    assert len(chunks) >= 2
