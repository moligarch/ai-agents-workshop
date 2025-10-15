from __future__ import annotations

import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

from qa import answer_offline


def test_answer_offline_shapes_response():
    retrieved = [
        (0.9, "Holidays include Nowruz and national days.", 0),
        (0.7, "Vacation policy allows carryover.", 1),
    ]
    out = answer_offline("What are the holidays?", retrieved)
    assert "Offline summary" in out
    assert "[chunk:0" in out
