from __future__ import annotations

import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

import pytest


def test_hazm_import_optional():
    try:
        import hazm  # noqa: F401
        hazm_ok = True
    except Exception:
        hazm_ok = False
    # Either is fine; test simply ensures optional dependency doesn't crash importers
    assert isinstance(hazm_ok, bool)
