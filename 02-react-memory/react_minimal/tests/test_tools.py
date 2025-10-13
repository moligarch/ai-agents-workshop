from __future__ import annotations

import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

import pytest

from tools import safe_calculate, call_tool, build_tool_registry
from memory import LongTermMemory


def test_calculator_basic_ops():
    assert safe_calculate("2+3*4") == "14"
    assert safe_calculate("(2+3)*4") == "20"
    assert safe_calculate("2**3 + 1") == "9"


def test_calculator_blocks_names():
    # Disallow names/attrs/calls
    assert "Unsupported" in safe_calculate("__import__('os').system('echo hi')")


def test_notes_roundtrip(tmp_path):
    mem = LongTermMemory(path=tmp_path / "m.json")
    reg = build_tool_registry(mem)
    assert call_tool(reg, "notes_write", {"key": "wifi", "text": "pass:1234"}) == "OK"
    assert call_tool(reg, "notes_read", {"key": "wifi"}) == "pass:1234"
    assert call_tool(reg, "notes_read", {"key": "missing"}) == ""


def test_time_now():
    mem = LongTermMemory(path=str(tmp_path := __import__('tempfile').gettempdir() + "/m.json"))
    reg = build_tool_registry(mem)
    out = call_tool(reg, "time_now", {})
    assert "T" in out or ":" in out
