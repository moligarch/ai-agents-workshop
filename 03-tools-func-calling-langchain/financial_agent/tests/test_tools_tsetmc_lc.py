from __future__ import annotations

import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

import json
import types

import pytest

import tools_tsetmc_lc as mod


def test_need_key_error(monkeypatch):
    # Ensure no key is set
    monkeypatch.delenv("BRSAPI_KEY", raising=False)
    # reload module-level key
    import importlib
    importlib.reload(mod)

    # Call tool directly
    out = mod._get_tsetmc_quote(symbol="IKCO", price_field="pc")
    data = json.loads(out)
    assert data["type"] == "TOOL_ERROR"
    assert "Missing BRSAPI_KEY" in data["message"]


def test_resolve_l18_from_ascii(monkeypatch):
    # Pretend we have a key
    monkeypatch.setenv("BRSAPI_KEY", "dummy")
    import importlib
    importlib.reload(mod)

    # Fake AllSymbols response
    def fake_get(path, params):
        assert path == "AllSymbols.php"
        return [
            {"code_4": "IKCO", "code_5": "IKCO1", "isin": "IR1234567890", "l18": "خودرو"}
        ], None

    monkeypatch.setattr(mod, "_get", fake_get)

    l18, err = mod._resolve_l18("IKCO")
    assert err is None
    assert l18 == "خودرو"


def test_symbol_snapshot_success(monkeypatch):
    monkeypatch.setenv("BRSAPI_KEY", "dummy")
    import importlib
    importlib.reload(mod)

    def fake_get(path, params):
        if path == "AllSymbols.php":
            return [
                {"code_4": "IKCO", "code_5": "IKCO1", "isin": "IR1234567890", "l18": "خودرو"}
            ], None
        if path == "Symbol.php":
            return {
                "date": "1403/07/20",
                "time": "12:30:00",
                "pl": 123456,
                "pc": 120000,
                "py": 118000,
                "pmax": 130000,
                "pmin": 115000,
                "tno": 1000,
                "tvol": 2000000,
                "tval": 240000000000,
            }, None
        raise AssertionError("unexpected path: " + path)

    monkeypatch.setattr(mod, "_get", fake_get)

    out = mod._get_tsetmc_quote(symbol="IKCO", price_field="pc")
    data = json.loads(out)
    assert data["type"] == "QUOTE"
    assert data["symbol_l18"] == "خودرو"
    assert data["price_field"] == "pc"
    assert data["price"] == 120000
    assert data["currency"] == "IRR"


def test_get_time_tool():
    out = mod._get_time(mod._GetTimeArgs())
    data = json.loads(out)
    assert data["type"] == "TIME"
    assert "iso" in data