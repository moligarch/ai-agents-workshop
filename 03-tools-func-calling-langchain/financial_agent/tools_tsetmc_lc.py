"""
Section 3 — LangChain **Structured Tools** for TSETMC (BrsApi.ir)

OVERVIEW
--------
Defines **LangChain** StructuredTools for: (1) local time and (2) TSETMC snapshot
quotes via **BrsApi.ir**. Inputs are typed with **Pydantic** and outputs are
**JSON strings** so the model can consume observations deterministically.

ENV VARS
--------
- `BRSAPI_KEY`  : required for TSETMC network calls
- `BRSAPI_BASE` : optional; defaults to `https://BrsApi.ir/Api/Tsetmc`

SAFETY
------
- All network/parse issues become JSON with `type: "TOOL_ERROR" | "DATA_ERROR"`.
- No exceptions escape to the agent layer.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Literal, List
from dataclasses import dataclass
from datetime import datetime
import json
import os
import unicodedata

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# ------------------ Config ------------------

BRSAPI_BASE = os.getenv("BRSAPI_BASE", "https://BrsApi.ir/Api/Tsetmc").rstrip("/")
BRSAPI_KEY = os.getenv("BRSAPI_KEY")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# ------------------ Helpers ------------------

def _json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _need_key() -> str:
    return _json({"type": "TOOL_ERROR", "message": "Missing BRSAPI_KEY environment variable"})


def _normalize(s: Optional[str]) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False


def _get(path: str, params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not BRSAPI_KEY:
        return None, "BRSAPI_KEY env var not set"
    url = f"{BRSAPI_BASE}/{path.lstrip('/') }"
    q = {"key": BRSAPI_KEY}
    q.update(params)
    try:
        r = requests.get(url, params=q, headers=DEFAULT_HEADERS, timeout=10)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        try:
            data = r.json()
        except Exception:
            data = json.loads(r.text)
        return data, None
    except Exception as e:  # pragma: no cover
        return None, str(e)


# ------------------ Tool implementations ------------------

class _GetTimeArgs(BaseModel):
    """No arguments; kept for symmetry with structured tools."""
    pass


def _get_time(_: _GetTimeArgs) -> str:
    return _json({"type": "TIME", "iso": datetime.now().isoformat(timespec="seconds")})


class _GetQuoteArgs(BaseModel):
    symbol: str = Field(..., description="TSETMC symbol: Persian l18 (e.g., 'خودرو') or Latin/ISIN")
    price_field: Literal["pl", "pc", "py"] = Field(
        "pc", description="Price field: pl (last), pc (close), py (yesterday close)"
    )


def _resolve_l18(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    sym = _normalize(symbol)
    if not sym:
        return None, "empty symbol"
    if not _is_ascii(sym):
        return sym, None

    data, err = _get("AllSymbols.php", {"type": 1})
    if err or not isinstance(data, list):
        return None, err or "unexpected response"

    sym_u = sym.upper()
    for item in data:
        code4 = _normalize(str(item.get("code_4", ""))).upper()
        code5 = _normalize(str(item.get("code_5", ""))).upper()
        isin = _normalize(str(item.get("isin", ""))).upper()
        l18 = _normalize(str(item.get("l18", "")))
        if sym_u in (code4, code5, isin):
            return l18 or None, None
    for item in data:
        if _normalize(str(item.get("l18", ""))).upper() == sym_u:
            return _normalize(str(item.get("l18", ""))), None
    return None, f"symbol '{symbol}' not found in AllSymbols"


def _get_tsetmc_quote(symbol: str, price_field: Literal["pl", "pc", "py"] = "pc") -> str:
    """Fetch a TSETMC quote for a symbol via BrsApi.

    LangChain StructuredTool passes validated arguments as **kwargs**, so this
    function must accept `symbol` and `price_field` as keyword parameters.
    """
    if not BRSAPI_KEY:
        return _need_key()

    l18, err = _resolve_l18(symbol)
    if err and not l18:
        return _json({"type": "DATA_ERROR", "message": err})

    data, err = _get("Symbol.php", {"l18": l18})
    if err:
        return _json({"type": "DATA_ERROR", "message": err, "symbol": symbol})
    if not isinstance(data, dict):
        return _json({"type": "DATA_ERROR", "message": "unexpected response", "symbol": symbol})

    time = data.get("time")
    date = data.get("date")
    last = data.get("pl")
    close = data.get("pc")
    yclose = data.get("py")

    price_map = {"pl": last, "pc": close, "py": yclose}

    return _json({
        "type": "QUOTE",
        "source": "TSETMC/BrsApi",
        "symbol_input": symbol,
        "symbol_l18": l18,
        "as_of": f"{date} {time}".strip(),
        "price_field": price_field,
        "price": price_map.get(price_field),
        "currency": "IRR",
        "raw": {k: data.get(k) for k in ("pl", "pc", "py", "pmax", "pmin", "tno", "tvol", "tval")},
    })


# ------------------ Public: build tool list ------------------

def build_tools() -> List[StructuredTool]:
    """Return a list of LangChain StructuredTool objects ready to bind to the LLM."""
    get_time_tool = StructuredTool.from_function(
        name="get_time",
        description="Return the current local time as ISO-8601 string.",
        func=_get_time,
        args_schema=_GetTimeArgs,
    )

    get_tsetmc_quote_tool = StructuredTool.from_function(
        name="get_tsetmc_quote",
        description=(
            "Fetch TSETMC quote via BrsApi.ir. Args: symbol (Persian l18 or Latin code/ISIN), "
            "price_field ('pl'=last | 'pc'=close | 'py'=yesterday close)."
        ),
        func=_get_tsetmc_quote,
        args_schema=_GetQuoteArgs,
    )

    return [get_time_tool, get_tsetmc_quote_tool]
