"""
Section 2 — Tooling Layer (safe calculator, time, notes, registry)

OVERVIEW
--------
Implements a minimal set of tools that the ReAct agent can call:
- Safe calculator         : Evaluates arithmetic expressions via AST (no eval).
- time_now                : Returns current local time in ISO-8601 format.
- notes_write / notes_read: Tiny long-term memory over a JSON file.

Also defines:
- Tool dataclass          : name, description, JSON-ish schema, call function.
- build_tool_registry     : returns a registry dict[str, Tool] bound to a
                            LongTermMemory instance.
- call_tool               : validates payloads and executes a tool, returning
                            string results or TOOL_ERROR messages.

SAFE CALCULATOR
---------------
The calculator parses expressions with `ast.parse(..., mode="eval")` and only
allows a whitelist of node types:
- BinOps: +, -, *, /, //, %, **  (via operator module)
- Unary:  +, -
- Parentheses are implicit in AST structure.

Any names, calls, attributes, comprehensions, etc. are rejected with a clean
error message (returned as a string "CALC_ERROR: ..."). This design avoids
`eval` and keeps the tool secure and deterministic.

NOTES (LONG-TERM MEMORY)
------------------------
`notes_write(mem, key, text)` and `notes_read(mem, key)` manipulate a small
JSON key→string store (see memory.py) that persists across runs.

REGISTRY & VALIDATION
---------------------
- The registry gives the model structured discoverability: tool name, a brief
  description, and a JSON-schema-like `schema` describing required keys.
- `call_tool` enforces presence of required fields and returns errors as strings
  prefixed by "TOOL_ERROR: ..." rather than raising exceptions. This lets the
  model self-correct in the next step.

EXTENSIONS
----------
- Add more domain-specific tools (unit_convert, web_search, RAG retrieval).
- Replace schema validation with jsonschema if you need strict typing.
- Rate-limit or log tool calls for auditing in real apps.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict
import ast, operator as op
from datetime import datetime

from memory import LongTermMemory

# ------------------ Safe Calculator ------------------

_ALLOWED_BINOP = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow
}
_ALLOWED_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n  # type: ignore[no-any-return]
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOP:
        return _ALLOWED_BINOP[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_ast(node.operand))
    # Parentheses are represented implicitly by AST structure
    raise ValueError("Unsupported expression for calculator")


def safe_calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_ast(tree)
        return str(result)
    except Exception as e:  # pragma: no cover
        return f"CALC_ERROR: {e}"

# ------------------ Time ------------------

def time_now() -> str:
    return datetime.now().isoformat(timespec="seconds")

# ------------------ Notes (Long-term memory) ------------------

def notes_write(mem: LongTermMemory, key: str, text: str) -> str:
    mem.set(key, text)
    return "OK"

def notes_read(mem: LongTermMemory, key: str) -> str:
    return mem.get(key) or ""

# ------------------ Tool registry ------------------

@dataclass
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]  # JSON schema-like dict for human/model guidance
    func: Callable[..., str]


def build_tool_registry(mem: LongTermMemory) -> Dict[str, Tool]:
    return {
        "calculator": Tool(
            name="calculator",
            description="Evaluate a safe arithmetic expression (no variables or functions).",
            schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            func=lambda *, expression: safe_calculate(expression),
        ),
        "time_now": Tool(
            name="time_now",
            description="Return the current local time as ISO8601.",
            schema={"type": "object", "properties": {}},
            func=lambda **_: time_now(),
        ),
        "notes_write": Tool(
            name="notes_write",
            description="Write a note string under a key to long-term memory.",
            schema={"type": "object", "properties": {"key": {"type": "string"}, "text": {"type": "string"}}, "required": ["key", "text"]},
            func=lambda *, key, text: notes_write(mem, key, text),
        ),
        "notes_read": Tool(
            name="notes_read",
            description="Read the note under a key from long-term memory (empty if missing).",
            schema={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
            func=lambda *, key: notes_read(mem, key),
        ),
    }


def tool_summaries(reg: Dict[str, Tool]) -> str:
    lines = []
    for t in reg.values():
        lines.append(f"- {t.name}: {t.description}. Input schema: {t.schema}")
    return "\n".join(lines)


def call_tool(reg: Dict[str, Tool], name: str, payload: Dict[str, Any]) -> str:
    tool = reg.get(name)
    if not tool:
        return f"TOOL_ERROR: Unknown tool '{name}'"
    # Simple schema validation (keys only)
    required = tool.schema.get("required", [])
    for key in required:
        if key not in payload:
            return f"TOOL_ERROR: Missing required field '{key}'"
    try:
        return tool.func(**payload)
    except TypeError as e:
        return f"TOOL_ERROR: Bad arguments: {e}"
    except Exception as e:  # pragma: no cover
        return f"TOOL_ERROR: {e}"