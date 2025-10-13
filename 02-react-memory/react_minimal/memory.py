"""
Section 2 — Memory Primitives (short-term scratchpad, long-term JSON store)

OVERVIEW
--------
Provides two simple memory utilities:

1) ShortTermMemory
   - Keeps a list of step dicts: {"thought", "action", "observation"}.
   - `add(...)` appends a new step.
   - `to_scratchpad()` renders a compact, human-readable trace used by the
     prompt so the model can incorporate the latest Observation (ReAct).
   - This scratchpad is **not shown to end users**; it’s internal context.

2) LongTermMemory
   - A tiny key→string store persisted to disk (JSON file).
   - On init, it attempts to load from disk; `set(...)` writes and persists;
     `get(...)` returns a string or None.
   - Default path: `<this_dir>/memory.json`. You can override the path.

ERROR HANDLING & ROBUSTNESS
---------------------------
- LongTermMemory silently tolerates load/save errors and falls back to an
  in-memory dict (keeps demo code resilient for workshops).
- No concurrency/locking; if you need multi-process safety, add a file lock
  or migrate to a lightweight DB.

WHY THIS SPLIT?
---------------
- ShortTermMemory belongs in the **prompt loop** and is ephemeral (session-scoped).
- LongTermMemory is a **persistence** mechanism accessed via tools (notes_*),
  making it visible to the agent's tool layer without leaking chain-of-thought.

EXTENSIONS
----------
- Store richer values (JSON objects) instead of plain strings.
- Add TTL/expiry for notes.
- Introduce encryption-at-rest if notes carry secrets.
"""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import os

DEFAULT_MEM_PATH = os.path.join(os.path.dirname(__file__), "memory.json")

@dataclass
class ShortTermMemory:
    # Holds a list of {thought, action, observation} for the current session
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, thought: str, action: Optional[Dict[str, Any]] = None, observation: Optional[str] = None) -> None:
        self.trace.append({"thought": thought, "action": action, "observation": observation})

    def to_scratchpad(self) -> str:
        lines = []
        for i, step in enumerate(self.trace, 1):
            lines.append(f"Step {i}:\n  THOUGHT: {step['thought']}")
            if step.get("action"):
                lines.append(f"  ACTION: {json.dumps(step['action'], ensure_ascii=False)}")
            if step.get("observation") is not None:
                lines.append(f"  OBSERVATION: {step['observation']}")
        return "\n".join(lines)


class LongTermMemory:
    def __init__(self, path: str = DEFAULT_MEM_PATH):
        self.path = path
        self._data: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, text: str) -> None:
        self._data[key] = text
        self._save()