"""
Section 2 — ReAct Agent (LLM loop, tools, memory, tracing)

OVERVIEW
--------
This module implements a minimal, production-friendly **ReAct** agent:
it interleaves model **reasoning** with **Actions** (tool calls), feeds back
**Observations**, and terminates with a **final** answer — all while keeping
a short-term scratchpad and optional long-term notes. It also supports
OpenAI-compatible routers (e.g., MetisAI) and a `--verbose` trace mode.

KEY CLASSES
-----------
- LLM:          Minimal interface that returns a single content string.
- OpenAIChat:   LLM wrapper built on the official `openai` SDK (>=1.0).
                Supports:
                  * API key via OPENAI_API_KEY (required)
                  * Model name via MODEL (optional; defaults "gpt-4o-mini")
                  * Router base URL via OPENAI_BASE_URL / OPENAI_API_BASE
                    or the constructor's `base_url` kwarg.
- AgentConfig:  Simple dataclass for agent controls (max_steps, verbose).
- ReActAgent:   The core agent with a Thought→Action→Observation loop.

ENVIRONMENT VARIABLES
---------------------
- OPENAI_API_KEY      : Required (use your router’s key if behind a proxy)
- MODEL               : Optional, e.g., "gpt-4o-mini"
- OPENAI_BASE_URL     : Optional router base (e.g., https://api.metisai.ir/openai/v1)

LOOP LOGIC (high level)
-----------------------
1) Build messages:
   - system: hard rules for strict JSON, plus dynamic tool registry summary
   - user:   user query + short-term scratchpad (previous steps)
2) Call the LLM. Parse the first JSON object from the response.
   JSON schema per step (enforced by prompt):
     {
       "thought": str,
       "action": {"tool": str, "input": any}   # OR
       "final": str
     }
   Exactly one of {"action", "final"} must be present.
3) If `action`:
   - Validate and execute the tool via `call_tool(registry, name, payload)`.
   - Append an Observation string (or TOOL_ERROR) to scratchpad.
   - Continue to the next step.
4) If `final`: return it to the caller and stop.
5) If output is malformed: record a recoverable error in scratchpad and retry.

TRACING
-------
When AgentConfig.verbose=True:
- Each step logs: action tool & payload, observation, or final output.
- Logs intentionally **do not** print chain-of-thought text (only structure).

DESIGN NOTES
------------
- Short-term memory is a list of {thought, action, observation} entries.
  It is included in the prompt as a structured scratchpad.
- Long-term memory is a simple JSON key→string store (see memory.py) used by
  the `notes_*` tools for persistence.
- Tool calls never raise to the agent; they return strings, including errors
  prefixed with "TOOL_ERROR: ..." so the model can self-correct.
- A hard `max_steps` cap prevents endless loops.

EXTENSION IDEAS
---------------
- Add more tools (unit conversion, web search, RAG).
- Add a "summary-on-exit": on final, write a 1-line memory via notes_write.
- Support streaming model responses for UI.
- Replace JSON extraction with a strict JSON mode or function-calling API.

TESTING & PORTABILITY
---------------------
- Tests use a `MockLLM` that feeds deterministic JSON strings (no network).
- The OpenAI SDK is imported lazily to keep tests light.
- Works cross-platform on Python 3.10/3.11.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os

from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT, TOOL_SPEC_TEMPLATE, USER_TEMPLATE
from memory import ShortTermMemory, LongTermMemory
from tools import build_tool_registry, tool_summaries, call_tool

load_dotenv()

# ------------------ LLM interface ------------------

class LLM:
    def complete(self, messages: List[Dict[str, str]]) -> str:  # returns content string
        raise NotImplementedError


class OpenAIChat(LLM):
    """
    OpenAI-compatible chat client.

    Supports routers/proxies (e.g., MetisAI) via either:
      - passing base_url to the constructor, or
      - setting OPENAI_BASE_URL / OPENAI_API_BASE in the environment.

    Required:
      - OPENAI_API_KEY  (use the router's key if you use a router)
    Optional:
      - MODEL (defaults to gpt-4o-mini)
      - OPENAI_BASE_URL (when using a router)
    """
    def __init__(self, model: Optional[str] = None, *, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import OpenAI  # lazy import to keep tests light
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or environment.")

        # Prefer explicit arg, then env vars (both common spellings)
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        # If a base_url is provided, point the SDK to that router; otherwise use default
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")

    def complete(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0)
        return resp.choices[0].message.content or "{}"


# ------------------ ReAct Agent ------------------

@dataclass
class AgentConfig:
    max_steps: int = 5
    verbose: bool = False


class ReActAgent:
    def __init__(self, llm: Optional[LLM] = None, config: Optional[AgentConfig] = None):
        self.llm = llm or OpenAIChat()
        self.config = config or AgentConfig()
        self.long_mem = LongTermMemory()
        self.tools = build_tool_registry(self.long_mem)
        self.short_mem = ShortTermMemory()

    # simple logger
    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[react] {msg}")

    def _build_messages(self, question: str) -> List[Dict[str, str]]:
        tool_spec = TOOL_SPEC_TEMPLATE.format(tool_summaries=tool_summaries(self.tools))
        scratchpad = self.short_mem.to_scratchpad()
        return [
            {"role": "system", "content": SYSTEM_PROMPT + "\n" + tool_spec},
            {"role": "user", "content": USER_TEMPLATE.format(question=question, scratchpad=scratchpad)},
        ]

    @staticmethod
    def _parse_json_block(text: str) -> Dict[str, Any]:
        # Extract first JSON object from text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return json.loads(text[start : end + 1])

    def run(self, question: str) -> str:
        for step in range(1, self.config.max_steps + 1):
            messages = self._build_messages(question)
            raw = self.llm.complete(messages)
            try:
                obj = self._parse_json_block(raw)
            except Exception as e:
                self._log(f"step={step} non_json_output error={e}")
                self.short_mem.add(thought=f"Model returned non-JSON: {e}")
                continue

            thought = obj.get("thought", "")
            action = obj.get("action")
            final = obj.get("final")

            if action and final:
                self._log(f"step={step} invalid both action+final present")
                self.short_mem.add(thought, observation="Tool+Final both present; choose one only.")
                continue

            if action:
                tool_name = action.get("tool")
                payload = action.get("input", {}) or {}
                self._log(f"step={step} action tool={tool_name} input={payload}")
                observation = call_tool(self.tools, tool_name, payload)
                self._log(f"step={step} observation={str(observation)[:200]}")
                self.short_mem.add(thought=thought, action=action, observation=observation)
                continue

            if isinstance(final, str):
                self._log(f"final={final}")
                return final

            self._log(f"step={step} invalid no action/final")
            self.short_mem.add(thought, observation="Invalid output (no action/final)")

        self._log("terminated: step limit reached")
        return "I couldn't complete the task within the step limit."
