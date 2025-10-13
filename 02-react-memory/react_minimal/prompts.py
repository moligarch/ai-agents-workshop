"""
Section 2 — Prompt Templates for ReAct Agent

OVERVIEW
--------
Defines the textual prompts used by the agent:
- SYSTEM_PROMPT        : Core rules (strict JSON contract, step format).
- TOOL_SPEC_TEMPLATE   : Human-readable list of available tools + JSON schemas.
- USER_TEMPLATE        : Embeds the user's question and a structured scratchpad.

STRICT JSON CONTRACT
--------------------
At each step, the model must produce exactly one JSON object with:
  {
    "thought": str,
    "action": {"tool": str, "input": any}
    // OR
    "final": str
  }

Rules:
- Exactly one of {"action", "final"} must be present.
- `thought` is for internal use (short-term memory) and should remain brief.
  The agent never prints chain-of-thought to users.
- The `input` can be any JSON that satisfies the tool’s schema.

TOOL DISCOVERY
--------------
The agent fills TOOL_SPEC_TEMPLATE with entries from the tool registry:
- name
- description
- a small, JSON-schema-like dict (keys & required fields)
This helps the model select the correct tool and shape its input.

SCRATCHPAD
----------
`USER_TEMPLATE` includes the serialized short-term memory (steps with
Thought/Action/Observation) so the model can incorporate the latest Observation
on each turn (classic ReAct reasoning pattern).

EXTENSION IDEAS
---------------
- Add an explicit "plan" field to final responses for user-facing transparency.
- Create task-specific system prompts (e.g., math-heavy, scheduling-heavy).
- Switch to a formal function-calling interface if your model supports it.
"""


SYSTEM_PROMPT = (
    "You are a helpful ReAct-style agent. You must respond in STRICT JSON. "
    "At each turn, return exactly one of:\n"
    "  {\"thought\": str, \"action\": {\"tool\": str, \"input\": any}}\n"
    "or\n"
    "  {\"thought\": str, \"final\": str}\n\n"
    "Rules:\n"
    "- Prefer tool use when needed; otherwise produce a final answer.\n"
    "- If you used a tool previously, incorporate the latest OBSERVATION.\n"
    "- Keep 'thought' brief and do NOT reveal chain-of-thought to the user; it's for internal use.\n"
)

TOOL_SPEC_TEMPLATE = (
    "Available tools (names and JSON input schemas):\n{tool_summaries}\n\n"
    "Only call tools from this list. If inputs are invalid or a tool is missing, correct yourself.\n"
)

USER_TEMPLATE = (
    "USER QUESTION: {question}\n"
    "SCRATCHPAD so far:\n{scratchpad}\n"
)