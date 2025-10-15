# Section 2 — ReAct & Memory (Minimal Agent Lab)

This section upgrades our agent from Section 1 into an **LLM-driven ReAct agent** that interleaves **reasoning** with **tool use** and keeps **memory**. The implementation is portable and OpenAI-compatible, with support for custom router/base URLs and a `--verbose` trace mode.

---

## Learning objectives

By the end, participants can:

* Explain the **ReAct** pattern and why **structured JSON outputs** make agents reliable.
* Distinguish **short-term (scratchpad)** vs. **long-term (persistent)** memory.
* Implement a minimal **tool registry** with input validation and safe execution.
* Run a robust agent loop with **max steps**, **timeouts**, graceful failures, and **trace logs**.

---

## Repository layout (this section)

```
ai-agents-workshop/
  02-react-memory/
    README.md
    react_minimal/
      agent.py
      tools.py
      memory.py
      prompts.py
      runner.py
      requirements.txt
      .env.example
      tests/
        test_tools.py
        test_loop.py
```

---

## Tools implemented

* `calculator(expression: str) -> number` — Safe arithmetic via Python AST (supports +, -, *, /, //, %, **, parentheses, unary +/-). Rejects anything else.
* `time_now() -> str` — Returns ISO8601 timestamp (local time).
* `notes_write(key: str, text: str) -> str` — Writes a note to long-term store.
* `notes_read(key: str) -> str` — Reads a note; returns empty string if absent.

---

## Memory model

* **Short-term (scratchpad):** recent steps `{thought, action, observation}` kept in context (not shown to end-users by default).
* **Long-term:** lightweight JSON at `react_minimal/memory.json` (created on demand) accessed via `notes_*` tools.

---

## Setup

```bash
cd ai-agents-workshop/02-react-memory/react_minimal
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env → set OPENAI_API_KEY=...  MODEL=gpt-4o-mini  (or any OpenAI-compatible chat model)
```

> **Portability:** Unit tests and the tool layer run without network. The agent runtime uses an OpenAI-compatible API via the official `openai` Python SDK. In CI/offline, you can inject a mock LLM (see tests) and still exercise the loop.

---

## Using a Router (e.g., MetisAI)

You can direct the OpenAI SDK to a router/proxy by setting a base URL.

**Option A — via `.env`**

```env
OPENAI_API_KEY=your_router_key
OPENAI_BASE_URL=https://api.metisai.ir/openai/v1
MODEL=gpt-4o-mini   # or a model name supported by your router
```

**Option B — via CLI**

```bash
python3 runner.py \
  --query "What is 12*8? Then add 10." \
  --max-steps 3 \
  --base-url https://api.metisai.ir/openai/v1 \
  --model gpt-4o-mini
```

> The code prefers the CLI `--base-url` if provided; otherwise it looks for `OPENAI_BASE_URL` (or `OPENAI_API_BASE`). Use your router’s API key in `OPENAI_API_KEY`.

---

## Quick start

Single question with tools enabled:

```bash
python3 runner.py --query "What is 12*8? Then add 10." --max-steps 3
```

Add and read a note:

```bash
python3 runner.py --query "Save a note key=wifi with: SSID:Workshop-Guest; Pass:1234" --max-steps 4
python3 runner.py --query "What is our wifi password? Read the 'wifi' note." --max-steps 4
```

---

## Verbose tracing (`--verbose`)

You can enable terminal tracing of the agent loop.

```bash
python3 runner.py --query "What is 12*8? Then add 10." --max-steps 3 --verbose
```

**What gets logged**

* Step numbers.
* Whether the model returned an **action** or **final**.
* For actions: tool name and input payload.
* The resulting **observation** (truncated sensibly if very long).
* Non-JSON outputs and other recoverable errors.

> The agent intentionally **does not print chain-of-thought** text. Logs show structure (actions, observations, final) so you can debug deterministically without exposing internal reasoning.

Example (abridged):

```
[react] step=1 action tool=calculator input={'expression': '12*8'}
[react] step=1 observation=96
[react] step=2 action tool=calculator input={'expression': '96+10'}
[react] step=2 observation=106
[react] final=106
```

---

## How it works (high level)

1. **Prompt & schema.** `prompts.py` instructs the model to output **JSON** with fields: `thought`, `action` (object with `tool` + `input`), or `final` (string). Only one of `action` or `final` appears per turn.
2. **Loop.** `agent.py` sends messages (System rules + tool spec + user + scratchpad). If the model returns an `action`, we validate inputs, execute the tool, append an **Observation**, and continue. If it returns `final`, we stop.
3. **Safety.** Calculator uses AST and whitelisted nodes; unknown tools/invalid args produce a tool **error observation** so the model can self-correct.
4. **Memory.** Short-term lives in the scratchpad. Long-term is a simple JSON DB handled by `memory.py` and accessed via `notes_*` tools.

---

## Design choices

* **JSON over free-form tags** → easier parsing & robust error handling.
* **Step limit** (`--max-steps`) → prevents endless loops.
* **Tool registry** → single source of truth for name, input schema, and function.
* **No chain-of-thought exposure** → logs show structure only.

---

## Troubleshooting

* `OPENAI_API_KEY` missing → set in `.env` or env vars.
* Using a router? Set `OPENAI_BASE_URL` (or pass `--base-url`).
* Agent loops without stopping → increase `--max-steps` or refine the prompt (defaults are conservative).
* Calculator rejects input → only basic arithmetic is allowed (no names, calls, or attributes).

---

## Exercises

1. Add a tool `unit_convert(value, from_unit, to_unit)` with a tight whitelist (`C/F`, `km/mi`).
2. Add a **summary-on-exit**: prompt the model to write a 1-line memory and save it via `notes_write` before `final`.
3. Add a `plan` field in the final response for user-facing transparency (keep chain-of-thought private).