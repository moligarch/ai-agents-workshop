# Section 3 — Tools & Function Calling with **LangChain** (TSETMC Edition)

This section rebuilds our function-calling agent on **LangChain**, using **StructuredTool**s and OpenAI function calling. The agent can fetch **TSETMC** (Tehran Stock Exchange) snapshot data via **BrsApi.ir**, and it’s portable, router-friendly (e.g., MetisAI), and traceable with `--verbose`.

---

## Why this section

We’ve seen manual ReAct loops (Section 2). Here you’ll:

* Use **LangChain**’s batteries-included tool-calling agent to reduce boilerplate.
* Define safe, typed tools with **Pydantic** schemas.
* Wire a real-world data source (TSETMC/BrsApi) into the agent.
* Keep it **portable**: clear env vars, minimal deps, offline-friendly tests.

---

## Learning objectives

By the end, participants can:

* Build a **tool-calling agent** with `create_tool_calling_agent` + `AgentExecutor`.
* Implement **StructuredTool**s and validate inputs via Pydantic.
* Call **TSETMC** endpoints and return structured JSON observations.
* Configure **router/base URL** (e.g., MetisAI) and use **`--verbose`** tracing.

---

## Repository layout (this section)

```
ai-agents-workshop/
  03-tools-func-calling-langchain/
    README.md
    financial_agent/
      __init__.py
      agent_lc.py
      tools_tsetmc_lc.py
      runner.py
      requirements.txt
      .env.example
      tests/
        test_tools_tsetmc_lc.py
        test_agent_lc.py
```

---

## Tools implemented (TSETMC via BrsApi.ir)

* `get_time()` → local ISO-8601 timestamp (no network).
* `get_tsetmc_quote(symbol: str, price_field: 'pl'|'pc'|'py'='pc')` →
  resolves Latin/ISIN or Persian `l18` and returns a snapshot with key fields (`pl` last, `pc` close, `py` yesterday). Currency: **IRR**.

**Env vars**

* `BRSAPI_KEY` (required for network tools)
* `BRSAPI_BASE` (optional; default `https://BrsApi.ir/Api/Tsetmc`)

---

## Setup

```bash
cd ai-agents-workshop/03-tools-func-calling-langchain/financial_agent
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env → set:
#   OPENAI_API_KEY=...
#   MODEL=gpt-4o-mini
#   BRSAPI_KEY=YourBrsApiKeyHere
# Optional router:
#   OPENAI_BASE_URL=https://api.metisai.ir/openai/v1
```

> Tests do **not** hit the network. They mock HTTP calls and/or avoid LLM calls entirely, so the suite is classroom-friendly.

---

## Quick start

Local time (tool call):

```bash
python runner.py --query "What time is it (ISO)?" --max-steps 2 --verbose
```

TSETMC quote (Persian `l18`):

```bash
python runner.py --query "Fetch pc (close) for نماد 'خودرو'" --max-steps 4 --verbose
```

TSETMC quote (Latin/ISIN → auto-resolve):

```bash
python runner.py --query "Get 'pl' (last) for IKCO" --max-steps 4 --verbose
```

Router + model override:

```bash
python runner.py --query "IKCO latest 'py' (yesterday)." \
  --base-url https://api.metisai.ir/openai/v1 \
  --model gpt-4o-mini --max-steps 3 --verbose
```

---

## Verbose tracing (`--verbose`)

* Prints agent decisions, tool calls, tool outputs (truncated), final message.
* **Never** prints chain-of-thought; it logs structure only.

Example (abridged):

```
[agent] tool_call name=get_tsetmc_quote args={"symbol": "IKCO", "price_field": "pc"}
[agent] observation={"type":"QUOTE","symbol_l18":"خودرو", ...}
[agent] final=Close price for IKCO (خودرو): 123,456 IRR.
```

---

## How it works (high level)

1. **Structured tools.** `tools_tsetmc_lc.py` defines Pydantic arg schemas and returns **JSON strings**.
2. **Agent.** `agent_lc.py` builds `ChatOpenAI` (router-friendly) → binds tools via `create_tool_calling_agent` → runs with `AgentExecutor`.
3. **CLI.** `runner.py` parses flags and prints the final answer.

---

## Design choices

* **LangChain** to lower boilerplate for tool calling and tracing.
* **Pydantic** to validate tool inputs (deterministic errors).
* **JSON string** tool outputs for easy model consumption.
* **Router/base-url** to work behind gateways like MetisAI.

---

## Troubleshooting

* `OPENAI_API_KEY` missing → set in `.env`.
* `BRSAPI_KEY` missing → TSETMC tool returns a clear error JSON; use time tool for offline.
* HTTP 4xx/5xx or symbol not found → check key/rate limits and exact symbol (`l18`).
* Model loops → increase `--max-steps`; use `--verbose` to see which step failed.

---

## Exercises

1. Add a candle tool: `get_tsetmc_candle(l18, timeframe)` and display OHLC later.
2. Add an in-memory cache for symbol resolution.
3. Add a `unit_convert` tool (°C/°F, km/mi) to showcase multiple tools.