"""
Section 3 — **LangChain** Tool-Calling Agent (with router & verbose tracing)

OVERVIEW
--------
Builds a LangChain agent that uses OpenAI **function calling** to invoke
StructuredTools. It binds the custom TSETMC tools (via BrsApi.ir) and provides
an easy CLI entry point. The agent is portable and router-friendly.

KEY PIECES
----------
- `ChatOpenAI` model (langchain-openai) with optional **base URL** for routers.
- `create_tool_calling_agent` to prime the LLM for tool calls.
- `AgentExecutor` to run the loop with `max_iterations` and `verbose` tracing.

ENV VARS
--------
- `OPENAI_API_KEY`         : required
- `MODEL`                  : optional (default: `gpt-4o-mini`)
- `OPENAI_BASE_URL`/`OPENAI_API_BASE` : optional router base (e.g., MetisAI)

HOW TRACING WORKS
-----------------
Set `verbose=True` on the executor (or pass `--verbose` via CLI). LangChain will
print the agent decisions, tool calls, tool outputs, and the final message.

EXTENSIONS
----------
- Add a custom callback handler to redact sensitive fields or log JSON to disk.
- Add more tools (unit conversion, FX rate) and domain prompts.
"""

from __future__ import annotations
import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools_tsetmc_lc import build_tools


def build_llm(model: str | None = None, base_url: str | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI LLM with optional router base URL support.

    We explicitly pass `api_key` (from env) and `base_url` so the OpenAI SDK
    never misses credentials even if environment loaders weren't triggered elsewhere.
    """
    model = model or os.getenv("MODEL", "gpt-4o-mini")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or environment.")
    return ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)


def build_agent(
    model: str | None = None,
    base_url: str | None = None,
    max_steps: int = 6,
    verbose: bool = False,
) -> AgentExecutor:
    tools = build_tools()
    llm = build_llm(model=model, base_url=base_url)

    system_msg = (
        "You are a helpful assistant. Use tools when they can improve factual accuracy. "
        "If a tool returns an error JSON (type=TOOL_ERROR/DATA_ERROR), explain it briefly and suggest a fix. "
        "Keep answers concise, mention prices with their currency where relevant."
    )

    # IMPORTANT: LangChain’s tool-calling agent expects an `agent_scratchpad` placeholder.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{input}"),
            # Executor will fill this with tool-call thoughts/observations
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_steps,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )
    return executor


def run_query(
    query: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    max_steps: int = 6,
    verbose: bool = False,
) -> str:
    agent = build_agent(model=model, base_url=base_url, max_steps=max_steps, verbose=verbose)
    result = agent.invoke({"input": query})
    # AgentExecutor returns a dict with an "output" key
    return str(result.get("output", ""))
