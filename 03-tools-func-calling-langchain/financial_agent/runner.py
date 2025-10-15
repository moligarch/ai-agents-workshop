"""
Section 3 — CLI Runner (LangChain tool-calling agent)

OVERVIEW
--------
Thin command-line wrapper around the LangChain agent. It reads CLI flags (query,
model, base-url, steps, verbose), builds the agent, runs a single query, and
prints the final answer to stdout.

FLAGS
-----
--query       : required user question/instruction.
--max-steps   : cap on iterations (default: 6).
--model       : optional model override (e.g., gpt-4o-mini).
--base-url    : optional router URL (OpenAI-compatible), e.g. https://api.metisai.ir/openai/v1
--verbose     : print step-by-step trace (tool calls, observations, final).

ENVIRONMENT
-----------
- OPENAI_API_KEY required; MODEL, OPENAI_BASE_URL/OPENAI_API_BASE optional.
- BRSAPI_KEY required for TSETMC tools.

EXAMPLES
--------
python runner.py --query "What time is it?" --max-steps 2 --verbose
python runner.py --query "Fetch pc (close) for نماد 'خودرو'" --max-steps 4 --verbose
python runner.py --query "Get 'pl' (last) for IKCO" --max-steps 4 --verbose
python runner.py --query "..." --model gpt-4o-mini --base-url https://api.metisai.ir/openai/v1 --verbose
"""

from __future__ import annotations
import argparse

from agent_lc import run_query


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--model", default=None, help="Override model name (optional)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (router)")
    p.add_argument("--verbose", action="store_true", help="Print step-by-step trace logs")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = run_query(
        args.query,
        model=args.model,
        base_url=args.base_url,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
