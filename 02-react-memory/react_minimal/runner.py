"""
Section 2 — CLI Runner for the ReAct Agent

OVERVIEW
--------
Thin command-line wrapper that:
- Parses user inputs (query, max steps, model override, router base URL, verbosity).
- Constructs an OpenAI-compatible client (with optional router base URL).
- Runs the ReAct agent and prints the final answer.

CLI FLAGS
---------
--query       : (str, required) the user question/instruction.
--max-steps   : (int) cap on Thought→Action→Observation iterations. Default: 5
--model       : (str) optional model override, e.g., "gpt-4o-mini".
--base-url    : (str) optional router URL (OpenAI-compatible), e.g.,
                https://api.metisai.ir/openai/v1
--verbose     : print step-by-step trace logs (actions, observations, final).
                Chain-of-thought is intentionally never printed.

ENVIRONMENT
-----------
Relies on the same environment variables as `agent.py` for API key/model/base URL
if CLI flags are not provided:
- OPENAI_API_KEY   (required)
- MODEL            (optional)
- OPENAI_BASE_URL  (optional)

EXAMPLES
--------
Default:
  python3 runner.py --query "What is 12*8? Then add 10." --max-steps 3

With a router:
  python3 runner.py --query "..." --max-steps 3 \
    --base-url https://api.metisai.ir/openai/v1 --model gpt-4o-mini

Verbose tracing:
  python3 runner.py --query "..." --max-steps 3 --verbose

EXIT CODE
---------
- Always exits 0 (success) after printing the agent's final string result (or
  a fallback message if step limit was hit). Integrations can capture stdout.
"""


from __future__ import annotations
import argparse

from agent import ReActAgent, AgentConfig, OpenAIChat


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--model", default=None, help="Override model name (optional)")
    p.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (router), e.g. https://api.metisai.ir/openai/v1",
    )
    p.add_argument("--verbose", action="store_true", help="Print step-by-step trace logs")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    agent = ReActAgent(
        llm=OpenAIChat(model=args.model, base_url=args.base_url),
        config=AgentConfig(max_steps=args.max_steps, verbose=args.verbose),
    )
    answer = agent.run(args.query)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
