from __future__ import annotations

import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

from agent_lc import build_agent


def test_build_agent_constructs_executor(monkeypatch):
    # Provide a dummy API key so ChatOpenAI init doesn't complain
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    executor = build_agent(max_steps=2, verbose=False)
    # We don't invoke the agent (would require a live model). Construction should succeed.
    assert hasattr(executor, "invoke")
