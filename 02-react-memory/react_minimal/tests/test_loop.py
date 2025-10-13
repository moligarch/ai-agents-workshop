from __future__ import annotations

import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

from typing import List, Dict
from agent import ReActAgent, AgentConfig, LLM


class MockLLM(LLM):
    def __init__(self, outputs: List[str]):
        self.outputs = outputs
        self.i = 0
    def complete(self, messages: List[Dict[str, str]]) -> str:
        out = self.outputs[min(self.i, len(self.outputs)-1)]
        self.i += 1
        return out


def test_loop_two_tool_calls_then_final(tmp_path):
    # 1) calc 12*8 → 96
    # 2) calc 96+10 → 106
    # 3) final answer
    outputs = [
        '{"thought":"We need math","action":{"tool":"calculator","input":{"expression":"12*8"}}}',
        '{"thought":"Add 10","action":{"tool":"calculator","input":{"expression":"96+10"}}}',
        '{"thought":"We are done","final":"106"}',
    ]

    agent = ReActAgent(llm=MockLLM(outputs), config=AgentConfig(max_steps=5))
    answer = agent.run("What is 12*8? Then add 10.")
    assert answer.strip() == "106"
