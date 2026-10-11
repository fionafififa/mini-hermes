from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

from mini_hermes.agent import Agent
from mini_hermes.context import ContextBudget, estimate_input_tokens
from mini_hermes.memory import MemoryStore
from mini_hermes.messages import Message, ModelResponse, ToolCall
from mini_hermes.providers.base import LLMProvider
from mini_hermes.session_store import SessionStore
from mini_hermes.tools.registry import ToolRegistry, ToolSpec


class ScriptedProvider(LLMProvider):
    """只记录请求和返回预设回答，不访问模型服务。"""

    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.requests: list[list[Message]] = []

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.requests.append(deepcopy(messages))
        return ModelResponse(self.responses.pop(0))


def build_tools(text: str, description: str = "获取资料") -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="fetch", description=description,
        parameters={"type": "object", "properties": {}},
        handler=lambda args: {"text": text},
    ))
    return registry


class Stage7Tests(unittest.TestCase):
    def test_trimmed_requests_keep_tool_pairs_memory_and_full_saved_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = build_tools("资料" * 400)
            calls = [ToolCall("call_a", "fetch"), ToolCall("call_b", "fetch")]
            request_tools = Message(role="assistant", tool_calls=calls)
            provider = ScriptedProvider([
                request_tools, Message(role="assistant", content="根据资料回答"),
            ])
            with SessionStore(root / "sessions.db") as store, MemoryStore(
                root / "memory.db",
            ) as memory:
                memory.remember("language", "Python", ["当前问题"])
                old_messages = [
                    Message(role="user", content="旧讨论" * 200),
                    Message(role="assistant", tool_calls=[ToolCall("old", "fetch")]),
                    Message(role="tool", content="旧结果", tool_call_id="old"),
                    Message(role="assistant", content="旧回答"),
                ]
                session_id = store.create_session("固定系统提示")
                store.append_message(session_id, old_messages)
                current = [
                    Message(role="system", content="固定系统提示"),
                    Message(
                        role="user",
                        content=f"{memory.recall('当前问题')}\n\n【当前用户输入】\n当前问题",
                    ),
                    request_tools,
                    *[Message(
                        role="tool", content=tools.execute(call), tool_call_id=call.id,
                    ) for call in calls],
                ]
                # 预算能容纳当前整个工具回合，但无法再加上旧回合。
                required = estimate_input_tokens(current, tools.schemas())
                budget = ContextBudget((required + 128) * 5 // 4 + 129, 128)
                agent = Agent(
                    provider, tools, "不覆盖已保存的提示",
                    store=store, session_id=session_id, memory=memory, context=budget,
                )
                before = deepcopy(agent.messages)
                with self.assertLogs("mini_hermes.context", level="INFO") as captured:
                    self.assertEqual(agent.run_turn("当前问题"), "根据资料回答")

                first, second = provider.requests
                self.assertEqual(first[:-1], before)
                self.assertEqual(second, current)
                self.assertIn("Python", second[1].content)
                self.assertEqual([m.tool_call_id for m in second if m.role == "tool"],
                                 [call.id for call in second[2].tool_calls])
                for request in provider.requests:
                    self.assertLessEqual(
                        estimate_input_tokens(request, tools.schemas()), budget.input_limit,
                    )
                self.assertTrue(any("dropped_turns=1" in log for log in captured.output))
                self.assertEqual(agent.messages[:len(before)], before)
                self.assertEqual(store.load_session(session_id).messages, agent.messages[1:])
                saved = deepcopy(agent.messages)

            with SessionStore(root / "sessions.db") as store:
                restored = Agent(
                    provider, tools, "不能覆盖历史", store=store,
                    session_id=session_id, context=budget,
                )
                self.assertEqual(restored.messages, saved)

    def test_unfit_current_turn_or_schema_fails_without_committing_history(self) -> None:
        cases = [
            ("当前输入过大", "长" * 2000, build_tools("短结果"), 0),
            ("工具定义过大", "查询", build_tools("短结果", "描述" * 2000), 0),
            ("工具结果过大", "查询", build_tools("资料" * 2000), 1),
        ]
        for name, question, tools, expected_requests in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                provider = ScriptedProvider([Message(
                    role="assistant", tool_calls=[ToolCall("large_call", "fetch")],
                )])
                with SessionStore(Path(directory) / "sessions.db") as store:
                    agent = Agent(
                        provider, tools, "固定提示", store=store,
                        context=ContextBudget(1000, 128),
                    )
                    before = deepcopy(agent.messages)
                    with self.assertRaisesRegex(RuntimeError, "超过预算"):
                        agent.run_turn(question)
                    self.assertEqual(len(provider.requests), expected_requests)
                    self.assertEqual(agent.messages, before)
                    self.assertEqual(store.load_session(agent.session_id).messages, [])


if __name__ == "__main__":
    unittest.main()