from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

from mini_hermes.agent import Agent
from mini_hermes.context import ContextBudget
from mini_hermes.memory import MemoryStore
from mini_hermes.messages import Message, ModelResponse, ToolCall
from mini_hermes.providers.anthropic import to_anthropic_messages
from mini_hermes.providers.base import LLMProvider
from mini_hermes.session_store import SessionStore
from mini_hermes.tools.registry import ToolRegistry, ToolSpec
from mini_hermes.tools.retry import RetryPolicy, TemporaryToolError


class ScriptedProvider(LLMProvider):
    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self.requests: list[list[Message]] = []

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.requests.append(deepcopy(messages))
        if len(self.requests) == 1:
            return ModelResponse(Message(role="assistant", tool_calls=[self.call]))
        result = json.loads(messages[-1].content)
        return ModelResponse(Message(
            role="assistant", content="完成" if result["ok"] else "工具失败",
        ))


class Stage8Tests(unittest.TestCase):
    def test_retry_limits_errors_and_argument_isolation(self) -> None:
        policy = RetryPolicy(3, 0.1, 0.15)
        cases = [
            # 名称、失败次数、异常、安全开关、策略、执行次数、等待、错误码。
            ("恢复成功", 2, TemporaryToolError, True, policy, 3, [0.1, 0.15], None),
            ("耗尽次数", 99, TemporaryToolError, True, policy, 3, [0.1, 0.15], "execution_error"),
            ("参数错误", 1, ValueError, True, policy, 1, [], "invalid_arguments"),
            ("普通异常", 1, RuntimeError, True, policy, 1, [], "execution_error"),
            ("未开启安全重试", 2, TemporaryToolError, False, policy, 1, [], "execution_error"),
            ("默认一次", 2, TemporaryToolError, True, RetryPolicy(), 1, [], "execution_error"),
            ("首次成功", 0, TemporaryToolError, True, policy, 1, [], None),
        ]
        for name, failures, error_type, safe, selected, count, delays, code in cases:
            with self.subTest(name=name):
                attempts: list[dict[str, Any]] = []
                waits: list[float] = []

                def flaky(args: dict[str, Any]) -> dict[str, Any]:
                    attempts.append(deepcopy(args))
                    args["payload"]["value"] = 999
                    if len(attempts) <= failures:
                        raise error_type("模拟失败")
                    return {"value": attempts[-1]["payload"]["value"]}

                registry = ToolRegistry(sleep=waits.append)
                spec = ToolSpec(
                    name="flaky", description="测试暂时性失败",
                    parameters={"type": "object", "properties": {}},
                    handler=flaky, safe_to_retry=safe, retry_policy=selected,
                )
                registry.register(spec)
                reference = ToolRegistry()
                reference.register(ToolSpec(
                    spec.name, spec.description, spec.parameters, flaky,
                ))
                self.assertEqual(registry.schemas(), reference.schemas())

                malformed = ToolCall("bad", "flaky", parse_error="非法 JSON")
                self.assertEqual(json.loads(registry.execute(malformed))["error"]["code"],
                                 "invalid_arguments")
                missing = ToolCall("missing", "unknown")
                self.assertEqual(json.loads(registry.execute(missing))["error"]["code"],
                                 "tool_not_found")
                self.assertEqual(attempts, [])

                call = ToolCall("call_flaky", "flaky", {"payload": {"value": 7}})
                original = deepcopy(call.arguments)
                result = json.loads(registry.execute(call))
                self.assertEqual(len(attempts), count)
                self.assertEqual(waits, delays)
                self.assertEqual(call.arguments, original)
                self.assertTrue(all(args == original for args in attempts))
                if code is None:
                    self.assertEqual(result, {"ok": True, "data": {"value": 7}})
                else:
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["error"]["code"], code)

        for error_type in (InterruptedError, KeyboardInterrupt):
            with self.subTest(interruption=error_type.__name__):
                calls: list[int] = []
                waits = []

                def interrupt(args: dict[str, Any]) -> dict[str, Any]:
                    calls.append(1)
                    raise error_type()

                registry = ToolRegistry(sleep=waits.append)
                registry.register(ToolSpec(
                    "interrupt", "测试中断", {"type": "object"}, interrupt,
                    safe_to_retry=True, retry_policy=policy,
                ))
                with self.assertRaises(error_type):
                    registry.execute(ToolCall("interrupt_call", "interrupt"))
                self.assertEqual(calls, [1])
                self.assertEqual(waits, [])

    def test_retry_keeps_one_tool_result_and_full_saved_history(self) -> None:
        for failures, expected_ok in ((2, True), (99, False)):
            with self.subTest(failures=failures), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                attempts: list[int] = []
                waits: list[float] = []

                def fetch(args: dict[str, Any]) -> dict[str, Any]:
                    attempts.append(1)
                    if len(attempts) <= failures:
                        raise TemporaryToolError("模拟服务暂时不可用")
                    return {"text": "资料"}

                registry = ToolRegistry(sleep=waits.append)
                registry.register(ToolSpec(
                    "fetch", "获取资料", {"type": "object", "properties": {}}, fetch,
                    safe_to_retry=True, retry_policy=RetryPolicy(3, 0.1, 0.15),
                ))
                schemas_before = deepcopy(registry.schemas())
                call = ToolCall("fetch_call", "fetch")
                provider = ScriptedProvider(call)
                with SessionStore(root / "sessions.db") as store, MemoryStore(
                    root / "memory.db",
                ) as memory:
                    memory.remember("language", "Python", ["查询"])
                    agent = Agent(
                        provider, registry, "固定系统提示", max_iterations=2,
                        store=store, memory=memory, context=ContextBudget(4096, 512),
                    )
                    answer = agent.run_turn("查询资料")
                    self.assertEqual(answer, "完成" if expected_ok else "工具失败")
                    self.assertEqual(len(attempts), 3)
                    self.assertEqual(waits, [0.1, 0.15])
                    self.assertEqual(len(provider.requests), 2)
                    tool_messages = [m for m in agent.messages if m.role == "tool"]
                    self.assertEqual(len(tool_messages), 1)
                    self.assertEqual(tool_messages[0].tool_call_id, call.id)
                    self.assertEqual(json.loads(tool_messages[0].content)["ok"], expected_ok)
                    self.assertEqual(provider.requests[1][-1], tool_messages[0])
                    self.assertIn("Python", provider.requests[0][1].content)
                    self.assertEqual(registry.schemas(), schemas_before)
                    self.assertEqual(store.load_session(agent.session_id).messages,
                                     agent.messages[1:])
                    restored = Agent(
                        provider, registry, "不能覆盖旧提示",
                        store=store, session_id=agent.session_id,
                    )
                    self.assertEqual(restored.messages, agent.messages)
                    _, native = to_anthropic_messages(provider.requests[1])
                    block = native[-1]["content"][0]
                    self.assertEqual(block.get("is_error", False), not expected_ok)


if __name__ == "__main__":
    unittest.main()