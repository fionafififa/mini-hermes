import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mini_hermes.agent import Agent
from mini_hermes.messages import Message, ModelResponse, ToolCall
from mini_hermes.providers.base import LLMProvider
from mini_hermes.tools.builtin import build_default_registry


class ScriptedProvider(LLMProvider):
    """按顺序返回预设结果，便于检查 Agent 主循环。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[list[Message]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.requests.append(list(messages))
        return self.responses.pop(0)


class Stage3Tests(unittest.TestCase):
    def test_tool_results_go_back_to_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.txt").write_text(
                "Mini Hermes", encoding="utf-8"
            )

            provider = ScriptedProvider([
                ModelResponse(
                    message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="call_add",
                                name="add_numbers",
                                arguments={"a": 2, "b": 3},
                            ),
                            ToolCall(
                                id="call_read",
                                name="read_file",
                                arguments={"path": "note.txt"},
                            ),
                        ],
                    )
                ),
                ModelResponse(
                    message=Message(
                        role="assistant",
                        content="和是 5，文件内容是 Mini Hermes。",
                    )
                ),
            ])

            agent = Agent(
                provider=provider,
                tools=build_default_registry(root),
                system_prompt="测试助手",
            )
            answer = agent.run_turn("计算并读取文件")

            self.assertEqual(
                answer, "和是 5，文件内容是 Mini Hermes。"
            )
            self.assertEqual(
                [m.role for m in agent.messages],
                [
                    "system", "user", "assistant",
                    "tool", "tool", "assistant",
                ],
            )
            self.assertEqual(len(provider.requests), 2)
            self.assertEqual(
                json.loads(agent.messages[3].content)["data"]["sum"],
                5,
            )
            self.assertEqual(
                agent.messages[3].tool_call_id, "call_add"
            )
            self.assertEqual(
                agent.messages[4].tool_call_id, "call_read"
            )

    def test_second_turn_sees_previous_history(self) -> None:
        provider = ScriptedProvider([
            ModelResponse(
                message=Message(role="assistant", content="第一轮回答")
            ),
            ModelResponse(
                message=Message(role="assistant", content="第二轮回答")
            ),
        ])

        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                provider=provider,
                tools=build_default_registry(Path(directory)),
                system_prompt="测试助手",
            )
            agent.run_turn("第一轮问题")
            agent.run_turn("第二轮问题")

        self.assertEqual(
            [m.role for m in provider.requests[1]],
            ["system", "user", "assistant", "user"],
        )

    def test_bad_tool_arguments_return_error_to_model(self) -> None:
        provider = ScriptedProvider([
            ModelResponse(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="bad_call",
                            name="add_numbers",
                            arguments={},
                            raw_arguments="{",
                            parse_error="工具参数解析失败",
                        )
                    ],
                )
            ),
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="工具参数有误，我已收到错误信息。",
                )
            ),
        ])

        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                provider=provider,
                tools=build_default_registry(Path(directory)),
                system_prompt="测试助手",
            )
            agent.run_turn("计算")

        tool_result = json.loads(agent.messages[3].content)
        self.assertFalse(tool_result["ok"])
        self.assertEqual(
            tool_result["error"]["code"],
            "invalid_arguments",
        )


if __name__ == "__main__":
    unittest.main()