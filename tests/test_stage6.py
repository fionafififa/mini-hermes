import tempfile
import unittest
from pathlib import Path
from typing import Any

from mini_hermes.agent import Agent
from mini_hermes.memory import MemoryStore
from mini_hermes.messages import Message, ModelResponse
from mini_hermes.providers.base import LLMProvider
from mini_hermes.session_store import SessionStore
from mini_hermes.tools.builtin import build_default_registry


class RecordingProvider(LLMProvider):
    """记录请求，不访问真实模型服务。"""

    def __init__(self) -> None:
        self.requests: list[list[Message]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.requests.append(list(messages))
        return ModelResponse(Message(role="assistant", content="测试回复"))


class Stage6Tests(unittest.TestCase):
    def test_reopened_memory_decodes_keywords_and_matches_complete_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "memory.db"
            keywords = ["代码", "示例"]
            with MemoryStore(db_path) as memory:
                memory.remember("language", "Python", keywords)

            with MemoryStore(db_path) as memory:
                entry = memory.list_memories()[0]
                self.assertEqual(entry.keywords, keywords)
                self.assertIn(entry.value, memory.recall("给我一个代码示例"))
                # 命中某个关键词里的一个字，不等于命中整个关键词。
                self.assertEqual(memory.recall("单独的代字"), "")
                self.assertEqual(memory.recall("今天天气如何"), "")

    def test_agent_sends_string_content_with_and_without_recall_and_restores_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with MemoryStore(root / "memory.db") as memory, SessionStore(
                root / "sessions.db"
            ) as store:
                memory.remember("language", "Python", ["代码", "示例"])
                cases = [
                    (None, "给我一个简单的代码示例。", False),
                    (memory, "今天天气如何？", False),
                    (memory, "给我一个简单的代码示例。", True),
                ]
                for selected_memory, question, expect_recall in cases:
                    with self.subTest(memory=selected_memory is not None, query=question):
                        provider = RecordingProvider()
                        agent = Agent(
                            provider=provider,
                            tools=build_default_registry(root),
                            system_prompt="固定系统提示",
                            store=store,
                            memory=selected_memory,
                        )
                        self.assertEqual(agent.run_turn(question), "测试回复")
                        request = provider.requests[0]
                        self.assertEqual([m.role for m in request], ["system", "user"])
                        self.assertIsInstance(request[-1].content, str)
                        self.assertIn(question, request[-1].content)
                        if expect_recall:
                            self.assertIn("Python", request[-1].content)
                        else:
                            self.assertEqual(request[-1].content, question)
                        self.assertEqual(request[0].content, "固定系统提示")
                        saved = store.load_session(agent.session_id)
                        self.assertEqual(saved.messages, agent.messages[1:])

                session_id = agent.session_id
                before_restart = list(agent.messages)

            with MemoryStore(root / "memory.db") as memory, SessionStore(
                root / "sessions.db"
            ) as store:
                restored = Agent(
                    provider=RecordingProvider(),
                    tools=build_default_registry(root),
                    system_prompt="不能覆盖旧提示",
                    store=store,
                    session_id=session_id,
                    memory=memory,
                )
                self.assertEqual(restored.messages, before_restart)


if __name__ == "__main__":
    unittest.main()
