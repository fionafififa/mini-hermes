import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mini_hermes.agent import Agent
from mini_hermes.messages import Message, ModelResponse, ToolCall
from mini_hermes.providers.base import LLMProvider
from mini_hermes.session_store import SessionStore
from mini_hermes.tools.builtin import build_default_registry


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[ModelResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[list[Message]] = []

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> ModelResponse:
        self.requests.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class Stage4Tests(unittest.TestCase):
    def test_reopen_restores_prompt_calls_results_and_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "sessions.db"
            first = ScriptedProvider([
                ModelResponse(Message(role="assistant", tool_calls=[
                    ToolCall("bad_call", "add_numbers", {}, "{", "非法 JSON"),
                    ToolCall("sum_call", "add_numbers", {"a": 19, "b": 23}),
                ])),
                ModelResponse(Message(role="assistant", content="答案是 42。")),
            ])
            with SessionStore(db_path) as store:
                agent = Agent(first, build_default_registry(root), "原始提示", store=store)
                self.assertEqual(agent.run_turn("请计算"), "答案是 42。")
                session_id = agent.session_id
                before_restart = [asdict(m) for m in agent.messages]

            second = ScriptedProvider([
                ModelResponse(Message(role="assistant", content="之前的答案是 42。")),
            ])
            with SessionStore(db_path) as store:
                agent = Agent(
                    second, build_default_registry(root), "后来改的提示",
                    store=store, session_id=session_id,
                )
                self.assertEqual([asdict(m) for m in agent.messages], before_restart)
                agent.run_turn("之前算出多少？")
                self.assertEqual(
                    [asdict(m) for m in second.requests[0][:-1]], before_restart
                )
                self.assertEqual(
                    len(store.load_session(session_id).messages), len(agent.messages) - 1
                )
                self.assertEqual(store.list_sessions()[0][0], session_id)

    def test_provider_failure_keeps_committed_history_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ScriptedProvider([
                ModelResponse(Message(role="assistant", tool_calls=[
                    ToolCall("call_add", "add_numbers", {"a": 1, "b": 2}),
                ])),
                RuntimeError("模拟模型调用失败"),
                ModelResponse(Message(role="assistant", content="重新回答")),
            ])
            with SessionStore(root / "sessions.db") as store:
                agent = Agent(provider, build_default_registry(root), "提示", store=store)
                before = list(agent.messages)
                with self.assertRaisesRegex(RuntimeError, "模拟模型调用失败"):
                    agent.run_turn("失败的回合")
                self.assertEqual(agent.messages, before)
                self.assertEqual(store.load_session(agent.session_id).messages, [])
                self.assertEqual(agent.run_turn("重新提问"), "重新回答")
                self.assertEqual(
                    [m.role for m in provider.requests[-1]], ["system", "user"]
                )

    def test_database_failure_rolls_back_the_whole_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "sessions.db"
            provider = ScriptedProvider([
                ModelResponse(Message(role="assistant", content="未能保存的回答")),
                ModelResponse(Message(role="assistant", content="成功保存的回答")),
            ])
            with SessionStore(db_path) as store:
                agent = Agent(provider, build_default_registry(root), "提示", store=store)
                # 在第二条消息插入时制造真实 SQLite 错误，检查第一条是否回滚。
                with closing(sqlite3.connect(db_path)) as connection:
                    connection.executescript("""
                        CREATE TRIGGER reject_second_message
                        BEFORE INSERT ON messages
                        WHEN (SELECT COUNT(*) FROM messages) >= 1
                        BEGIN
                            SELECT RAISE(ABORT, 'test write failure');
                        END;
                    """)
                with self.assertRaises(sqlite3.IntegrityError):
                    agent.run_turn("第一次提问")
                self.assertEqual(len(agent.messages), 1)
                self.assertEqual(store.load_session(agent.session_id).messages, [])
                with closing(sqlite3.connect(db_path)) as connection:
                    connection.execute("DROP TRIGGER reject_second_message")
                    connection.commit()
                self.assertEqual(agent.run_turn("再次提问"), "成功保存的回答")
                self.assertEqual(
                    store.load_session(agent.session_id).messages, agent.messages[1:]
                )


if __name__ == "__main__":
    unittest.main()
