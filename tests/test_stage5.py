import copy
import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from mini_hermes.agent import Agent
from mini_hermes.config import AppConfig, load_config
from mini_hermes.messages import Message, ToolCall
from mini_hermes.providers.anthropic import to_anthropic_messages
from mini_hermes.providers.factory import PROVIDERS, create_provider
from mini_hermes.session_store import SessionStore
from mini_hermes.tools.builtin import build_default_registry


@contextmanager
def fake_api(responses):
    """真实 SDK 请求本地 HTTP 服务，不访问收费模型服务。"""
    remaining = list(responses)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            requests.append((self.path, body))
            if remaining:
                status, response = 200, remaining.pop(0)
            else:
                status, response = 500, {"error": "unexpected request"}
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            return

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {
                "STAGE5_TEST_API_KEY": "offline-test-key",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }):
                yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)


def openai_response(message):
    return {
        "id": "chatcmpl-offline",
        "object": "chat.completion",
        "created": 0,
        "model": "offline-model",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def anthropic_response(content, stop_reason="end_turn"):
    return {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": "offline-model",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class Stage5Tests(unittest.TestCase):
    def test_conversion_preserves_history_and_groups_tool_results(self):
        messages = [
            Message("system", "固定系统提示"),
            Message("user", "执行两个工具"),
            Message("assistant", "我来执行", tool_calls=[
                ToolCall("call_a", "add_numbers", {"a": 2, "b": 3}),
                ToolCall("call_b", "read_file", {"path": "missing.txt"}),
            ]),
            Message("tool", '{"ok":true,"data":{"sum":5}}', tool_call_id="call_a"),
            Message("tool", '{"ok":false,"error":{"code":"test"}}', tool_call_id="call_b"),
            Message("assistant", "结果是 5,文件不存在"),
            Message("user", "继续"),
        ]
        before = copy.deepcopy([asdict(message) for message in messages])
        system, api_messages = to_anthropic_messages(messages)
        self.assertEqual(system, messages[0].content)
        self.assertEqual([item["role"] for item in api_messages],
                         ["user", "assistant", "user", "assistant", "user"])
        results = api_messages[2]["content"]
        self.assertEqual([item["tool_use_id"] for item in results], ["call_a", "call_b"])
        self.assertNotIn("is_error", results[0])
        self.assertTrue(results[1]["is_error"])
        self.assertEqual([asdict(message) for message in messages], before)

        bad_history = [Message("assistant", tool_calls=[
            ToolCall("bad", "add_numbers", {}, raw_arguments="{", parse_error="invalid JSON"),
        ])]
        with self.assertRaisesRegex(ValueError, "无法解析"):
            to_anthropic_messages(bad_history)

    def test_factory_and_config_are_backwards_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                'provider="openai_compatible"\nmodel="offline-model"\nbase_url="http://localhost"\n',
                encoding="utf-8-sig",
            )
            config = load_config(path)
            self.assertGreater(config.max_output_tokens, 0)
            for name, provider_class in PROVIDERS.items():
                with self.subTest(provider=name):
                    provider = create_provider(AppConfig(name, config.model, config.base_url))
                    self.assertIsInstance(provider, provider_class)
            with self.assertRaisesRegex(ValueError, "不支持的 provider"):
                create_provider(AppConfig("typo", config.model, config.base_url))
            for value in (0, -1, True, "1024"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    AppConfig(config.provider, config.model, config.base_url, max_output_tokens=value)

    def test_real_sdk_tool_loop_and_sqlite_restore_across_protocols(self):
        responses = [
            openai_response({"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_add", "type": "function", "function": {
                    "name": "add_numbers", "arguments": '{"a":2,"b":3}'}},
                {"id": "call_missing", "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path":"missing.txt"}'}},
            ]}),
            openai_response({"role": "assistant", "content": "5，文件不存在。"}),
            anthropic_response([
                {"type": "text", "text": "再算一次"},
                {"type": "tool_use", "id": "toolu_next", "name": "add_numbers",
                 "input": {"a": 5, "b": 7}},
            ], "tool_use"),
            anthropic_response([{"type": "text", "text": "12。"}]),
        ]
        with tempfile.TemporaryDirectory() as directory, fake_api(responses) as (url, requests):
            root = Path(directory)
            db_path = root / "sessions.db"
            config = AppConfig("openai_compatible", "offline-model", url + "/v1",
                               api_key_env="STAGE5_TEST_API_KEY")
            with SessionStore(db_path) as store:
                agent = Agent(create_provider(config), build_default_registry(root),
                              "原始系统提示", store=store)
                self.assertEqual(agent.run_turn("计算 2+3 并读取 missing.txt"), "5，文件不存在。")
                session_id = agent.session_id
                saved_history = [asdict(message) for message in agent.messages]

            # 换协议并重新打开数据库，Agent 的实现没有任何改动。
            config = AppConfig("anthropic", "offline-model", url,
                               api_key_env="STAGE5_TEST_API_KEY", max_output_tokens=800)
            with SessionStore(db_path) as store:
                agent = Agent(create_provider(config), build_default_registry(root),
                              "这个提示不能覆盖旧会话", store=store, session_id=session_id)
                self.assertEqual([asdict(message) for message in agent.messages], saved_history)
                self.assertEqual(agent.run_turn("再计算 5+7"), "12。")
                history = [asdict(message) for message in agent.messages[1:]]
                self.assertEqual([asdict(message) for message in store.load_session(session_id).messages], history)

            self.assertEqual([path for path, _ in requests],
                             ["/v1/chat/completions", "/v1/chat/completions", "/v1/messages", "/v1/messages"])
            self.assertEqual(requests[1][1]["messages"][-1]["role"], "tool")
            native_request = requests[2][1]
            self.assertEqual(native_request["system"], "原始系统提示")
            self.assertEqual(native_request["max_tokens"], config.max_output_tokens)
            self.assertTrue(native_request["messages"][2]["content"][1]["is_error"])
            schemas = build_default_registry(root).schemas()
            self.assertEqual(
                {item["name"]: item["input_schema"] for item in native_request["tools"]},
                {item["function"]["name"]: item["function"]["parameters"] for item in schemas},
            )
            result = requests[3][1]["messages"][-1]["content"][0]
            self.assertEqual(result["tool_use_id"], "toolu_next")
            self.assertEqual(json.loads(result["content"])["data"]["sum"], 12)

    def test_truncated_response_does_not_commit_a_turn(self):
        response = anthropic_response([
            {"type": "tool_use", "id": "toolu_cut", "name": "add_numbers", "input": {"a": 1, "b": 2}},
        ], "max_tokens")
        with tempfile.TemporaryDirectory() as directory, fake_api([response]) as (url, requests):
            root = Path(directory)
            config = AppConfig("anthropic", "offline-model", url, api_key_env="STAGE5_TEST_API_KEY")
            with SessionStore(root / "sessions.db") as store:
                agent = Agent(create_provider(config), build_default_registry(root), "测试", store=store)
                with patch.object(agent.tools, "execute", wraps=agent.tools.execute) as execute:
                    with self.assertRaisesRegex(RuntimeError, "max_output_tokens"):
                        agent.run_turn("计算")
                    execute.assert_not_called()
                self.assertEqual(len(agent.messages), 1)
                self.assertEqual(store.load_session(agent.session_id).messages, [])
            self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
