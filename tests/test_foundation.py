import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_hermes.config import get_api_key, load_config
from mini_hermes.messages import Message, ModelResponse, ToolCall


class FoundationTests(unittest.TestCase):
    def test_message_contract(self) -> None:
        call = ToolCall(
            id="call_test",
            name="add_numbers",
            arguments={"a": 2, "b": 3},
        )

        response = ModelResponse(
            message=Message(
                role="assistant",
                tool_calls=[call],
            )
        )

        result = Message(
            role="tool",
            content='{"result": 5}',
            tool_call_id=call.id,
        )

        self.assertEqual(
            response.message.tool_calls[0].id,
            result.tool_call_id,
        )

        # 工具结果不能没有关联的调用 ID。
        with self.assertRaises(ValueError):
            Message(role="tool", content="5")

        # 用户消息不能冒充模型发出工具调用。
        with self.assertRaises(ValueError):
            Message(role="user", tool_calls=[call])

        # Provider 必须返回 assistant 消息。
        with self.assertRaises(ValueError):
            ModelResponse(message=Message(role="user"))

    def test_config_loading_and_validation(self) -> None:
        config_text = """
provider = "test_provider"
model = "test_model"
base_url = "https://example.com/v1"
api_key_env = "MINI_HERMES_TEST_KEY"
max_iterations = 2
"""

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(config_text, encoding="utf-8")

            config = load_config(config_path)

            self.assertEqual(config.model, "test_model")
            self.assertEqual(config.max_iterations, 2)

            # 使用测试占位值，不读取或调用真实服务。
            with patch.dict(
                os.environ,
                {"MINI_HERMES_TEST_KEY": "test-only-key"},
            ):
                self.assertEqual(get_api_key(config), "test-only-key")

            with patch.dict(
                os.environ,
                {"MINI_HERMES_TEST_KEY": ""},
            ):
                with self.assertRaises(ValueError):
                    get_api_key(config)

            invalid_text = config_text.replace(
                "max_iterations = 2",
                "max_iterations = 0",
            )
            config_path.write_text(invalid_text, encoding="utf-8")

            with self.assertRaises(ValueError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()