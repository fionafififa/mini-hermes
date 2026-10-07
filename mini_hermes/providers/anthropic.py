from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic

from mini_hermes.config import AppConfig, get_api_key
from mini_hermes.messages import Message, ModelResponse, TokenUsage, ToolCall
from mini_hermes.providers.base import LLMProvider

def to_anthropic_messages(
    messages: list[Message],
) ->  tuple[str,list[dict[str, Any]]]:
    """只生成请求数据，不修改Agent中的历史。"""
    system = ""
    result: list[dict[str, Any]] = []

    for index,message in enumerate(messages):
        if message.role == "system":
            if index != 0:
                raise ValueError("system消息只能出现在历史开头。")
            system = message.content
            continue

        blocks: list[dict[str, Any]] = []
        if message.role == "tool":
            role = "user"
            block:dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict) and payload.get("ok") is False:
                block["is_error"] = True

            blocks.append(block)

        else:
            role = message.role
            if message.content:
                blocks.append({
                    "type": "text",
                    "text": message.content,
                })

            for call in message.tool_calls:
                if call.parse_error:
                    raise ValueError(
                        "历史包含无法解析的工具调用，请检查工具调用的格式。"
                        "input 对象：请使用原provider或新建对话"
                    )

                blocks.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                })

        if not blocks:
            raise ValueError(f"不能发送空的{message.role}消息。")

        #多个内部tool消息变成同一个user消息里的多个结果块。
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(blocks)
        else:
            result.append({
                "role": role,
                "content": blocks,
            })
    return system, result

def to_anthropic_tools(
    tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []

    for tool in tools:
        function = tool["function"]
        result.append({
            "name": function["name"],
            "description": function.get("description",""),
            "input_schema": function["parameters"],
        })

    return result

class AnthropicProvider(LLMProvider):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """生成模型回复。"""
        system, api_messages = to_anthropic_messages(messages)

        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "messages": api_messages,
            "stream": False,
        }

        if system:
            request["system"] = system

        if tools:
            request["tools"] = to_anthropic_tools(tools)

        if self.config.extra_body:
            request["extra_body"] = self.config.extra_body

        with Anthropic(
            api_key=get_api_key(self.config),
            base_url=self.config.base_url,
            timeout=60.0,
            max_retries=0,
        ) as client:
            response = client.messages.create(
                **request
            )

        if response.stop_reason =="max_tokens":
            raise RuntimeError(
                "模型输出到达max_output_tokens，本轮未保存："
                "请增大配置值后重试"
            )

        texts:list[str] = []
        calls:list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                texts.append(block.text)

            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

            else:
                raise RuntimeError(
                    f"模型返回了未知类型的消息：{block.type}"
                )

        return ModelResponse(
            message=Message(
                role="assistant",
                content="".join(texts),
                tool_calls=calls,
            ),
            usage =TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )
