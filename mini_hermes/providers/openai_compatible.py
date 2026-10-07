from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from mini_hermes.config import AppConfig, get_api_key
from mini_hermes.messages import (
    Message,
    ModelResponse,
    TokenUsage,
    ToolCall,
)
from mini_hermes.providers.base import LLMProvider


def to_api_message(message: Message) -> dict[str, Any]:
    """将 Message 转换为 OpenAI API 消息格式。"""

    result: dict[str, Any] = {
        "role": message.role,
        "content": (
            None
            if message.role == "assistant" and not message.content and message.tool_calls
            else message.content
        )
    }

    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": (
                    call.raw_arguments
                    if call.raw_arguments is not None
                    else json.dumps(call.arguments, ensure_ascii=False)
                ),
                },
            }
            for call in message.tool_calls
        ]

    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id

    return result


class OpenAICompatibleProvider(LLMProvider):
    """调用支持 chat.completions 的 OpenAI 兼容模型。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate(
            self,
            messages: list[Message],
            tools: list[dict[str,Any]] | None = None,
    ) -> ModelResponse:
        #1. 将 Message 转换为 API 消息格式
        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "messages": [
                to_api_message(m)
                for m in messages
            ],
            "stream": False,
        }

        if tools:
            request["tools"] = tools

        if self.config.extra_body:
            request["extra_body"] = self.config.extra_body
        
        #2.创建客户端并发送一次请求
        #with会在结束时关闭客户端连接
        with OpenAI(
            api_key=get_api_key(self.config),
            base_url=self.config.base_url,
            timeout=60.0,
            max_retries=0,
        ) as client:
            completion = client.chat.completions.create(**request)

        if not completion.choices:
            raise RuntimeError("模型服务返回了空 choices")

        #completion 是 SDK 类型，里面装着模型的回答
        raw_message = completion.choices[0].message

        #3.把API 工具调用转换成内部的ToolCall
        tool_calls: list[ToolCall] = []


        for call in raw_message.tool_calls or []:
            if call.type != "function":
                raise ValueError(f"不支持的工具调用类型：{call.type}")

            raw_arguments = call.function.arguments

            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")
                parse_error = None
            except ValueError as exc:
                arguments = {}
                parse_error = f"工具参数解析失败：{exc}"

            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                    raw_arguments=raw_arguments if parse_error else None,
                    parse_error=parse_error,
                )
            )

        #4.把API用量转换成内部 TokenUsage
        usage = None
        if completion.usage is not None:
            usage = TokenUsage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
            )

        #5.返回ModelResponse
        return ModelResponse(
            message=Message(
                role='assistant',
                content=raw_message.content or raw_message.refusal or "",
                tool_calls=tool_calls,
            ),
            usage=usage,
        )
