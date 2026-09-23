from __future__ import annotations

import json
from dataclasses import asdict

from mini_hermes.config import load_config
from mini_hermes.messages import Message
from mini_hermes.providers.openai_compatible import (
    OpenAICompatibleProvider,
)


def main() -> None:
    config = load_config()

    if config.provider != "openai_compatible":
        raise ValueError(
            "第二阶段请设置 provider = 'openai_compatible'"
        )

    question = input("你：").strip()

    if not question:
        print("输入为空，本次不发送请求。")
        return

    messages = [
        Message(
            role="system",
            content="你是一名耐心的编程老师，请用简洁的中文回答。",
        ),
        Message(
            role="user",
            content=question,
        ),
    ]

    provider = OpenAICompatibleProvider(config)

    print("\n正在请求模型……")
    response = provider.generate(messages)

    print("\n模型回复：")
    if response.message.content:
        print(response.message.content)
    elif response.message.tool_calls:
        print("模型返回了工具调用，本阶段只展示调用信息。")
    else:
        print("模型没有返回正文或工具调用，请检查服务端响应。")

    if response.usage is not None:
        print(
            "\nToken 用量："
            f"输入 {response.usage.input_tokens}，"
            f"输出 {response.usage.output_tokens}，"
            f"合计 {response.usage.total_tokens}"
        )

    print("\n内部 ModelResponse：")
    print(
        json.dumps(
            asdict(response),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()