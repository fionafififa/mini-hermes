from __future__ import annotations

from pathlib import Path

from mini_hermes.agent import Agent
from mini_hermes.config import load_config
from mini_hermes.providers.openai_compatible import (
    OpenAICompatibleProvider,
)
from mini_hermes.tools.builtin import build_default_registry


def main() -> None:
    config = load_config()

    if config.provider != "openai_compatible":
        raise ValueError(
            "第三阶段请设置 provider = 'openai_compatible'"
        )

    agent = Agent(
        provider=OpenAICompatibleProvider(config),
        tools=build_default_registry(Path.cwd()),
        system_prompt=(
            "你是一名耐心的编程老师，请用简洁的中文回答。"
            "需要准确计算整数加法时可以调用 add_numbers；"
            "需要查看当前项目文件时可以调用 read_file。"
            "使用工具后，根据工具结果回答用户。"
        ),
        max_iterations=config.max_iterations,
    )

    print("第三阶段 Agent 已启动。输入 /exit 退出。")

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if question == "/exit":
            print("再见。")
            return
        if not question:
            continue

        before = len(agent.messages)

        try:
            answer = agent.run_turn(question)
        except Exception as exc:
            print(f"\n本轮失败：{type(exc).__name__}: {exc}")
            return

        # 展示本轮发生了哪些工具调用，便于学习消息流。
        for message in agent.messages[before:]:
            if message.role == "assistant":
                for call in message.tool_calls:
                    print(f"[工具调用] {call.name}({call.arguments})")
            elif message.role == "tool":
                preview = message.content[:300]
                if len(message.content) > 300:
                    preview += "..."
                print(
                    f"[工具结果 {message.tool_call_id}] {preview}"
                )

        print(f"\n助手：{answer}")


if __name__ == "__main__":
    main()