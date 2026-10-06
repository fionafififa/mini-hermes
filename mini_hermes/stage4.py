from __future__ import annotations

import argparse
from pathlib import Path

from mini_hermes.agent import Agent
from mini_hermes.config import load_config
from mini_hermes.messages import Message
from mini_hermes.providers.openai_compatible import OpenAICompatibleProvider
from mini_hermes.session_store import SessionStore
from mini_hermes.tools.builtin import build_default_registry


SYSTEM_PROMPT = (
    "你是一名耐心的编程老师，请用简洁的中文回答。"
    "需要准确计算整数加法时可以调用 add_numbers；"
    "需要查看当前项目文件时可以调用 read_file。"
    "使用工具后，根据工具结果回答用户。"
)


def show_messages(messages: list[Message]) -> None:
    for message in messages:
        if message.role == "tool":
            print(
                f"[工具结果 {message.tool_call_id}] "
                f"{message.content[:300]}"
            )
        else:
            if message.content:
                print(f"[{message.role}] {message.content}")

            for call in message.tool_calls:
                print(
                    f"[工具调用 {call.id}] "
                    f"{call.name}({call.arguments})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="第四阶段：保存与恢复会话"
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--session",
        help="恢复指定的会话 ID",
    )

    group.add_argument(
        "--list",
        action="store_true",
        help="列出已保存的会话",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/sessions.db"),
    )

    args = parser.parse_args()

    with SessionStore(args.db) as store:
        if args.list:
            print("会话 ID                            创建时间（UTC）")

            for session_id, created_at in store.list_sessions():
                print(f"{session_id}  {created_at}")

            return

        config = load_config()

        if config.provider != "openai_compatible":
            raise ValueError(
                "第四阶段请设置 provider = 'openai_compatible'"
            )

        agent = Agent(
            provider=OpenAICompatibleProvider(config),
            tools=build_default_registry(Path.cwd()),
            system_prompt=SYSTEM_PROMPT,
            max_iterations=config.max_iterations,
            store=store,
            session_id=args.session,
        )

        print(f"数据库：{store.db_path}")
        print(f"当前会话：{agent.session_id}")
        print(f"已加载 {len(agent.messages) - 1} 条历史消息。")
        print("输入 /history 查看历史，/exit 退出。")

        while True:
            try:
                question = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                return

            if question == "/exit":
                return

            if question == "/history":
                show_messages(agent.messages)
                continue

            if not question:
                continue

            before = len(agent.messages)

            try:
                agent.run_turn(question)
            except KeyboardInterrupt:
                print("\n本轮已中断，未保存本轮消息。")
                return
            except Exception as exc:
                print(
                    f"本轮未保存：{type(exc).__name__}: {exc}"
                )
                continue

            show_messages(agent.messages[before:])


if __name__ == "__main__":
    main()