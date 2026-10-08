from __future__ import annotations

import argparse
from pathlib import Path

from mini_hermes.agent import Agent
from mini_hermes.config import load_config
from mini_hermes.memory import MemoryStore
from mini_hermes.providers.factory import create_provider
from mini_hermes.session_store import SessionStore
from mini_hermes.stage4 import SYSTEM_PROMPT, show_messages
from mini_hermes.tools.builtin import build_default_registry


def handle_memory_command(
    text: str,
    memory: MemoryStore,
) -> bool:
    command, _, body = text.partition(" ")
    body = body.strip()

    if command == "/remember":
        pair, separator, keyword_text = body.partition("|")
        key, equals, value = pair.partition("=")

        if not separator or not equals:
            raise ValueError(
                "用法：/remember key=value | 关键词1,关键词2"
            )

        memory.remember(
            key,
            value,
            keyword_text.split(","),
        )
        print(f"已保存／更新记忆：{key.strip()}")
        return True

    if command == "/forget":
        deleted = memory.forget(body)
        print(
            "已删除记忆。"
            if deleted
            else "没有找到这个 key。"
        )
        return True

    if command == "/memories":
        entries = memory.list_memories()

        if not entries:
            print("暂无长期记忆。")

        for entry in entries:
            print(
                f"{entry.key}={entry.value} | "
                f"{','.join(entry.keywords)}"
            )

        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="第六阶段 P0：显式长期记忆"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/sessions.db"),
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=Path("data/memory.db"),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="恢复指定会话")
    group.add_argument(
        "--list",
        action="store_true",
        help="列出会话",
    )
    args = parser.parse_args()

    with SessionStore(args.db) as store:
        if args.list:
            for session_id, created_at in store.list_sessions():
                print(f"{session_id}  {created_at}")
            return

        with MemoryStore(
            args.memory_db,
            max_items=3,
            max_chars=1000,
        ) as memory:
            config = load_config(args.config)

            agent = Agent(
                provider=create_provider(config),
                tools=build_default_registry(Path.cwd()),
                system_prompt=SYSTEM_PROMPT,
                max_iterations=config.max_iterations,
                store=store,
                session_id=args.session,
                memory=memory,
            )

            print(f"模型：{config.provider} / {config.model}")
            print(f"会话数据库：{store.db_path}")
            print(f"记忆数据库：{memory.db_path}")
            print(f"当前会话：{agent.session_id}")
            print(
                f"已加载 {len(agent.messages) - 1} 条历史消息。"
            )
            print("/remember key=value | 关键词1,关键词2")
            print("/memories 查看记忆；/forget key 删除记忆。")
            print("/history 查看上下文；/exit 退出。")

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

                try:
                    if handle_memory_command(question, memory):
                        continue

                    before = len(agent.messages)
                    agent.run_turn(question)

                except KeyboardInterrupt:
                    print("\n已中断。")
                    return

                except Exception as exc:
                    print(
                        f"操作失败：{type(exc).__name__}: {exc}"
                    )
                    continue

                show_messages(agent.messages[before:])


if __name__ == "__main__":
    main()