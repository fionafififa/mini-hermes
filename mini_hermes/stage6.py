from __future__ import annotations

import argparse
import logging
import os
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
    *,
    session_id: str | None = None,
) -> bool:
    command, _, body = text.partition(" ")
    body = body.strip()

    remember_scopes = {"/remember": "user", "/remember-project": "project"}
    if command in remember_scopes:
        pair, separator, keyword_text = body.partition("|")
        key, equals, value = pair.partition("=")
        if not separator or not equals:
            raise ValueError(f"用法：{command} key=value | 关键词1,关键词2")
        scope = remember_scopes[command]
        memory.remember(
            key, value, keyword_text.split(","),
            scope=scope,
            source=f"cli:{command}",
            source_session_id=session_id,
        )
        print(f"已保存／更新 [{scope}] 记忆：{key.strip()}")
        return True

    forget_scopes = {"/forget": "user", "/forget-project": "project"}
    if command in forget_scopes:
        deleted = memory.forget(body, scope=forget_scopes[command])
        print("已删除记忆。" if deleted else "这个范围内没有找到该 key。")
        return True

    if command == "/memories":
        entries = memory.list_memories()
        if not entries:
            print("当前用户／项目暂无长期记忆。")
        for entry in entries:
            print(
                f"[{entry.scope}] {entry.key}={entry.value} | "
                f"{','.join(entry.keywords)}"
            )
            print(
                f"  source={entry.source}  session={entry.source_session_id}\n"
                f"  created={entry.created_at}  updated={entry.updated_at}"
            )
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="第六阶段 P1：可追溯的分区记忆")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--db", type=Path, default=Path("data/sessions.db"))
    parser.add_argument("--memory-db", type=Path, default=Path("data/memory.db"))
    parser.add_argument("--user", default="default", help="本地记忆用户分区，不是认证")
    parser.add_argument(
        "--project", type=Path, default=Path.cwd(),
        help="项目记忆分区路径；不改变工具工作目录",
    )
    parser.add_argument("--memory-log", action="store_true", help="显示每轮召回诊断")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="恢复指定会话；须保持原用户和项目不变")
    group.add_argument("--list", action="store_true", help="列出会话")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("mini_hermes.memory").setLevel(
        logging.INFO if args.memory_log else logging.WARNING
    )
    project_id = os.path.normcase(str(args.project.resolve()))

    with SessionStore(args.db) as store:
        if args.list:
            for session_id, created_at in store.list_sessions():
                print(f"{session_id}  {created_at}")
            return

        with MemoryStore(
            args.memory_db, max_items=3, max_chars=1000,
            user_id=args.user, project_id=project_id,
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
            print(f"记忆用户：{memory.user_id}；项目：{memory.project_id}")
            print(f"当前会话：{agent.session_id}")
            print(f"已加载 {len(agent.messages) - 1} 条历史消息。")
            print("/remember key=value | 关键词1,关键词2（用户范围）")
            print("/remember-project key=value | 关键词1,关键词2（项目范围）")
            print("/memories 查看；/forget key；/forget-project key。")
            print("/history 查看上下文；/exit 退出。切换用户／项目请新建会话。")

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
                    if handle_memory_command(
                        question, memory, session_id=agent.session_id,
                    ):
                        continue
                    before = len(agent.messages)
                    agent.run_turn(question)
                except KeyboardInterrupt:
                    print("\n已中断。")
                    return
                except Exception as exc:
                    print(f"操作失败：{type(exc).__name__}: {exc}")
                    continue
                show_messages(agent.messages[before:])


if __name__ == "__main__":
    main()
