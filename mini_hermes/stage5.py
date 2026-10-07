from __future__ import annotations

import argparse
from pathlib import Path

from mini_hermes.agent import Agent
from mini_hermes.config import load_config
from mini_hermes.providers.factory import create_provider
from mini_hermes.session_store import SessionStore
from mini_hermes.stage4 import SYSTEM_PROMPT, show_messages
from mini_hermes.tools.builtin import build_default_registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="第五阶段：切换 provider"
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
            print("会话 ID                            创建时间（UTC）")
            for session_id, created_at in store.list_sessions():
                print(f"{session_id}  {created_at}")
            return

        config = load_config(args.config)

        agent = Agent(
            provider=create_provider(config),
            tools=build_default_registry(Path.cwd()),
            system_prompt=SYSTEM_PROMPT,
            max_iterations=config.max_iterations,
            store=store,
            session_id=args.session,
        )

        print(f"模型：{config.provider} / {config.model}")
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
                print(f"本轮未保存：{type(exc).__name__}: {exc}")

                response = getattr(exc, "response", None)

                if response is not None:
                    print("HTTP 状态：", response.status_code)
                    print("请求地址：", response.request.url)
                    print(
                        "响应类型：",
                        response.headers.get("content-type"),
                    )
                    print(
                        "响应正文：",
                        repr(response.text[:3000]),
                    )

                continue

            show_messages(agent.messages[before:])


if __name__ == "__main__":
    main()
