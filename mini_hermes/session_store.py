from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from mini_hermes.messages import Message, ToolCall


@dataclass
class StoredSession:
    session_id: str
    system_prompt: str
    messages: list[Message]  #不包含system消息


class SessionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.db_path)
        self._connection.execute("PRAGMA foreign_keys = ON")

        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                system_prompt TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                payload TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, id);
        """)

    def create_session(self, system_prompt: str) -> str:
        session_id =  uuid.uuid4().hex

        with self._connection:
            self._connection.execute(
                "INSERT INTO sessions (id, system_prompt) VALUES (?, ?)",
                (session_id, system_prompt)
            )
        return session_id

    def append_message(
        self, session_id: str, message: list[Message]
    ) -> None:
        #先完成序列化，再在用一事务中写入整个回合
        rows = [
            (
                session_id,
                json.dumps(asdict(msg), ensure_ascii=False)
            )
            for msg in message
        ]
        with self._connection:
            self._connection.executemany(
                "INSERT INTO messages (session_id, payload) VALUES (?, ?)",
                rows
            )

    def load_session(self, session_id: str) -> StoredSession:
        row = self._connection.execute(
            "SELECT system_prompt FROM sessions WHERE id = ?",
            (session_id,)
        ).fetchone()

        if row is None:
            raise ValueError(f"Session with id {session_id} not found")

        rows = self._connection.execute(
            "SELECT payload FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()

        messages = []

        for(payload,) in rows:
            data = json.loads(payload)

            #JSON中的工具调用是字典，需要还原成ToolCall对象
            calls = [
                ToolCall(**call)
                for call in data.pop("tool_calls")
            ]

            messages.append(
                Message(tool_calls=calls, **data)
            )

        return StoredSession(
            session_id=session_id,
            system_prompt=row[0],
            messages=messages
        )

    def list_sessions(self) -> list[tuple[str, str]]:
        return self._connection.execute(
            "SELECT id, created_at FROM sessions ORDER BY rowid DESC"
        ).fetchall()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()