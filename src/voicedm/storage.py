"""SQLite storage for sessions and messages.

One connection, used from worker threads via ``asyncio.to_thread`` so queries never
block the event loop; a lock serialises access to it. Schema changes are numbered
migrations tracked in SQLite's ``user_version`` and applied when the database opens.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

DB_NAME = "voicedm.db"
TITLE_LENGTH = 60  # characters of the first user utterance used as a session title

# Each entry upgrades the schema by one version; append new ones, never edit old ones.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE sessions (
      id            INTEGER PRIMARY KEY,
      title         TEXT,
      created_at    TEXT NOT NULL,
      updated_at    TEXT NOT NULL,
      model         TEXT NOT NULL,
      voice         TEXT NOT NULL,
      system_prompt TEXT NOT NULL,
      summary       TEXT NOT NULL DEFAULT '',
      summary_upto  INTEGER
    );
    CREATE TABLE messages (
      id          INTEGER PRIMARY KEY,
      session_id  INTEGER NOT NULL REFERENCES sessions(id),
      role        TEXT NOT NULL,
      content     TEXT NOT NULL,
      model       TEXT,
      interrupted INTEGER NOT NULL DEFAULT 0,
      created_at  TEXT NOT NULL
    );
    CREATE INDEX messages_session ON messages(session_id, id);
    """,
]

SESSION_FIELDS = ("title", "model", "voice", "system_prompt")


def now() -> str:
    """UTC timestamp in a fixed-width ISO 8601 form, so text order is time order."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def auto_title(text: str) -> str:
    """A session title from the first thing the user said."""
    text = " ".join(text.split())
    if len(text) <= TITLE_LENGTH:
        return text
    return text[: TITLE_LENGTH - 1].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def migrate(conn: sqlite3.Connection) -> int:
    """Apply the migrations the database has not seen yet; return the schema version."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > len(MIGRATIONS):
        raise RuntimeError(
            f"database schema version {version} is newer than this program "
            f"({len(MIGRATIONS)}); refusing to open it"
        )
    for number in range(version + 1, len(MIGRATIONS) + 1):
        # executescript commits first; wrap each migration and its version bump in a
        # transaction of its own so a failure leaves the previous version intact.
        conn.executescript(
            f"BEGIN; {MIGRATIONS[number - 1]}; PRAGMA user_version = {number}; COMMIT;"
        )
    return len(MIGRATIONS)


class Storage:
    """Sessions and their messages in one SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.version = migrate(self._conn)

    @classmethod
    def in_dir(cls, data_dir: Path) -> Storage:
        return cls(data_dir / DB_NAME)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        def locked() -> T:
            with self._lock:
                return fn(*args)

        return await asyncio.to_thread(locked)

    # -- sessions --

    def _get_session(self, session_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def _create_session(self, model: str, voice: str, system_prompt: str) -> dict:
        stamp = now()
        cursor = self._conn.execute(
            "INSERT INTO sessions (created_at, updated_at, model, voice, system_prompt)"
            " VALUES (?, ?, ?, ?, ?)",
            (stamp, stamp, model, voice, system_prompt),
        )
        return self._get_session(cursor.lastrowid)

    def _update_session(self, session_id: int, changes: dict[str, Any]) -> dict | None:
        unknown = set(changes) - set(SESSION_FIELDS)
        if unknown:
            raise ValueError(f"unknown session fields: {sorted(unknown)}")
        if changes:
            assignments = ", ".join(f"{name} = ?" for name in changes)
            self._conn.execute(
                f"UPDATE sessions SET {assignments}, updated_at = ? WHERE id = ?",
                (*changes.values(), now(), session_id),
            )
        return self._get_session(session_id)

    def _list_sessions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT s.*, (SELECT count(*) FROM messages m WHERE m.session_id = s.id)"
            " AS message_count FROM sessions s ORDER BY s.updated_at DESC, s.id DESC"
        )
        return [dict(row) for row in rows]

    async def get_session(self, session_id: int) -> dict | None:
        return await self._run(self._get_session, session_id)

    async def latest_session(self) -> dict | None:
        """The most recently used session, if any."""

        def latest() -> dict | None:
            row = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC, id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

        return await self._run(latest)

    async def create_session(self, model: str, voice: str, system_prompt: str) -> dict:
        return await self._run(self._create_session, model, voice, system_prompt)

    async def update_session(self, session_id: int, **changes: Any) -> dict | None:
        """Change some of title, model, voice and system prompt. None if no such session."""
        return await self._run(self._update_session, session_id, changes)

    async def list_sessions(self) -> list[dict]:
        """All sessions, most recently used first, with their message counts."""
        return await self._run(self._list_sessions)

    # -- messages --

    def _messages(self, session_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        )
        return [{**dict(row), "interrupted": bool(row["interrupted"])} for row in rows]

    def _add_message(
        self, session_id: int, role: str, content: str, model: str | None, interrupted: bool
    ) -> dict:
        if role not in ("user", "assistant"):
            raise ValueError(f"invalid role {role!r}")
        stamp = now()
        with self._conn:  # one transaction: message, session timestamp and title
            self._conn.execute("BEGIN")
            cursor = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, model, interrupted, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, model, int(interrupted), stamp),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (stamp, session_id)
            )
            if role == "user":
                self._conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ? AND title IS NULL",
                    (auto_title(content), session_id),
                )
        row = self._conn.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,))
        message = dict(row.fetchone())
        return {**message, "interrupted": bool(message["interrupted"])}

    async def messages(self, session_id: int) -> list[dict]:
        """A session's messages, oldest first."""
        return await self._run(self._messages, session_id)

    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        *,
        model: str | None = None,
        interrupted: bool = False,
    ) -> dict:
        """Append a message and mark the session as used; the first user message titles
        an untitled session."""
        return await self._run(self._add_message, session_id, role, content, model, interrupted)
