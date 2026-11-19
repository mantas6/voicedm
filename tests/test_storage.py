import sqlite3

import pytest

from voicedm import storage
from voicedm.storage import MIGRATIONS, Storage, auto_title, migrate


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


def test_fresh_database_gets_the_schema(tmp_path):
    s = Storage(tmp_path / "sub" / "fresh.db")
    assert s.version == len(MIGRATIONS)
    conn = sqlite3.connect(tmp_path / "sub" / "fresh.db")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    assert columns == {
        "id", "title", "created_at", "updated_at", "model", "voice", "system_prompt",
        "summary", "summary_upto",
    }  # fmt: skip
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    assert columns == {
        "id", "session_id", "role", "content", "model", "interrupted", "created_at",
    }  # fmt: skip
    conn.close()
    s.close()


async def test_data_survives_reopening(tmp_path):
    path = tmp_path / "test.db"
    s = Storage(path)
    session = await s.create_session("m", "v.wav", "be brief")
    await s.add_message(session["id"], "user", "hello")
    s.close()

    s = Storage(path)
    assert (await s.get_session(session["id"]))["system_prompt"] == "be brief"
    assert [m["content"] for m in await s.messages(session["id"])] == ["hello"]
    s.close()


def test_migrations_are_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "test.db", isolation_level=None)
    assert migrate(conn) == len(MIGRATIONS)
    assert migrate(conn) == len(MIGRATIONS)  # nothing left to apply, no error
    conn.close()


def test_new_migrations_apply_on_top_of_existing_data(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path, isolation_level=None)
    migrate(conn)
    conn.execute(
        "INSERT INTO sessions (created_at, updated_at, model, voice, system_prompt)"
        " VALUES ('t', 't', 'm', 'v', 'p')"
    )
    extra = "ALTER TABLE sessions ADD COLUMN extra TEXT NOT NULL DEFAULT 'x'"
    monkeypatch.setattr(storage, "MIGRATIONS", [*MIGRATIONS, extra])
    assert migrate(conn) == len(MIGRATIONS) + 1
    assert conn.execute("SELECT model, extra FROM sessions").fetchall() == [("m", "x")]
    conn.close()


def test_failed_migration_leaves_previous_version(tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "test.db", isolation_level=None)
    migrate(conn)
    broken = "CREATE TABLE half (x INTEGER); INSERT INTO no_such_table VALUES (1)"
    monkeypatch.setattr(storage, "MIGRATIONS", [*MIGRATIONS, broken])
    with pytest.raises(sqlite3.OperationalError):
        migrate(conn)
    if conn.in_transaction:
        conn.execute("ROLLBACK")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "half" not in tables
    conn.close()


def test_newer_database_is_refused(tmp_path):
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        Storage(tmp_path / "test.db")


async def test_messages_keep_order_role_model_and_interruption(store):
    session = await store.create_session("m1", "v.wav", "")
    sid = session["id"]
    await store.add_message(sid, "user", "What is two plus two?")
    await store.add_message(sid, "assistant", "Four.", model="m1")
    await store.add_message(sid, "user", "And three?")
    await store.add_message(sid, "assistant", "Three plus", model="m2", interrupted=True)
    messages = await store.messages(sid)
    assert [(m["role"], m["content"], m["model"], m["interrupted"]) for m in messages] == [
        ("user", "What is two plus two?", None, False),
        ("assistant", "Four.", "m1", False),
        ("user", "And three?", None, False),
        ("assistant", "Three plus", "m2", True),
    ]
    with pytest.raises(ValueError):
        await store.add_message(sid, "system", "nope")


async def test_first_user_message_titles_the_session(store):
    session = await store.create_session("m", "v", "p")
    assert session["title"] is None
    await store.add_message(session["id"], "assistant", "Hi, how can I help?", model="m")
    await store.add_message(session["id"], "user", "Tell me  about\nthe moon")
    await store.add_message(session["id"], "user", "Something else")
    assert (await store.get_session(session["id"]))["title"] == "Tell me about the moon"


async def test_set_title_is_not_overwritten(store):
    session = await store.create_session("m", "v", "p")
    await store.update_session(session["id"], title="Mine")
    await store.add_message(session["id"], "user", "hello")
    assert (await store.get_session(session["id"]))["title"] == "Mine"


def test_auto_title_truncates_at_a_word():
    title = auto_title("word " * 30)
    assert len(title) <= storage.TITLE_LENGTH
    assert title.endswith("word…")


async def test_sessions_list_most_recently_used_first(store):
    a = await store.create_session("m", "v", "p")
    b = await store.create_session("m", "v", "p")
    assert [s["id"] for s in await store.list_sessions()] == [b["id"], a["id"]]
    assert (await store.latest_session())["id"] == b["id"]
    await store.add_message(a["id"], "user", "hello")
    listed = await store.list_sessions()
    assert [(s["id"], s["message_count"]) for s in listed] == [(a["id"], 1), (b["id"], 0)]


async def test_update_session(store):
    session = await store.create_session("m", "v", "p")
    updated = await store.update_session(session["id"], model="m2", system_prompt="")
    assert (updated["model"], updated["voice"], updated["system_prompt"]) == ("m2", "v", "")
    assert updated["updated_at"] >= session["updated_at"]
    assert await store.update_session(999, model="x") is None
    with pytest.raises(ValueError):
        await store.update_session(session["id"], summary="no")
