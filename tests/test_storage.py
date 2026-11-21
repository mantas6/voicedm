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
        "summary", "summary_upto", "expressions", "silent_actions",
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


async def test_expressions_are_off_unless_set_and_read_back_as_a_flag(store):
    plain = await store.create_session("m", "v", "p")
    assert plain["expressions"] is False
    assert (await store.create_session("m", "v", "p", expressions=True))["expressions"] is True
    assert (await store.update_session(plain["id"], expressions=True))["expressions"] is True
    assert (await store.get_session(plain["id"]))["expressions"] is True
    assert (await store.latest_session())["expressions"] is True
    assert {s["expressions"] for s in await store.list_sessions()} == {True}
    assert (await store.update_session(plain["id"], expressions=False))["expressions"] is False


async def test_silent_actions_are_off_unless_set_and_separate_from_expressions(store):
    plain = await store.create_session("m", "v", "p")
    assert plain["silent_actions"] is False
    silent = await store.create_session("m", "v", "p", silent_actions=True)
    assert (silent["silent_actions"], silent["expressions"]) == (True, False)
    updated = await store.update_session(plain["id"], silent_actions=True)
    assert (updated["silent_actions"], updated["expressions"]) == (True, False)
    assert (await store.get_session(plain["id"]))["silent_actions"] is True


async def test_summary_and_messages_after_it(store):
    session = await store.create_session("m", "v", "p")
    sid = session["id"]
    assert (session["summary"], session["summary_upto"]) == ("", None)
    ids = [(await store.add_message(sid, "user", f"m{i}"))["id"] for i in range(4)]
    before = (await store.get_session(sid))["updated_at"]
    await store.set_summary(sid, "Talked about m0 and m1.", ids[1])
    stored = await store.get_session(sid)
    assert (stored["summary"], stored["summary_upto"]) == ("Talked about m0 and m1.", ids[1])
    assert stored["updated_at"] == before  # summarising is not using the session
    assert [m["content"] for m in await store.messages(sid, after=ids[1])] == ["m2", "m3"]
    assert len(await store.messages(sid)) == 4  # folded messages stay
    assert len(await store.messages(sid, after=None)) == 4


async def test_delete_session_removes_it_with_its_messages(store):
    keep = await store.create_session("m", "v", "p")
    gone = await store.create_session("m", "v", "p")
    await store.add_message(keep["id"], "user", "stays")
    await store.add_message(gone["id"], "user", "goes")
    await store.add_message(gone["id"], "assistant", "goes too", model="m")
    assert await store.delete_session(gone["id"]) is True
    assert await store.get_session(gone["id"]) is None
    assert await store.messages(gone["id"]) == []
    assert [s["id"] for s in await store.list_sessions()] == [keep["id"]]
    assert [m["content"] for m in await store.messages(keep["id"])] == ["stays"]
    assert await store.delete_session(gone["id"]) is False  # already gone
    assert await store.delete_session(999) is False


async def test_session_ids_are_never_reused(store):
    first = await store.create_session("m", "v", "p")
    last = await store.create_session("m", "v", "p")
    await store.delete_session(last["id"])
    new = await store.create_session("m", "v", "p")
    assert new["id"] > last["id"]
    await store.delete_session(first["id"])
    await store.delete_session(new["id"])
    assert (await store.create_session("m", "v", "p"))["id"] > new["id"]  # even when empty


def test_version_1_database_is_rebuilt_with_its_data(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path, isolation_level=None)
    monkeypatch.setattr(storage, "MIGRATIONS", MIGRATIONS[:1])
    assert migrate(conn) == 1
    for sid, title in ((1, "one"), (2, None), (5, "five")):
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, model, voice,"
            " system_prompt, summary, summary_upto) VALUES (?, ?, 'c', 'u', 'm', 'v', '', 's', 3)",
            (sid, title),
        )
    for mid, sid in ((1, 1), (2, 5), (3, 5)):
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at)"
            " VALUES (?, ?, 'user', 'hi', 't')",
            (mid, sid),
        )
    columns = (
        "id, title, created_at, updated_at, model, voice, system_prompt, summary, summary_upto"
    )
    before = conn.execute(f"SELECT {columns} FROM sessions ORDER BY id").fetchall()
    conn.close()
    monkeypatch.undo()

    s = Storage(path)  # migrates to the current version
    assert s.version == len(MIGRATIONS)
    conn = sqlite3.connect(path)
    assert conn.execute(f"SELECT {columns} FROM sessions ORDER BY id").fetchall() == before
    flags = "SELECT DISTINCT expressions, silent_actions FROM sessions"
    assert conn.execute(flags).fetchall() == [(0, 0)]
    assert conn.execute("SELECT id, session_id FROM messages ORDER BY id").fetchall() == [
        (1, 1),
        (2, 5),
        (3, 5),
    ]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "sessions_new" not in tables
    assert "messages_session" in tables  # the index survived
    conn.close()
    # Foreign keys are enforced again, and new ids continue after the highest old one.
    with pytest.raises(sqlite3.IntegrityError):
        s._conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at)"
            " VALUES (999, 'user', 'orphan', 't')"
        )
    new_id = s._conn.execute(
        "INSERT INTO sessions (created_at, updated_at, model, voice, system_prompt)"
        " VALUES ('c', 'u', 'm', 'v', '')"
    ).lastrowid
    assert new_id == 6
    s.close()
