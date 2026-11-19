import httpx
from fakes import FakeBot, fake_upstreams
from fastapi.testclient import TestClient

from voicedm.app import create_app
from voicedm.config import DEFAULT_SYSTEM_PROMPT, Config

CONFIG = Config(ollama_url="http://ollama.test", chatterbox_url="http://tts.test")


def client_for(handler=fake_upstreams) -> TestClient:
    return TestClient(create_app(CONFIG, transport=httpx.MockTransport(handler), bot=FakeBot()))


def test_first_session_uses_first_model_voice_and_default_prompt():
    with client_for() as client:
        assert client.get("/api/sessions").json() == []
        response = client.post("/api/sessions")
    assert response.status_code == 201
    session = response.json()
    assert session["model"] == "first-model:latest"
    assert session["voice"] == "Alpha.wav"
    assert session["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    assert session["title"] is None
    assert session["messages"] == []
    assert session["in_call"] is False


def test_first_session_without_upstreams_is_unavailable():
    with client_for(lambda request: httpx.Response(503)) as client:
        assert client.post("/api/sessions").status_code == 503


def test_new_session_copies_the_latest_session_settings():
    with client_for() as client:
        first = client.post("/api/sessions").json()
        client.patch(
            f"/api/sessions/{first['id']}",
            json={"model": "second-model:7b", "voice": "Beta.wav", "system_prompt": "Be terse."},
        )
        second = client.post("/api/sessions").json()
    assert second["id"] != first["id"]
    assert (second["model"], second["voice"], second["system_prompt"]) == (
        "second-model:7b",
        "Beta.wav",
        "Be terse.",
    )


def test_new_session_copies_a_named_session():
    with client_for() as client:
        a = client.post("/api/sessions").json()
        client.patch(f"/api/sessions/{a['id']}", json={"system_prompt": "A"})
        b = client.post("/api/sessions").json()
        client.patch(f"/api/sessions/{b['id']}", json={"system_prompt": "B"})
        c = client.post("/api/sessions", json={"copy_from": a["id"]}).json()
        assert c["system_prompt"] == "A"
        assert client.post("/api/sessions", json={"copy_from": 999}).status_code == 404
        assert client.post("/api/sessions", json={"nope": 1}).status_code == 422
        # Nothing was deleted along the way.
        assert len(client.get("/api/sessions").json()) == 3


def test_get_session_with_messages():
    with client_for() as client:
        session = client.post("/api/sessions").json()
        store = client.app.state.store
        client.portal.call(store.add_message, session["id"], "user", "Hello there")
        client.portal.call(
            lambda: store.add_message(
                session["id"], "assistant", "Hi", model="first-model:latest", interrupted=True
            )
        )
        body = client.get(f"/api/sessions/{session['id']}").json()
        listed = client.get("/api/sessions").json()
    assert body["title"] == "Hello there"
    assert [(m["role"], m["content"], m["model"], m["interrupted"]) for m in body["messages"]] == [
        ("user", "Hello there", None, False),
        ("assistant", "Hi", "first-model:latest", True),
    ]
    assert listed[0]["message_count"] == 2
    assert "messages" not in listed[0]


def test_unknown_session_is_not_found():
    with client_for() as client:
        assert client.get("/api/sessions/7").status_code == 404
        assert client.patch("/api/sessions/7", json={"title": "x"}).status_code == 404


def test_patch_changes_fields_and_keeps_others():
    with client_for() as client:
        session = client.post("/api/sessions").json()
        response = client.patch(
            f"/api/sessions/{session['id']}", json={"title": "Moon chat", "system_prompt": ""}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["title"] == "Moon chat"
        assert body["system_prompt"] == ""  # empty is a real value, not "use the default"
        assert body["model"] == session["model"]
        assert client.get(f"/api/sessions/{session['id']}").json()["system_prompt"] == ""


def test_patch_rejects_bad_fields():
    with client_for() as client:
        sid = client.post("/api/sessions").json()["id"]
        for body in (
            {"model": ""},
            {"voice": ""},
            {"model": None},
            {"system_prompt": None},
            {"title": None},
            {"summary": "x"},
            {"model": 3},
            {"title": "x" * 201},
        ):
            assert client.patch(f"/api/sessions/{sid}", json=body).status_code == 422, body
