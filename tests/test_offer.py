from dataclasses import replace

import httpx
import pytest
from fakes import FakeBot, fake_upstreams, make_offer
from fastapi.testclient import TestClient

from voicedm.app import ActiveCall, create_app
from voicedm.config import Config

CONFIG = Config(ollama_url="http://ollama.test", chatterbox_url="http://tts.test")


@pytest.fixture(scope="module")
def offer() -> dict:
    return make_offer()


def client_for(handler=fake_upstreams, config: Config = CONFIG, bot: FakeBot | None = None):
    bot = bot or FakeBot()
    app = create_app(config, transport=httpx.MockTransport(handler), bot=bot)
    return TestClient(app), bot


def new_session(client: TestClient) -> int:
    response = client.post("/api/sessions")
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_startup_loads_bot():
    client, bot = client_for()
    with client:
        assert bot.loaded


def test_offer_returns_answer_and_starts_call_on_the_session(offer):
    client, bot = client_for()
    with client:
        session_id = new_session(client)
        response = client.post("/api/offer", json={**offer, "session_id": session_id})
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["type"] == "answer"
        assert "m=audio" in answer["sdp"]
        assert "m=application" in answer["sdp"]  # the data channel for RTVI messages
        assert bot.calls == [(answer["pc_id"], session_id)]
        assert client.get(f"/api/sessions/{session_id}").json()["in_call"] is True


def test_offer_for_unknown_session_is_not_found(offer):
    client, bot = client_for()
    with client:
        response = client.post("/api/offer", json={**offer, "session_id": 42})
    assert response.status_code == 404
    assert bot.calls == []


def test_second_offer_during_a_call_is_refused(offer):
    client, bot = client_for()
    with client:
        session_id = new_session(client)
        first = {**offer, "session_id": session_id}
        assert client.post("/api/offer", json=first).status_code == 200
        second = {**make_offer(), "session_id": session_id}
        assert client.post("/api/offer", json=second).status_code == 409
        assert len(bot.calls) == 1


def test_offer_rejects_malformed_body(offer):
    client, _ = client_for()
    with client:
        assert client.post("/api/offer", json={"type": "offer"}).status_code == 422
        assert client.post("/api/offer", json=offer).status_code == 422  # no session_id


def test_patch_during_a_call_reaches_the_pipeline(offer):
    client, bot = client_for()
    with client:
        session_id = new_session(client)
        other = new_session(client)
        client.post("/api/offer", json={**offer, "session_id": session_id})
        client.patch(f"/api/sessions/{session_id}", json={"voice": "Beta.wav", "title": "x"})
        client.patch(f"/api/sessions/{session_id}", json={"system_prompt": ""})
        client.patch(f"/api/sessions/{session_id}", json={"title": "only a title"})
        client.patch(f"/api/sessions/{other}", json={"model": "second-model:7b"})
    assert bot.settings_changes == [{"voice": "Beta.wav"}, {"system_prompt": ""}]


def test_configured_defaults_are_used_without_listing():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    config = replace(CONFIG, default_model="my-model", default_voice="my-voice.wav")
    client, _ = client_for(handler, config)
    with client:
        session = client.post("/api/sessions").json()
    assert (session["model"], session["voice"]) == ("my-model", "my-voice.wav")


def test_models_and_voices_are_proxied():
    client, _ = client_for()
    with client:
        assert client.get("/api/models").json() == ["first-model:latest", "second-model:7b"]
        assert client.get("/api/voices").json() == ["Alpha.wav", "Beta.wav"]


def test_listing_when_upstream_down_is_bad_gateway():
    client, _ = client_for(lambda request: httpx.Response(503))
    with client:
        assert client.get("/api/models").status_code == 502
        assert client.get("/api/voices").status_code == 502


async def test_settings_changed_before_the_pipeline_is_ready_are_applied_on_attach():
    call = ActiveCall(session_id=1)
    await call.update_settings({"voice": "Beta.wav"})
    await call.update_settings({"model": "second-model:7b"})
    applied = []

    async def apply(changes):
        applied.append(changes)

    await call.attach(apply)
    await call.update_settings({"system_prompt": ""})
    call.detach()
    await call.update_settings({"voice": "Alpha.wav"})  # after the call: not applied
    assert applied == [{"voice": "Beta.wav", "model": "second-model:7b"}, {"system_prompt": ""}]
