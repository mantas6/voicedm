import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest
from fakes import FakeBot, fake_upstreams, make_offer, make_offer_async
from fastapi.testclient import TestClient

from voicedm import bot as bot_module
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
        response = client.post("/api/offer", json=second)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["session_id"] == session_id
        assert "already active" in detail["message"]
        assert len(bot.calls) == 1
        assert client.get("/api/health").json()["call"] == {"session_id": session_id}


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


def test_session_in_a_call_cannot_be_deleted(offer):
    client, bot = client_for()
    with client:
        session_id = new_session(client)
        other = new_session(client)
        client.post("/api/offer", json={**offer, "session_id": session_id})
        response = client.delete(f"/api/sessions/{session_id}")
        assert response.status_code == 409
        assert "Hang up" in response.json()["detail"]
        assert client.get(f"/api/sessions/{session_id}").status_code == 200
        assert client.delete(f"/api/sessions/{other}").status_code == 204  # others can go


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
        models, voices = client.get("/api/models"), client.get("/api/voices")
    assert (models.status_code, voices.status_code) == (502, 502)
    assert models.json()["detail"] == (
        "The language model server (Ollama) returned an error (HTTP 503)"
    )
    assert voices.json()["detail"].startswith("The voice server (Chatterbox) returned an error")


def test_first_session_without_any_model_says_so():
    client, _ = client_for(lambda request: httpx.Response(200, json={"models": [], "voices": []}))
    with client:
        response = client.post("/api/sessions")
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "On the language model server (Ollama), no models are installed."
    )


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


# -- the one-call slot, driven concurrently on one event loop --


@asynccontextmanager
async def running(bot: FakeBot | object, handler=fake_upstreams, config: Config = CONFIG):
    """The app with its lifespan, and an HTTP client into it."""
    app = create_app(config, transport=httpx.MockTransport(handler), bot=bot)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://app.test") as client:
            yield client


async def offer_for(client: httpx.AsyncClient, session_id: int, **extra) -> httpx.Response:
    offer = await make_offer_async()
    response = await client.post("/api/offer", json={**offer, "session_id": session_id, **extra})
    await asyncio.sleep(0.05)  # let the call task start
    return response


async def call_ends(client: httpx.AsyncClient) -> None:
    async with asyncio.timeout(5):
        while (await client.get("/api/health")).json()["call"] is not None:  # noqa: ASYNC110
            await asyncio.sleep(0.05)


async def test_concurrent_offers_start_exactly_one_call():
    bot = FakeBot()
    async with running(bot) as client:
        session_id = (await client.post("/api/sessions")).json()["id"]
        offers = [await make_offer_async() for _ in range(5)]
        responses = await asyncio.gather(
            *(client.post("/api/offer", json={**o, "session_id": session_id}) for o in offers)
        )
        assert sorted(r.status_code for r in responses) == [200, 409, 409, 409, 409]
        assert len(bot.calls) == 1


async def test_slot_is_released_when_the_pipeline_fails_to_start():
    bot = FakeBot(fail=RuntimeError("no pipeline"))
    async with running(bot) as client:
        session_id = (await client.post("/api/sessions")).json()["id"]
        assert (await offer_for(client, session_id)).status_code == 200
        await call_ends(client)
        bot.fail = None
        assert (await offer_for(client, session_id)).status_code == 200
        assert len(bot.calls) == 2


async def test_slot_is_released_when_a_call_crashes():
    bot = FakeBot()
    async with running(bot) as client:
        session_id = (await client.post("/api/sessions")).json()["id"]
        assert (await offer_for(client, session_id)).status_code == 200
        bot.crash.set()
        await call_ends(client)
        bot.crash.clear()
        assert (await offer_for(client, session_id)).status_code == 200


async def test_slot_is_released_when_the_offer_fails():
    bot = FakeBot()
    async with running(bot) as client:
        session_id = (await client.post("/api/sessions")).json()["id"]
        offer = await make_offer_async()
        bad = {**offer, "type": "answer", "session_id": session_id}
        response = await client.post("/api/offer", json=bad)
        assert response.status_code == 400
        assert bot.calls == []
        assert (await offer_for(client, session_id)).status_code == 200


async def test_reconnect_replaces_its_own_dropped_call():
    bot = FakeBot()
    async with running(bot) as client:
        session_id = (await client.post("/api/sessions")).json()["id"]
        first = await offer_for(client, session_id)
        pc_id = first.json()["pc_id"]
        # Naming some other call does not get past the limit.
        assert (await offer_for(client, session_id, replaces="other")).status_code == 409
        second = await offer_for(client, session_id, replaces=pc_id)
        assert second.status_code == 200
        assert bot.ended == [("replaced", None, False)]
        assert len(bot.calls) == 2
        health = (await client.get("/api/health")).json()
        assert health["call"] == {"session_id": session_id}


async def test_shutdown_ends_the_call_and_asks_the_page_to_call_again():
    bot = FakeBot()
    async with running(bot) as client:
        session_id = (await client.post("/api/sessions")).json()["id"]
        assert (await offer_for(client, session_id)).status_code == 200
    assert [(reason, retry) for reason, _, retry in bot.ended] == [("shutdown", True)]


async def test_real_bot_frees_the_slot_when_the_client_never_connects(monkeypatch):
    """The real pipeline, minus speech-to-text: an offer whose client never completes
    the connection ends after the connect timeout, without any upstream request."""
    monkeypatch.setattr(bot_module, "CONNECT_TIMEOUT", 0.5)
    real = bot_module.Bot(CONFIG)

    async def no_model() -> None:  # no download; the model is never used: nothing is said
        real.whisper = object()

    real.load = no_model

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path in ("/api/tags", "/v1/audio/voices"):
            return fake_upstreams(request)
        raise AssertionError(f"unexpected request {request.url}")

    async with running(real, handler) as client:
        session_id = (await client.post("/api/sessions")).json()["id"]
        assert (await offer_for(client, session_id)).status_code == 200
        assert (await client.get("/api/health")).json()["call"] is not None
        await call_ends(client)


# -- PATCH validation against the installed models and voices --


def test_patch_rejects_a_model_or_voice_that_is_not_installed():
    client, _ = client_for()
    with client:
        session_id = new_session(client)
        response = client.patch(f"/api/sessions/{session_id}", json={"model": "gone:1b"})
        assert response.status_code == 422
        assert "gone:1b" in response.json()["detail"]
        response = client.patch(f"/api/sessions/{session_id}", json={"voice": "Nobody.wav"})
        assert response.status_code == 422
        assert "Nobody.wav" in response.json()["detail"]
        ok = client.patch(f"/api/sessions/{session_id}", json={"model": "second-model:7b"})
        assert ok.status_code == 200
        assert ok.json()["model"] == "second-model:7b"


def test_patch_is_not_blocked_while_the_listing_upstream_is_down():
    up = True

    def handler(request: httpx.Request) -> httpx.Response:
        if not up:
            raise httpx.ConnectError("connection refused", request=request)
        return fake_upstreams(request)

    client, _ = client_for(handler)
    with client:
        session_id = new_session(client)
        up = False
        response = client.patch(
            f"/api/sessions/{session_id}", json={"model": "any:1b", "voice": "Any.wav"}
        )
        assert response.status_code == 200
        assert (response.json()["model"], response.json()["voice"]) == ("any:1b", "Any.wav")
