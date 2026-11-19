import asyncio
from dataclasses import replace

import httpx
import pytest
from aiortc import RTCConfiguration, RTCPeerConnection
from fakes import FakeBot
from fastapi.testclient import TestClient

from voicedm.app import create_app
from voicedm.config import Config

CONFIG = Config(ollama_url="http://ollama.test", chatterbox_url="http://tts.test")
TAGS = {"models": [{"name": "first-model:latest"}, {"name": "second-model:7b"}]}
VOICES = {"voices": ["Alpha.wav", "Beta.wav"]}


def fake_upstreams(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/tags":
        return httpx.Response(200, json=TAGS)
    if request.url.path == "/v1/audio/voices":
        return httpx.Response(200, json=VOICES)
    return httpx.Response(404)


def make_offer() -> dict:
    """A real browser-like SDP offer (audio + data channel), host candidates only."""

    async def offer() -> dict:
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pc.addTransceiver("audio", direction="sendrecv")
        pc.createDataChannel("messages")
        await pc.setLocalDescription(await pc.createOffer())
        description = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        await pc.close()
        return description

    return asyncio.run(offer())


@pytest.fixture(scope="module")
def offer() -> dict:
    return make_offer()


def client_for(handler=fake_upstreams, config: Config = CONFIG, bot: FakeBot | None = None):
    bot = bot or FakeBot()
    app = create_app(config, transport=httpx.MockTransport(handler), bot=bot)
    return TestClient(app), bot


def test_startup_loads_bot():
    client, bot = client_for()
    with client:
        assert bot.loaded


def test_offer_returns_answer_and_starts_call(offer):
    client, bot = client_for()
    with client:
        response = client.post("/api/offer", json=offer)
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["type"] == "answer"
        assert "m=audio" in answer["sdp"]
        assert "m=application" in answer["sdp"]  # the data channel for RTVI messages
        assert bot.calls == [(answer["pc_id"], "first-model:latest", "Alpha.wav")]


def test_configured_defaults_are_used_without_listing(offer):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    config = replace(CONFIG, default_model="my-model", default_voice="my-voice.wav")
    client, bot = client_for(handler, config)
    with client:
        assert client.post("/api/offer", json=offer).status_code == 200
    assert [call[1:] for call in bot.calls] == [("my-model", "my-voice.wav")]


def test_second_offer_during_a_call_is_refused(offer):
    client, bot = client_for()
    with client:
        assert client.post("/api/offer", json=offer).status_code == 200
        response = client.post("/api/offer", json=make_offer())
        assert response.status_code == 409
        assert len(bot.calls) == 1


def test_offer_without_a_model_is_unavailable(offer):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, bot = client_for(handler)
    with client:
        response = client.post("/api/offer", json=offer)
    assert response.status_code == 503
    assert "connection refused" in response.json()["detail"]
    assert bot.calls == []


def test_offer_rejects_malformed_body():
    client, _ = client_for()
    with client:
        assert client.post("/api/offer", json={"type": "offer"}).status_code == 422


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
