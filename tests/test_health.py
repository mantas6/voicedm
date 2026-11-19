import asyncio

import httpx
from fakes import FakeBot
from fastapi.testclient import TestClient

from voicedm import upstream
from voicedm.app import create_app
from voicedm.config import Config

CONFIG = Config(ollama_url="http://ollama.test", chatterbox_url="http://tts.test")


def client_for(handler) -> TestClient:
    return TestClient(create_app(CONFIG, transport=httpx.MockTransport(handler), bot=FakeBot()))


def get_health(handler) -> dict:
    with client_for(handler) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    return response.json()


def test_both_upstreams_up():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"voices": []})
        return httpx.Response(404)

    body = get_health(handler)
    assert body == {"status": "ok", "upstreams": {"ollama": True, "chatterbox": True}}
    assert sorted(seen) == ["http://ollama.test/api/tags", "http://tts.test/v1/audio/voices"]


def test_unreachable_upstreams_reported_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    body = get_health(handler)
    assert body == {"status": "ok", "upstreams": {"ollama": False, "chatterbox": False}}


def test_error_status_reported_down():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ollama.test":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(503)

    body = get_health(handler)
    assert body["upstreams"] == {"ollama": True, "chatterbox": False}


def test_slow_upstream_times_out(monkeypatch):
    monkeypatch.setattr(upstream, "HEALTH_TIMEOUT", 0.1)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "tts.test":
            await asyncio.sleep(10)
        return httpx.Response(200, json={})

    async def check() -> dict[str, bool]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ups = upstream.Upstreams(CONFIG.ollama_url, CONFIG.chatterbox_url, client)
            return await asyncio.wait_for(ups.health(), timeout=2)

    assert asyncio.run(check()) == {"ollama": True, "chatterbox": False}


def test_index_page_served():
    with client_for(lambda request: httpx.Response(200)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "voicedm" in response.text
