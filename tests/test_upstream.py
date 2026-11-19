import asyncio
import json

import httpx
import pytest

from voicedm import upstream
from voicedm.upstream import UpstreamError, Upstreams

TAGS = {"models": [{"name": "first-model:latest"}, {"name": "second-model:7b"}]}
VOICES = {"status": "ok", "voices": ["Alpha.wav", "Beta.wav"]}


def make(handler, tts_timeout: float = 5.0) -> tuple[Upstreams, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Upstreams("http://ollama.test", "http://tts.test", client, tts_timeout), client


def fake_upstreams(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/tags":
        return httpx.Response(200, json=TAGS)
    if request.url.path == "/v1/audio/voices":
        return httpx.Response(200, json=VOICES)
    return httpx.Response(404)


async def test_list_models_and_voices():
    ups, client = make(fake_upstreams)
    async with client:
        assert await ups.list_models() == ["first-model:latest", "second-model:7b"]
        assert await ups.list_voices() == ["Alpha.wav", "Beta.wav"]


async def test_resolve_prefers_configured_values_without_asking():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    ups, client = make(handler)
    async with client:
        assert await ups.resolve_model("my-model") == "my-model"
        assert await ups.resolve_voice("my-voice.wav") == "my-voice.wav"


async def test_resolve_falls_back_to_first_listed():
    ups, client = make(fake_upstreams)
    async with client:
        assert await ups.resolve_model(None) == "first-model:latest"
        assert await ups.resolve_voice(None) == "Alpha.wav"


async def test_resolve_with_nothing_installed_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [], "voices": []})

    ups, client = make(handler)
    async with client:
        with pytest.raises(UpstreamError, match="no models"):
            await ups.resolve_model(None)
        with pytest.raises(UpstreamError, match="no voices"):
            await ups.resolve_voice(None)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, text="not json", headers={"content-type": "application/json"}),
    ],
)
async def test_listing_errors_raise_upstream_error(response):
    ups, client = make(lambda request: response)
    async with client:
        with pytest.raises(UpstreamError):
            await ups.list_models()
        with pytest.raises(UpstreamError):
            await ups.list_voices()


async def test_listing_unreachable_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    ups, client = make(handler)
    async with client:
        with pytest.raises(UpstreamError, match="connection refused"):
            await ups.list_models()


async def test_warm_up_loads_model_and_synthesizes_a_word():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(200, content=b"RIFF....WAVE", headers={"content-type": "audio/wav"})

    ups, client = make(handler)
    async with client:
        assert await ups.warm_up("some-model", "Beta.wav") == {"ollama": True, "chatterbox": True}
    by_path = dict(seen)
    assert by_path["/api/generate"] == {"model": "some-model", "stream": False}
    speech = by_path["/v1/audio/speech"]
    assert speech["voice"] == "Beta.wav"
    assert speech["input"]


async def test_warm_up_reports_each_failure_separately():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, content=b"RIFF")

    ups, client = make(handler)
    async with client:
        assert await ups.warm_up("missing", "Alpha.wav") == {"ollama": False, "chatterbox": True}


async def test_warm_up_times_out(monkeypatch):
    monkeypatch.setattr(upstream, "OLLAMA_WARMUP_TIMEOUT", 0.1)

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    ups, client = make(handler, tts_timeout=0.1)
    async with client:
        result = await asyncio.wait_for(ups.warm_up("m", "v"), timeout=2)
    assert result == {"ollama": False, "chatterbox": False}
