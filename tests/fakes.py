"""Test doubles shared by the test modules."""

import asyncio

import httpx
from aiortc import RTCConfiguration, RTCPeerConnection


class FakeBot:
    """Stands in for the pipeline: records calls and their settings changes, and stays
    'in a call' until cancelled."""

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[tuple[str, int]] = []
        self.settings_changes: list[dict[str, str]] = []

    async def load(self) -> None:
        self.loaded = True

    async def run(self, connection, upstreams, store, call) -> None:
        self.calls.append((connection.pc_id, call.session_id))

        async def apply(changes: dict[str, str]) -> None:
            self.settings_changes.append(changes)

        await call.attach(apply)
        try:
            await asyncio.Event().wait()
        finally:
            call.detach()


TAGS = {"models": [{"name": "first-model:latest"}, {"name": "second-model:7b"}]}
VOICES = {"voices": ["Alpha.wav", "Beta.wav"]}


def fake_upstreams(request: httpx.Request) -> httpx.Response:
    """Ollama and Chatterbox listing endpoints with two models and two voices."""
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
