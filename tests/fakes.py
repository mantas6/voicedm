"""Test doubles shared by the test modules."""

import asyncio

import httpx
from aiortc import RTCConfiguration, RTCPeerConnection


class FakeBot:
    """Stands in for the pipeline: records calls and their settings changes, and stays
    'in a call' until cancelled."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.loaded = False
        self.calls: list[tuple[str, int]] = []
        self.settings_changes: list[dict[str, str]] = []
        self.ended: list[tuple[str, str | None, bool]] = []
        self.fail = fail  # raised when a call starts, as a broken pipeline would
        self.crash = asyncio.Event()  # set to make the running call fail

    async def load(self) -> None:
        self.loaded = True

    async def run(self, connection, upstreams, store, call) -> None:
        self.calls.append((connection.pc_id, call.session_id))
        if self.fail:
            raise self.fail
        done = asyncio.Event()

        async def apply(changes: dict[str, str]) -> None:
            self.settings_changes.append(changes)

        async def end(reason: str, message: str | None, retry: bool) -> None:
            self.ended.append((reason, message, retry))
            done.set()

        await call.attach(apply, end)
        try:
            crash = asyncio.create_task(self.crash.wait())
            finished = asyncio.create_task(done.wait())
            await asyncio.wait({crash, finished}, return_when=asyncio.FIRST_COMPLETED)
            crash.cancel()
            finished.cancel()
            if self.crash.is_set():
                raise RuntimeError("the pipeline crashed")
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


async def make_offer_async() -> dict:
    """A real browser-like SDP offer (audio + data channel), host candidates only."""
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    pc.addTransceiver("audio", direction="sendrecv")
    pc.createDataChannel("messages")
    await pc.setLocalDescription(await pc.createOffer())
    description = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    await pc.close()
    return description


def make_offer() -> dict:
    return asyncio.run(make_offer_async())
