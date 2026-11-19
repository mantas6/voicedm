"""FastAPI application: API routes and static files."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pydantic import BaseModel

from voicedm import __version__
from voicedm.config import Config
from voicedm.upstream import UpstreamError, Upstreams

STATIC_DIR = Path(__file__).parent / "static"


class CallRunner(Protocol):
    """What the app needs from the bot; tests substitute a fake."""

    async def load(self) -> None: ...

    async def run(
        self, connection: SmallWebRTCConnection, upstreams: Upstreams, model: str, voice: str
    ) -> None: ...


class Offer(BaseModel):
    sdp: str
    type: str


def create_app(
    config: Config | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    bot: CallRunner | None = None,
) -> FastAPI:
    """Build the app.

    ``transport`` replaces the network for upstream calls and ``bot`` replaces the voice
    pipeline (and its model loading) in tests.
    """
    config = config or Config.from_env()
    if bot is None:
        from voicedm.bot import Bot

        bot = Bot(config)

    ice_servers = [IceServer(urls=config.stun_url)] if config.stun_url else None
    webrtc = SmallWebRTCRequestHandler(ice_servers=ice_servers)
    calls: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await bot.load()
        async with httpx.AsyncClient(transport=transport) as client:
            app.state.upstreams = Upstreams(
                config.ollama_url, config.chatterbox_url, client, tts_timeout=config.tts_timeout
            )
            yield
            for task in list(calls):
                task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*calls)
            await webrtc.close()

    app = FastAPI(title="voicedm", version=__version__, lifespan=lifespan)
    app.state.config = config

    def upstreams_of(request: Request) -> Upstreams:
        return request.app.state.upstreams

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        return {"status": "ok", "upstreams": await upstreams_of(request).health()}

    @app.get("/api/models")
    async def models(request: Request) -> list[str]:
        try:
            return await upstreams_of(request).list_models()
        except UpstreamError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.get("/api/voices")
    async def voices(request: Request) -> list[str]:
        try:
            return await upstreams_of(request).list_voices()
        except UpstreamError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.post("/api/offer")
    async def offer(request: Request, body: Offer) -> dict[str, Any]:
        # Only one call at a time (the proper 409 handling is a later milestone).
        if calls:
            raise HTTPException(status_code=409, detail="A call is already active")
        upstreams = upstreams_of(request)
        try:
            model = await upstreams.resolve_model(config.default_model)
            voice = await upstreams.resolve_voice(config.default_voice)
        except UpstreamError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

        async def start_call(connection: SmallWebRTCConnection) -> None:
            task = asyncio.create_task(bot.run(connection, upstreams, model, voice))
            calls.add(task)
            task.add_done_callback(finished)

        def finished(task: asyncio.Task) -> None:
            calls.discard(task)
            if not task.cancelled() and task.exception():
                logger.opt(exception=task.exception()).error("Call failed")

        answer = await webrtc.handle_web_request(
            SmallWebRTCRequest(sdp=body.sdp, type=body.type), start_call
        )
        if answer is None:
            raise HTTPException(status_code=500, detail="No SDP answer")
        return answer

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
