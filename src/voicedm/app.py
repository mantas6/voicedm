"""FastAPI application: API routes and static files."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
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
from pydantic import BaseModel, ConfigDict, Field

from voicedm import __version__
from voicedm.config import DEFAULT_SYSTEM_PROMPT, Config
from voicedm.storage import Storage
from voicedm.upstream import UpstreamError, Upstreams

STATIC_DIR = Path(__file__).parent / "static"

SettingsHandler = Callable[[dict[str, str]], Awaitable[None]]


class ActiveCall:
    """The running call: which session it is on, and a way into its pipeline.

    Settings changed while the pipeline is still being set up are kept and applied
    once the bot attaches.
    """

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id
        self._handler: SettingsHandler | None = None
        self._pending: dict[str, str] = {}

    async def update_settings(self, changes: dict[str, str]) -> None:
        if self._handler is None:
            self._pending.update(changes)
        else:
            await self._handler(changes)

    async def attach(self, handler: SettingsHandler) -> None:
        """Called by the bot once its pipeline can take settings changes."""
        self._handler = handler
        pending, self._pending = self._pending, {}
        if pending:
            await handler(pending)

    def detach(self) -> None:
        self._handler = None


class CallRunner(Protocol):
    """What the app needs from the bot; tests substitute a fake."""

    async def load(self) -> None: ...

    async def run(
        self,
        connection: SmallWebRTCConnection,
        upstreams: Upstreams,
        store: Storage,
        call: ActiveCall,
    ) -> None: ...


class Offer(BaseModel):
    sdp: str
    type: str
    session_id: int


class NewSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Session whose settings to copy; by default the most recently used one.
    copy_from: int | None = None


class SessionPatch(BaseModel):
    """Fields to change. ``None`` defaults are never validated, so an explicit null is
    rejected; an empty system prompt is a valid prompt (no system message)."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default=None, max_length=200)
    model: str = Field(default=None, min_length=1, max_length=200)
    voice: str = Field(default=None, min_length=1, max_length=200)
    system_prompt: str = Field(default=None, max_length=20000)


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
    calls: dict[asyncio.Task, ActiveCall] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = Storage.in_dir(config.data_dir)
        await bot.load()
        try:
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
        finally:
            app.state.store.close()

    app = FastAPI(title="voicedm", version=__version__, lifespan=lifespan)
    app.state.config = config

    def upstreams_of(request: Request) -> Upstreams:
        return request.app.state.upstreams

    def store_of(request: Request) -> Storage:
        return request.app.state.store

    def active_call(session_id: int) -> ActiveCall | None:
        return next((c for c in calls.values() if c.session_id == session_id), None)

    def with_call_state(session: dict) -> dict:
        return {**session, "in_call": active_call(session["id"]) is not None}

    async def existing_session(request: Request, session_id: int) -> dict:
        session = await store_of(request).get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"No session {session_id}")
        return session

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

    @app.get("/api/sessions")
    async def list_sessions(request: Request) -> list[dict]:
        return [with_call_state(s) for s in await store_of(request).list_sessions()]

    @app.post("/api/sessions", status_code=201)
    async def new_session(request: Request, body: NewSession | None = None) -> dict:
        """A new, empty session with the settings of ``copy_from``, else of the most
        recently used session, else the configured or first available model and voice
        and the default system prompt."""
        store = store_of(request)
        if body and body.copy_from is not None:
            source = await existing_session(request, body.copy_from)
        else:
            source = await store.latest_session()
        if source:
            model, voice, prompt = source["model"], source["voice"], source["system_prompt"]
        else:
            upstreams = upstreams_of(request)
            try:
                model = await upstreams.resolve_model(config.default_model)
                voice = await upstreams.resolve_voice(config.default_voice)
            except UpstreamError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            prompt = DEFAULT_SYSTEM_PROMPT
        session = await store.create_session(model, voice, prompt)
        return {**with_call_state(session), "messages": []}

    @app.get("/api/sessions/{session_id}")
    async def get_session(request: Request, session_id: int) -> dict:
        session = await existing_session(request, session_id)
        messages = await store_of(request).messages(session_id)
        return {**with_call_state(session), "messages": messages}

    @app.patch("/api/sessions/{session_id}")
    async def patch_session(request: Request, session_id: int, body: SessionPatch) -> dict:
        """Change settings; a call running on the session picks them up from its next
        turn."""
        await existing_session(request, session_id)
        changes = body.model_dump(exclude_unset=True)
        store = store_of(request)
        session = await store.update_session(session_id, **changes)
        call = active_call(session_id)
        settings = {k: v for k, v in changes.items() if k != "title"}
        if call and settings:
            await call.update_settings(settings)
        messages = await store.messages(session_id)
        return {**with_call_state(session), "messages": messages}

    @app.post("/api/offer")
    async def offer(request: Request, body: Offer) -> dict[str, Any]:
        # Only one call at a time (the proper 409 handling is a later milestone).
        if calls:
            raise HTTPException(status_code=409, detail="A call is already active")
        await existing_session(request, body.session_id)
        upstreams, store = upstreams_of(request), store_of(request)

        async def start_call(connection: SmallWebRTCConnection) -> None:
            call = ActiveCall(body.session_id)
            task = asyncio.create_task(bot.run(connection, upstreams, store, call))
            calls[task] = call
            task.add_done_callback(finished)

        def finished(task: asyncio.Task) -> None:
            calls.pop(task, None)
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
