"""FastAPI application: API routes and static files."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from voicedm import __version__
from voicedm.config import Config
from voicedm.upstream import Upstreams

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    config: Config | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the app. ``transport`` replaces the network for upstream calls in tests."""
    config = config or Config.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(transport=transport) as client:
            app.state.upstreams = Upstreams(config.ollama_url, config.chatterbox_url, client)
            yield

    app = FastAPI(title="voicedm", version=__version__, lifespan=lifespan)
    app.state.config = config

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        upstreams: Upstreams = request.app.state.upstreams
        return {"status": "ok", "upstreams": await upstreams.health()}

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
