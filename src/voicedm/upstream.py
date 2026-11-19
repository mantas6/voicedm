"""Access to the upstream services: Ollama and the Chatterbox TTS server."""

from __future__ import annotations

import asyncio

import httpx

HEALTH_TIMEOUT = 3.0


class Upstreams:
    """Thin client for Ollama and Chatterbox over one shared ``httpx.AsyncClient``."""

    def __init__(self, ollama_url: str, chatterbox_url: str, client: httpx.AsyncClient) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.chatterbox_url = chatterbox_url.rstrip("/")
        self.client = client

    async def _reachable(self, url: str) -> bool:
        # httpx timeouts apply per phase; asyncio.timeout bounds the whole request.
        timeout = HEALTH_TIMEOUT
        try:
            async with asyncio.timeout(timeout):
                response = await self.client.get(url, timeout=timeout)
        except (httpx.HTTPError, TimeoutError):
            return False
        return response.is_success

    async def ollama_ok(self) -> bool:
        return await self._reachable(f"{self.ollama_url}/api/tags")

    async def chatterbox_ok(self) -> bool:
        return await self._reachable(f"{self.chatterbox_url}/v1/audio/voices")

    async def health(self) -> dict[str, bool]:
        ollama, chatterbox = await asyncio.gather(self.ollama_ok(), self.chatterbox_ok())
        return {"ollama": ollama, "chatterbox": chatterbox}
