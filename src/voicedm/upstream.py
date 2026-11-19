"""Access to the upstream services: Ollama and the Chatterbox TTS server."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

HEALTH_TIMEOUT = 3.0
LIST_TIMEOUT = 10.0
# Loading a large model into memory can take minutes on a cold Ollama.
OLLAMA_WARMUP_TIMEOUT = 300.0
WARMUP_TEXT = "Hi."


class UpstreamError(RuntimeError):
    """An upstream request failed or returned something unusable."""


class Upstreams:
    """Thin client for Ollama and Chatterbox over one shared ``httpx.AsyncClient``."""

    def __init__(
        self,
        ollama_url: str,
        chatterbox_url: str,
        client: httpx.AsyncClient,
        tts_timeout: float = 120.0,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.chatterbox_url = chatterbox_url.rstrip("/")
        self.client = client
        self.tts_timeout = tts_timeout

    async def _request(self, method: str, url: str, limit: float, **kwargs: Any) -> Any:
        """Send a request and return its JSON body, raising ``UpstreamError`` on failure."""
        # httpx timeouts apply per phase; asyncio.timeout bounds the whole request.
        try:
            async with asyncio.timeout(limit):
                response = await self.client.request(method, url, timeout=limit, **kwargs)
                response.raise_for_status()
        except TimeoutError as e:
            raise UpstreamError(f"{method} {url}: timed out after {limit:g}s") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"{method} {url}: {e}") from e
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                return response.json()
            except ValueError as e:
                raise UpstreamError(f"{method} {url}: invalid JSON: {e}") from e
        return None

    async def _reachable(self, url: str) -> bool:
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

    async def list_models(self) -> list[str]:
        """Installed Ollama models, in the order Ollama lists them."""
        body = await self._request("GET", f"{self.ollama_url}/api/tags", LIST_TIMEOUT)
        try:
            return [m["name"] for m in body["models"]]
        except (TypeError, KeyError) as e:
            raise UpstreamError(f"unexpected /api/tags response: {e!r}") from e

    async def list_voices(self) -> list[str]:
        """Voices offered by the Chatterbox server."""
        url = f"{self.chatterbox_url}/v1/audio/voices"
        body = await self._request("GET", url, LIST_TIMEOUT)
        try:
            voices = body["voices"]
            return [v if isinstance(v, str) else v["name"] for v in voices]
        except (TypeError, KeyError) as e:
            raise UpstreamError(f"unexpected /v1/audio/voices response: {e!r}") from e

    async def resolve_model(self, preferred: str | None) -> str:
        """``preferred`` if set, else the first installed model."""
        if preferred:
            return preferred
        models = await self.list_models()
        if not models:
            raise UpstreamError("Ollama has no models installed")
        return models[0]

    async def resolve_voice(self, preferred: str | None) -> str:
        """``preferred`` if set, else the first listed voice."""
        if preferred:
            return preferred
        voices = await self.list_voices()
        if not voices:
            raise UpstreamError("Chatterbox lists no voices")
        return voices[0]

    async def chat(self, model: str, messages: list[dict[str, str]], limit: float) -> str:
        """A one-off chat completion on Ollama's native API, not streamed; the reply text.

        Used for side requests such as summarisation, apart from the call's own LLM.
        ``limit`` bounds the whole request, in seconds.
        """
        url = f"{self.ollama_url}/api/chat"
        body = {"model": model, "messages": messages, "stream": False}
        reply = await self._request("POST", url, limit, json=body)
        try:
            content = reply["message"]["content"]
        except (TypeError, KeyError) as e:
            raise UpstreamError(f"unexpected /api/chat response: {e!r}") from e
        if not isinstance(content, str):
            raise UpstreamError("unexpected /api/chat response: content is not text")

        def seconds(key: str) -> float:
            return (reply.get(key) or 0) / 1e9  # Ollama reports nanoseconds

        logger.info(
            f"Ollama chat on {model}: {reply.get('prompt_eval_count')} prompt tokens in "
            f"{seconds('prompt_eval_duration'):.1f}s, {reply.get('eval_count')} reply tokens "
            f"in {seconds('eval_duration'):.1f}s, {seconds('total_duration'):.1f}s in total"
        )
        return content

    async def warm_up_ollama(self, model: str) -> None:
        """Load ``model`` into memory: a generate request without a prompt does only that."""
        url = f"{self.ollama_url}/api/generate"
        await self._request(
            "POST", url, OLLAMA_WARMUP_TIMEOUT, json={"model": model, "stream": False}
        )

    async def warm_up_chatterbox(self, voice: str) -> None:
        """Synthesize one word so a stopped TTS server starts and loads its model."""
        url = f"{self.chatterbox_url}/v1/audio/speech"
        body = {"model": "chatterbox", "input": WARMUP_TEXT, "voice": voice}
        await self._request("POST", url, self.tts_timeout, json=body)

    async def warm_up(self, model: str, voice: str) -> dict[str, bool]:
        """Warm both services concurrently; report which ones answered."""

        async def attempt(name: str, coro: Any) -> bool:
            try:
                await coro
            except UpstreamError as e:
                logger.warning(f"Warm-up of {name} failed: {e}")
                return False
            return True

        ollama, chatterbox = await asyncio.gather(
            attempt("Ollama", self.warm_up_ollama(model)),
            attempt("Chatterbox", self.warm_up_chatterbox(voice)),
        )
        return {"ollama": ollama, "chatterbox": chatterbox}
