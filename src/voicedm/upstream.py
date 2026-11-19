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


OLLAMA = "the language model server (Ollama)"
CHATTERBOX = "the voice server (Chatterbox)"


class UpstreamError(RuntimeError):
    """An upstream request failed or returned something unusable.

    ``kind`` says how: ``unreachable`` (no connection), ``timeout``, ``status`` (an
    HTTP error status, in ``status``, with the server's own message in ``detail``) or
    ``invalid`` (an answer that could not be used) or ``empty`` (nothing installed).
    """

    def __init__(
        self, message: str, kind: str = "invalid", status: int | None = None, detail: str = ""
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.detail = detail


def error_detail(response: httpx.Response) -> str:
    """The short reason an upstream gives in an error response, if any."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200].strip()
    if isinstance(body, dict):
        for key in ("error", "detail", "message"):
            if isinstance(body.get(key), str):
                return body[key][:200]
    return ""


def explain(error: UpstreamError, service: str, what: str = "") -> str:
    """A message for the page about a failed request to ``service`` (``OLLAMA`` or
    ``CHATTERBOX``). ``what`` names the model or voice the request was for."""
    if error.kind == "unreachable":
        return f"Cannot reach {service}."
    if error.kind == "empty":
        listed = "no models are installed" if service == OLLAMA else "no voices are offered"
        return f"On {service}, {listed}."
    if error.kind == "timeout":
        return f"{service[0].upper()}{service[1:]} did not answer in time."
    if error.kind == "status":
        if error.status == 404 and what:
            kind = "model" if service == OLLAMA else "voice"
            return f"The {kind} “{what}” is not available on {service}. Pick another in Settings."
        detail = f": {error.detail}" if error.detail else ""
        return f"{service[0].upper()}{service[1:]} returned an error (HTTP {error.status}){detail}"
    return f"{service[0].upper()}{service[1:]} sent an answer that could not be used."


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
            raise UpstreamError(f"{method} {url}: timed out after {limit:g}s", "timeout") from e
        except httpx.HTTPStatusError as e:
            status, detail = e.response.status_code, error_detail(e.response)
            raise UpstreamError(
                f"{method} {url}: HTTP {status}: {detail}", "status", status, detail
            ) from e
        except httpx.TransportError as e:
            kind = "timeout" if isinstance(e, httpx.TimeoutException) else "unreachable"
            raise UpstreamError(f"{method} {url}: {e!r}", kind) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"{method} {url}: {e!r}") from e
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
            raise UpstreamError("Ollama has no models installed", "empty")
        return models[0]

    async def resolve_voice(self, preferred: str | None) -> str:
        """``preferred`` if set, else the first listed voice."""
        if preferred:
            return preferred
        voices = await self.list_voices()
        if not voices:
            raise UpstreamError("Chatterbox lists no voices", "empty")
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

    async def warm_up_one(self, name: str, what: str) -> str | None:
        """Warm ``name`` ("ollama" with model ``what``, or "chatterbox" with voice
        ``what``). None if it answered, else a message saying why not."""
        try:
            if name == "ollama":
                await self.warm_up_ollama(what)
            else:
                await self.warm_up_chatterbox(what)
        except UpstreamError as e:
            logger.warning(f"Warm-up of {name} failed: {e}")
            return explain(e, OLLAMA if name == "ollama" else CHATTERBOX, what)
        return None

    async def warm_up(self, model: str, voice: str) -> dict[str, bool]:
        """Warm both services concurrently; report which ones answered."""
        ollama, chatterbox = await asyncio.gather(
            self.warm_up_one("ollama", model), self.warm_up_one("chatterbox", voice)
        )
        return {"ollama": ollama is None, "chatterbox": chatterbox is None}

    async def check_installed(self, model: str | None, voice: str | None) -> str | None:
        """Whether ``model`` is installed in Ollama and ``voice`` offered by Chatterbox.

        A message naming what is missing, or None. A service that cannot be asked
        right now does not block the change: it is only checked when it answers.
        """
        checks = [
            (model, self.list_models, "model", "Ollama"),
            (voice, self.list_voices, "voice", "Chatterbox"),
        ]
        for value, listing, kind, service in checks:
            if value is None:
                continue
            try:
                available = await listing()
            except UpstreamError as e:
                logger.info(f"Not checking the {kind} {value!r}: {service} not available: {e}")
                continue
            if value not in available:
                return f"The {kind} “{value}” is not available on {service}."
        return None
