"""Pipecat TTS service for the Chatterbox TTS server.

The server's OpenAI-style ``/v1/audio/speech`` endpoint returns a complete WAV file
per request (``wav``, ``mp3`` or ``opus``; no raw PCM), so this service asks for
``wav`` once per sentence, parses the RIFF header and yields 16-bit PCM frames.
Pipecat's sentence aggregation feeds it text, and the base class resamples the
frames if the server's rate differs from the pipeline's output rate.
"""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx
import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

SPEECH_PATH = "/v1/audio/speech"
CHUNK_SECONDS = 0.2  # audio per yielded frame
STOP_FRAME_MARGIN = 1.0  # seconds


class WavError(ValueError):
    """The response is not a WAV file this service can play."""


@dataclass(frozen=True)
class PcmAudio:
    pcm: bytes  # 16-bit signed little-endian, mono
    sample_rate: int


def parse_wav(data: bytes) -> PcmAudio:
    """Extract mono 16-bit PCM from a WAV file.

    Walks the RIFF chunks instead of assuming a 44-byte header, so extra chunks
    (``LIST``, ``fact``, ...) and ``WAVE_FORMAT_EXTENSIBLE`` headers are fine.
    Multi-channel audio is mixed down to mono; other sample widths are rejected.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError(f"not a WAV file (starts with {data[:12]!r})")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise WavError(f"unreadable WAV: {e}") from e
    if rate <= 0:
        raise WavError(f"invalid WAV sample rate: {rate}")
    if width != 2:
        raise WavError(f"unsupported WAV sample width: {8 * width} bits (need 16)")
    if channels > 1:
        samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % (2 * channels)], dtype="<i2")
        pcm = samples.reshape(-1, channels).mean(axis=1).astype("<i2").tobytes()
    return PcmAudio(pcm=pcm, sample_rate=rate)


class ChatterboxTTSService(TTSService):
    """Speaks each sentence with one ``/v1/audio/speech`` request."""

    Settings = TTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str,
        voice: str,
        client: httpx.AsyncClient,
        timeout: float = 120.0,
        model: str = "chatterbox",
        **kwargs,
    ) -> None:
        settings = self.Settings(model=model, voice=voice, language=None)
        kwargs.setdefault("push_start_frame", True)
        kwargs.setdefault("push_stop_frames", True)
        # The base class closes a reply's audio context after this long without new
        # audio, and drops what arrives later, including our error frames. A slow LLM
        # or TTS leaves gaps of several seconds between sentences, so the default
        # (3 s) is too short; outlast the request timeout instead.
        kwargs.setdefault("stop_frame_timeout_s", timeout + STOP_FRAME_MARGIN)
        super().__init__(settings=settings, **kwargs)
        self._url = f"{base_url.rstrip('/')}{SPEECH_PATH}"
        self._client = client
        self._timeout = timeout

    def can_generate_metrics(self) -> bool:
        return True

    async def synthesize(self, text: str) -> PcmAudio:
        """Fetch and parse the WAV for ``text``. Raises on any failure."""
        body = {
            "model": self._settings.model,
            "input": text,
            "voice": self._settings.voice,
            "response_format": "wav",
        }
        # httpx timeouts apply per phase; asyncio.timeout bounds the whole request.
        async with asyncio.timeout(self._timeout):
            response = await self._client.post(self._url, json=body, timeout=self._timeout)
        if not response.is_success:
            detail = response.text[:200].strip()
            raise RuntimeError(f"Chatterbox HTTP {response.status_code}: {detail}")
        return parse_wav(response.content)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            audio = await self.synthesize(text)
        except TimeoutError:
            yield ErrorFrame(error=f"Chatterbox TTS timed out after {self._timeout:g}s")
            return
        except httpx.HTTPError as e:
            yield ErrorFrame(error=f"Chatterbox TTS request failed: {e!r}")
            return
        except (RuntimeError, WavError) as e:
            yield ErrorFrame(error=f"Chatterbox TTS failed: {e}")
            return

        await self.stop_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        seconds = len(audio.pcm) / 2 / audio.sample_rate
        logger.debug(f"{self}: got {seconds:.1f}s of audio at {audio.sample_rate} Hz")
        # Frames carry the server's rate; the base class resamples to the pipeline's.
        step = max(1, int(audio.sample_rate * CHUNK_SECONDS)) * 2
        for start in range(0, len(audio.pcm), step):
            yield TTSAudioRawFrame(
                audio.pcm[start : start + step],
                audio.sample_rate,
                1,
                context_id=context_id,
            )
