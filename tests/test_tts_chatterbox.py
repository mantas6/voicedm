import asyncio
import io
import json
import struct
import wave

import httpx
import numpy as np
import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.tests.utils import run_test

from voicedm.tts_chatterbox import ChatterboxTTSService, WavError, parse_wav

PCM_GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def tone(rate: int, seconds: float = 0.5, channels: int = 1) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    mono = (np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2")
    return np.repeat(mono, channels).tobytes()


def simple_wav(pcm: bytes, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def chunk(cid: bytes, body: bytes) -> bytes:
    pad = b"\x00" if len(body) % 2 else b""
    return cid + struct.pack("<I", len(body)) + body + pad


def riff(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def fmt_pcm(rate: int, channels: int = 1) -> bytes:
    return struct.pack("<HHIIHH", 1, channels, rate, rate * 2 * channels, 2 * channels, 16)


def fmt_extensible(rate: int, channels: int = 1) -> bytes:
    base = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * 2 * channels, 2 * channels, 16)
    return base + struct.pack("<HHI", 22, 16, 0) + struct.pack("<H", 1) + PCM_GUID_TAIL


# -- WAV parsing --


def test_parse_plain_44_byte_header():
    pcm = tone(24000)
    data = simple_wav(pcm)
    assert data.index(b"data") == 36  # the classic 44-byte layout
    audio = parse_wav(data)
    assert audio.sample_rate == 24000
    assert audio.pcm == pcm


def test_parse_skips_extra_chunks():
    pcm = tone(22050)
    data = riff(
        chunk(b"fmt ", fmt_pcm(22050)),
        chunk(b"LIST", b"INFOISFT" + struct.pack("<I", 5) + b"test\x00"),
        chunk(b"data", pcm),
    )
    assert data.index(b"data") > 44
    audio = parse_wav(data)
    assert audio.sample_rate == 22050
    assert audio.pcm == pcm


def test_parse_wave_format_extensible():
    pcm = tone(24000)
    audio = parse_wav(riff(chunk(b"fmt ", fmt_extensible(24000)), chunk(b"data", pcm)))
    assert audio.pcm == pcm


def test_parse_mixes_stereo_down_to_mono():
    stereo = tone(24000, channels=2)
    audio = parse_wav(simple_wav(stereo, channels=2))
    assert len(audio.pcm) == len(stereo) // 2
    left = np.frombuffer(stereo, dtype="<i2")[::2]
    assert np.array_equal(np.frombuffer(audio.pcm, dtype="<i2"), left)


def test_parse_rejects_8_bit():
    with pytest.raises(WavError, match="8 bits"):
        parse_wav(simple_wav(b"\x80" * 100, width=1))


@pytest.mark.parametrize(
    "data",
    [b"", b'{"detail": "error"}', b"RIFF\x00\x00\x00\x00WAVE", b"ID3\x04" + b"\x00" * 64],
)
def test_parse_rejects_non_wav(data):
    with pytest.raises(WavError):
        parse_wav(data)


# -- service in a pipeline, against a fake Chatterbox --


def service_for(handler, timeout: float = 1.0) -> tuple[ChatterboxTTSService, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = ChatterboxTTSService(
        base_url="http://tts.test/", voice="example.wav", client=client, timeout=timeout
    )
    return tts, client


async def speak(handler, tts_timeout: float = 0.5):
    tts, client = service_for(handler, tts_timeout)
    async with client:
        down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("Hello there.")])
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    # Pipecat adds its own "context completed with no audio" error after ours.
    errors = [f for f in up if isinstance(f, ErrorFrame) and "no audio" not in f.error]
    return audio, errors


async def test_speaks_one_request_per_sentence_and_yields_pcm():
    requests = []
    pcm = tone(24000, seconds=0.5)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=simple_wav(pcm), headers={"content-type": "audio/wav"})

    audio, errors = await speak(handler)
    assert not errors
    assert len(requests) == 1
    assert str(requests[0].url) == "http://tts.test/v1/audio/speech"
    body = json.loads(requests[0].content)
    assert body["input"] == "Hello there."
    assert body["voice"] == "example.wav"
    assert body["response_format"] == "wav"
    assert {f.sample_rate for f in audio} == {24000}
    assert all(f.num_channels == 1 for f in audio)
    assert b"".join(f.audio for f in audio) == pcm


async def test_other_sample_rates_are_resampled():
    pcm = tone(16000, seconds=0.5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=simple_wav(pcm, rate=16000))

    audio, errors = await speak(handler)
    assert not errors
    assert {f.sample_rate for f in audio} == {24000}
    total = sum(len(f.audio) for f in audio) // 2
    assert abs(total - 12000) < 600  # 0.5 s at 24 kHz, give or take the resampler


async def test_http_error_becomes_error_frame():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "voice not found"})

    audio, errors = await speak(handler)
    assert not audio
    assert len(errors) == 1
    assert errors[0].error.startswith("The voice “example.wav” is not available")


async def test_other_http_error_keeps_the_server_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "out of memory"})

    audio, errors = await speak(handler)
    assert not audio
    assert len(errors) == 1
    assert "HTTP 500" in errors[0].error and "out of memory" in errors[0].error


async def test_connection_error_becomes_error_frame():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    audio, errors = await speak(handler)
    assert not audio
    assert len(errors) == 1
    assert errors[0].error == "Cannot reach the voice server (Chatterbox)."


async def test_timeout_becomes_error_frame():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, content=simple_wav(tone(24000)))

    audio, errors = await speak(handler, tts_timeout=0.2)
    assert not audio
    assert len(errors) == 1
    assert "did not answer in time" in errors[0].error


async def test_bad_wav_becomes_error_frame():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\xff\xfb\x90\x00 an mp3, not a wav")

    audio, errors = await speak(handler)
    assert not audio
    assert len(errors) == 1
    assert "could not be played" in errors[0].error
