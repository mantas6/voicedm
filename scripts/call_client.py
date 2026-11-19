"""Headless test call: speak WAV files to a running voicedm server and record the reply.

Usage (server already running):

    uv run python scripts/call_client.py --url http://127.0.0.1:8780 --session 3 \\
        --say question.wav --barge-in interrupt.wav --record reply.wav --duration 60

The call is on session ``--session``, or on a new session if none is given. ``--say``
files are spoken one after another from the start of the call (after ``--delay``
seconds); with ``--wait-reply`` each one after the first waits until the bot has
answered the previous one (its audio started and then stayed quiet for ``--quiet``
seconds). ``--patch`` is a JSON body sent as a PATCH to the session after the first
reply, before the next ``--say``, to check mid-call settings changes. ``--barge-in``
is spoken ``--barge-in-after`` seconds after the bot's audio first starts, to check
that talking over the bot stops it. The client logs RTVI events from the data channel
and when the bot's audio starts and stops, with times relative to the start of the
call, and prints the session's stored messages at the end. Input WAVs must be 16-bit
PCM.
"""

from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import time
import uuid
import wave
from pathlib import Path

import httpx
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

RATE = 48000
FRAME_SAMPLES = 960  # 20 ms
SILENCE_RMS = 300  # int16 RMS below which received audio counts as silence
SILENCE_GAP = 0.6  # seconds of silence that end a stretch of bot audio

T0 = time.monotonic()


def log(message: str) -> None:
    print(f"[{time.monotonic() - T0:7.2f}s] {message}", flush=True)


def load_wav(path: Path) -> np.ndarray:
    """Read a 16-bit WAV and return mono int16 samples at ``RATE``."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit PCM")
        channels, rate = wav.getnchannels(), wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != RATE:
        positions = np.arange(0, len(samples), rate / RATE)
        samples = np.interp(positions, np.arange(len(samples)), samples)
    return samples.astype(np.int16)


class SpeechTrack(MediaStreamTrack):
    """Microphone stand-in: silence, plus whatever clips are queued with ``say``."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: list[np.ndarray] = []
        self._current = np.zeros(0, dtype=np.int16)
        self._pts = 0
        self._start: float | None = None

    def say(self, samples: np.ndarray, label: str) -> None:
        log(f"speaking {label} ({len(samples) / RATE:.1f}s)")
        self._queue.append(samples)

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.monotonic()
        # Pace frames in real time, like a microphone.
        due = self._start + self._pts / RATE
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        if not len(self._current) and self._queue:
            self._current = self._queue.pop(0)
        chunk = self._current[:FRAME_SAMPLES]
        self._current = self._current[FRAME_SAMPLES:]
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        frame = AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, RATE)
        self._pts += FRAME_SAMPLES
        return frame


class Recorder:
    """Writes received audio to a WAV and logs when bot speech starts and stops."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.chunks: list[np.ndarray] = []
        self.rate = RATE
        self.speaking = False
        self.last_loud = 0.0
        self.first_audio = asyncio.Event()
        self.stretches: list[tuple[float, float]] = []

    async def consume(self, track: MediaStreamTrack) -> None:
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            samples = frame.to_ndarray().reshape(-1, len(frame.layout.channels))[:, 0]
            self.rate = frame.sample_rate
            self.chunks.append(samples.astype(np.int16))
            self._detect(samples)

    def _detect(self, samples: np.ndarray) -> None:
        now = time.monotonic() - T0
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        if rms >= SILENCE_RMS:
            if not self.speaking:
                self.speaking = True
                self.stretches.append((now, now))
                log("bot audio started")
                self.first_audio.set()
            self.last_loud = now
        elif self.speaking and now - self.last_loud > SILENCE_GAP:
            self.speaking = False
            self.stretches[-1] = (self.stretches[-1][0], self.last_loud)
            log(f"bot audio stopped (stretch {self.last_loud - self.stretches[-1][0]:.1f}s)")

    def save(self) -> None:
        if self.speaking:
            self.stretches[-1] = (self.stretches[-1][0], self.last_loud)
        if not self.path or not self.chunks:
            return
        with wave.open(str(self.path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.rate)
            wav.writeframes(np.concatenate(self.chunks).tobytes())
        log(f"recorded {sum(map(len, self.chunks)) / self.rate:.1f}s to {self.path}")


def on_message(raw: str) -> None:
    try:
        msg = json.loads(raw)
    except ValueError:
        return
    kind, data = msg.get("type"), msg.get("data") or {}
    if kind in ("metrics", "bot-llm-text", "bot-output", "user-speaking", "llm-raw-text"):
        return  # too chatty to log
    if kind in ("user-transcription", "bot-transcription", "bot-tts-text"):
        if kind == "user-transcription" and not data.get("final", True):
            return
        log(f"{kind}: {data.get('text', '').strip()!r}")
    elif kind == "server-message":
        log(f"server-message: {data}")
    else:
        log(kind)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8780")
    parser.add_argument("--session", type=int, help="session id (default: a new session)")
    parser.add_argument("--say", type=Path, action="append", default=[])
    parser.add_argument("--delay", type=float, default=2.0, help="seconds before --say")
    parser.add_argument("--wait-reply", action="store_true", help="say the next after a reply")
    parser.add_argument("--quiet", type=float, default=3.0, help="silence that ends a reply")
    parser.add_argument("--patch", type=json.loads, help="JSON PATCH body after the 1st reply")
    parser.add_argument("--barge-in", type=Path)
    parser.add_argument("--barge-in-after", type=float, default=2.0)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    says = [(path, load_wav(path)) for path in args.say]
    barge_in = load_wav(args.barge_in) if args.barge_in else None

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))  # host candidates only
    mic = SpeechTrack()
    recorder = Recorder(args.record)
    pc.addTransceiver(mic, direction="sendrecv")
    channel = pc.createDataChannel("messages", ordered=True)
    tasks: list[asyncio.Task] = []

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            tasks.append(asyncio.create_task(recorder.consume(track)))

    @channel.on("open")
    def on_open() -> None:
        ready = {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": str(uuid.uuid4()),
            "data": {"version": "2.3.0", "about": {"library": "voicedm-call-client"}},
        }
        channel.send(json.dumps(ready))

    channel.on("message", on_message)

    http = httpx.AsyncClient(base_url=args.url, timeout=30)
    session_id = args.session
    if session_id is None:
        response = await http.post("/api/sessions")
        response.raise_for_status()
        session_id = response.json()["id"]
        log(f"created session {session_id}")
    session = (await http.get(f"/api/sessions/{session_id}")).raise_for_status().json()
    log(
        f"session {session_id}: model {session['model']}, voice {session['voice']}, "
        f"{len(session['messages'])} earlier messages"
    )

    await pc.setLocalDescription(await pc.createOffer())
    response = await http.post(
        "/api/offer",
        json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id,
        },
    )
    response.raise_for_status()
    answer = response.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    log(f"connected, pc_id {answer.get('pc_id')}")

    async def reply_done(stretches_before: int) -> None:
        """Wait until bot audio started after ``stretches_before`` and then went quiet."""
        while True:
            await asyncio.sleep(0.2)
            now = time.monotonic() - T0
            if (
                len(recorder.stretches) > stretches_before
                and not recorder.speaking
                and now - recorder.last_loud > args.quiet
            ):
                return

    async def speak() -> None:
        await asyncio.sleep(args.delay)
        stretches = 0
        for number, (path, samples) in enumerate(says):
            if number and args.wait_reply:
                await reply_done(stretches)
            if number == 1 and args.patch is not None:
                response = await http.patch(f"/api/sessions/{session_id}", json=args.patch)
                log(f"PATCH {args.patch} -> HTTP {response.status_code}")
            stretches = len(recorder.stretches)
            mic.say(samples, path.name)
            await asyncio.sleep(len(samples) / RATE + 0.1)

    async def interrupt() -> None:
        await recorder.first_audio.wait()
        await asyncio.sleep(args.barge_in_after)
        mic.say(barge_in, f"{args.barge_in.name} (barge-in)")

    tasks.append(asyncio.create_task(speak()))
    if barge_in is not None:
        tasks.append(asyncio.create_task(interrupt()))
    await asyncio.sleep(args.duration)

    for task in tasks:
        task.cancel()
    await pc.close()
    recorder.save()
    for start, end in recorder.stretches:
        log(f"bot audio stretch {start:.2f}s - {end:.2f}s")

    await asyncio.sleep(1.0)  # let the server save the last turn
    session = (await http.get(f"/api/sessions/{session_id}")).raise_for_status().json()
    await http.aclose()
    log(f"session {session_id} ({session['title']!r}) now has:")
    for m in session["messages"]:
        flags = " [interrupted]" if m["interrupted"] else ""
        model = f" ({m['model']})" if m["model"] else ""
        print(f"  #{m['id']} {m['role']}{model}{flags}: {m['content']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
