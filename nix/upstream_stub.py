"""Fake Ollama and Chatterbox for the NixOS VM test: just enough of each API.

Usage: upstream_stub.py ollama PORT | upstream_stub.py chatterbox PORT
"""

import io
import json
import math
import struct
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "stub-model:latest"
VOICE = "stub-voice.wav"
REPLY = "Paris is the capital of France."


def tone_wav(seconds: float = 1.0, rate: int = 24000) -> bytes:
    """A 440 Hz tone, loud enough to count as speech for the call client."""
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(int(seconds * rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return buffer.getvalue()


def chunk(delta: dict, finish: str | None = None) -> bytes:
    body = {
        "id": "stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    service = ""

    def send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: object) -> None:
        self.send(200, json.dumps(value).encode())

    def do_GET(self) -> None:
        if self.service == "ollama" and self.path == "/api/tags":
            self.send_json({"models": [{"name": MODEL}]})
        elif self.service == "chatterbox" and self.path == "/v1/audio/voices":
            self.send_json({"voices": [VOICE]})
        else:
            self.send(404, b'{"error": "not found"}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.service == "ollama" and self.path == "/api/generate":
            self.send_json({"model": MODEL, "response": "", "done": True})
        elif self.service == "ollama" and self.path == "/api/chat":
            self.send_json({"model": MODEL, "message": {"role": "assistant", "content": REPLY}})
        elif self.service == "ollama" and self.path == "/v1/chat/completions":
            # Streamed reply; HTTP/1.0 without a length, so the body ends at close.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(chunk({"role": "assistant", "content": REPLY}))
            self.wfile.write(chunk({}, "stop"))
            self.wfile.write(b"data: [DONE]\n\n")
        elif self.service == "chatterbox" and self.path == "/v1/audio/speech":
            if request.get("voice") != VOICE:
                self.send(404, b'{"detail": "voice not found"}')
            else:
                self.send(200, tone_wav(), "audio/wav")
        else:
            self.send(404, b'{"error": "not found"}')


def main() -> None:
    Handler.service, port = sys.argv[1], int(sys.argv[2])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
