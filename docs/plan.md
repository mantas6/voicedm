# Implementation plan

voicedm is a self-hosted, hands-free voice conversation with a local LLM, used from a
browser. You open the page, press **Call**, and talk; the bot answers with speech and can
be interrupted mid-sentence.

This document describes the first version (v1): a browser call over WebRTC. Phone-style
calling (SIP, Matrix, messenger bridges) is out of scope for v1, but the design keeps the
voice pipeline independent of the transport so it can be added later.

## Goals

- Low latency: speech starts streaming sentence by sentence while the LLM is still
  generating, instead of waiting for the whole reply.
- Barge-in: talking over the bot stops its speech and starts a new turn.
- Fully self-hosted and open source: no cloud STT, LLM or TTS.
- Experimentation friendly: pick the LLM model, TTS voice and system prompt from the UI.
- Durable sessions with summary memory, so long conversations stay within the context
  window.
- Deployed as a NixOS module.

## Non-goals for v1

- SIP / phone calls, Matrix, Telegram or any other call transport.
- Authentication and multiple users. v1 is a single-user app on a trusted network.
- Languages other than English.
- Importing history from earlier prototypes.
- Context-usage meter, listening sound cues, loudness normalisation.

## Architecture

```
Browser (mic/speaker)
   │  WebRTC audio (UDP, direct on the local network)
   │  HTTP: page, REST API, WebRTC offer/answer
   ▼
voicedm server (Python, FastAPI + Pipecat)
   ├─ Silero VAD ─ turn detection and interruptions
   ├─ faster-whisper (CPU) ─ speech-to-text
   ├─ Ollama (HTTP) ─ LLM, streamed
   ├─ Chatterbox TTS server (HTTP) ─ speech, per sentence
   └─ SQLite ─ sessions, messages, summaries
```

External services, all reached over HTTP and configured by URL:

| Service | API used | Notes |
|---|---|---|
| Ollama | `/v1/chat/completions` (streaming), `/api/tags`, `/api/generate` (warm-up) | May run on CPU only; prompt prefix caching matters a lot (see Memory). |
| Chatterbox TTS Server | `/v1/audio/speech`, `/v1/audio/voices` | OpenAI-style endpoint, but returns `wav`/`mp3`/`opus`, not raw PCM. May stop when idle, so the first request can take tens of seconds. |

Speech-to-text runs inside the voicedm process.

### Pipeline (Pipecat)

```
transport.input()
  → STT (faster-whisper)
  → user context aggregator
  → LLM (Ollama)
  → TTS (Chatterbox)
  → transport.output()
  → assistant context aggregator
```

- **Transport:** Pipecat `SmallWebRTCTransport`. The server exposes `POST /api/offer`
  for the SDP offer/answer. On a local network, host ICE candidates are enough; a STUN
  server URL is optional config.
- **VAD / turns:** `SileroVADAnalyzer` with interruptions enabled. A smart-turn model is
  a later improvement if the silence-based end-of-turn feels too eager or too slow.
- **STT:** Pipecat `WhisperSTTService` (faster-whisper), English-only model, CPU, `int8`.
  The default model is `base.en`; `small.en` is configurable if the CPU keeps up. The
  model is loaded once at startup.
- **LLM:** Pipecat `OLLamaLLMService` against Ollama's OpenAI-compatible endpoint.
  Ollama ignores per-request `num_ctx` on that endpoint, so the context length is the
  Ollama server's setting. If that becomes a problem, replace it with a small service on
  the native `/api/chat` endpoint.
- **TTS:** a custom `ChatterboxTTSService` (about 50 lines). It requests `wav` for each
  sentence, strips the header and yields 24 kHz mono PCM frames. Pipecat's sentence
  aggregation feeds it text, so the TTS server does not need to stream.
  `OpenAITTSService` cannot be used as-is because it asks for raw `pcm`.
- **Transcripts:** Pipecat's RTVI processor sends user and bot transcription events over
  the WebRTC data channel, and the page shows them live.
- **Warm-up:** when a call connects, start a background no-op request to Ollama (load
  the selected model) and a one-word synthesis to Chatterbox, so the first real turn
  does not pay the cold-start cost. The UI shows "warming up" until both answer.
- **Metrics:** enable Pipecat metrics and log time-to-first-byte per service (STT, LLM,
  TTS) for each turn, to find where latency goes.

### Settings and mid-call changes

Each session stores its own model, voice and system prompt. The UI can change them at
any time:

- Between calls, the new values apply to the next call.
- During a call, the server pushes Pipecat update-settings frames (LLM model, TTS voice)
  into the running pipeline and replaces the system message in the context. These take
  effect from the next turn.

The model list comes from Ollama `/api/tags` and the voice list from Chatterbox
`/v1/audio/voices`. Both are proxied by the server so the browser only talks to voicedm.

## Storage

SQLite in the service's state directory, WAL mode.

```sql
sessions (
  id            INTEGER PRIMARY KEY,
  title         TEXT,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  model         TEXT NOT NULL,
  voice         TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  summary       TEXT NOT NULL DEFAULT '',
  summary_upto  INTEGER          -- last message id folded into the summary
)

messages (
  id          INTEGER PRIMARY KEY,
  session_id  INTEGER NOT NULL REFERENCES sessions(id),
  role        TEXT NOT NULL,     -- 'user' | 'assistant'
  content     TEXT NOT NULL,
  model       TEXT,              -- model that produced an assistant message
  interrupted INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL
)
```

- Messages are saved when the context aggregators finish a turn. For an interrupted
  reply, only the text that was actually spoken is stored, marked `interrupted`.
- Schema changes go through numbered migrations, applied at startup.
- "New session" never deletes anything. Deleting sessions is a later feature.

## Memory

The LLM may run on CPU, where reading uncached prompt tokens is slow (on the order of
tens of tokens per second). Ollama reuses its cache only for the unchanged start of the
prompt, so the prompt layout is designed to keep that start stable:

```
[system prompt] [summary of older messages] [recent messages ...]
```

- Recent messages are appended turn by turn, so every turn only processes the new
  tokens.
- When the recent messages exceed a size budget, the older half is folded into the
  summary in one step: a separate summarisation request, then the context is rebuilt.
  That costs one full prompt read and then stays cached again. It is not done on every
  turn.
- Summarisation runs between turns, never while the user is waiting for a reply. If a
  turn starts while it runs, the turn uses the old context.
- Changing the system prompt or model also invalidates the cache. That is expected and
  only costs the next turn.

The budget and the summary prompt are config values, so they can be tuned per model.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Page (static files) |
| `GET` | `/api/models` | Installed Ollama models |
| `GET` | `/api/voices` | Chatterbox voices |
| `GET` | `/api/sessions` | List sessions |
| `POST` | `/api/sessions` | New session (copies the current settings) |
| `GET` | `/api/sessions/{id}` | Session with settings and messages |
| `PATCH` | `/api/sessions/{id}` | Change model, voice, system prompt or title |
| `POST` | `/api/offer` | WebRTC offer/answer for a call on a session |
| `GET` | `/api/health` | Liveness, plus reachability of Ollama and Chatterbox |

Only one call runs at a time. A second offer while a call is active gets HTTP 409.

## Web UI

Plain HTML, CSS and JavaScript served by the app, with no build step. It talks to
`RTCPeerConnection` and the RTVI data-channel messages directly. If that gets awkward, it
can switch to Pipecat's JS client SDK vendored as ES modules.

- Session list, new session.
- Model, voice and system prompt controls for the current session.
- Call / Hang up button with state: connecting, warming up, listening, thinking,
  speaking.
- Live transcript, plus history loaded from the API.
- Mute toggle.
- Works on a phone-width screen.

The browser's echo cancellation and noise suppression are requested on the mic track,
which is what makes barge-in work on a phone speaker.

### HTTPS and the microphone

The app serves plain HTTP. Browsers only allow microphone access on secure origins, so
for use from another device either:

- put an HTTPS reverse proxy or tunnel in front. The page and the WebRTC offer/answer go
  through it; the audio still flows directly between the browser and the server on the
  local network.
- or mark the LAN origin as secure in the browser for testing (in Chrome:
  `chrome://flags/#unsafely-treat-insecure-origin-as-secure`).

v1 has no authentication, so anything that exposes it beyond a trusted network must add
access control in front of it.

## Configuration

Environment variables, set by the NixOS module:

| Variable | Default | Meaning |
|---|---|---|
| `VOICEDM_HOST` | `0.0.0.0` | Listen address |
| `VOICEDM_PORT` | `8780` | Listen port |
| `VOICEDM_DATA_DIR` | `./data` | SQLite database and model cache |
| `VOICEDM_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `VOICEDM_CHATTERBOX_URL` | `http://localhost:8004` | Chatterbox TTS server base URL |
| `VOICEDM_WHISPER_MODEL` | `base.en` | faster-whisper model |
| `VOICEDM_DEFAULT_MODEL` | first installed model | Model for new sessions |
| `VOICEDM_DEFAULT_VOICE` | first listed voice | Voice for new sessions |
| `VOICEDM_STUN_URL` | unset | Optional STUN server |
| `VOICEDM_TTS_TIMEOUT` | `120` | Seconds; allows for a TTS cold start |

## Repository layout

```
flake.nix                  package, NixOS module, dev shell
nix/package.nix
nix/module.nix
pyproject.toml, uv.lock
src/voicedm/
  __main__.py              entry point (uvicorn)
  config.py
  app.py                   FastAPI routes, static files
  bot.py                   Pipecat pipeline per call
  tts_chatterbox.py
  storage.py               SQLite access and migrations
  memory.py                context building and summarisation
  upstream.py              Ollama and Chatterbox listing, health, warm-up
  static/                  index.html, app.js, style.css
tests/
docs/
```

## Packaging and deployment

- **Development:** `uv` for Python dependencies, a flake dev shell with Python, `uv`,
  `ffmpeg` and `ruff`.
- **Package:** built from `uv.lock` with uv2nix (pyproject-nix), so the NixOS service
  uses pinned dependencies. Binary wheels (onnxruntime for Silero, ctranslate2 for
  faster-whisper, aiortc) may need fix-ups such as `autoPatchelfHook`. If this takes too
  long, the fallback is a service that runs `uv sync` into its state directory at start;
  that is less pure but unblocks deployment.
- **NixOS module** `services.voicedm`:
  - `enable`, `package`, `host`, `port`, `openFirewall`
  - `ollamaUrl`, `chatterboxUrl`, `whisperModel`, `stunUrl`
  - `environment` for extra variables
  - systemd service with `DynamicUser`, `StateDirectory=voicedm` (database plus the
    Hugging Face cache for the Whisper model), and the usual hardening options. The
    firewall needs the HTTP port and the UDP range aiortc uses for media, unless that
    range is pinned in config.
- **Wiring:** the NixOS configuration that hosts the app adds this flake as an input,
  imports `nixosModules.default`, enables the service and sets the two upstream URLs.
  The repository is private, so the machine that builds that configuration needs read
  access to it.

The app host can be CPU-only. Speech-to-text is the only model that runs in the voicedm
process. If faster-whisper is too slow on the target CPU, use a smaller model or move the
service to a faster host; only the URLs change.

## Milestones

1. **Scaffold.** `pyproject.toml`, `uv.lock`, flake dev shell, `ruff`, `pytest`, an empty
   FastAPI app with `/api/health`.
2. **Voice loop.** Pipeline with SmallWebRTC, Silero, faster-whisper, Ollama (model from
   config), the Chatterbox TTS service and warm-up. A bare page with Call / Hang up.
   *Done when:* a spoken question gets a spoken answer that starts before the reply is
   fully generated, and talking over the bot stops it.
3. **Sessions and settings.** SQLite storage, sessions API, model/voice/system-prompt
   controls, live transcript and history in the UI, mid-call setting changes.
4. **Memory.** Summary folding with the cache-friendly prompt layout, configurable
   budget.
5. **Hardening.** Error states in the UI (Ollama or TTS down, cold start), the
   one-call-at-a-time limit, per-turn latency logging, reconnect after a dropped call.
6. **Nix.** Package, NixOS module, a NixOS VM test that starts the service and checks
   `/api/health`.
7. **Deploy.** Wire it into the hosting NixOS configuration and test on a real phone.

## Testing

- **Unit:** storage and migrations, memory folding (budget edges, interrupted messages),
  the Chatterbox service (WAV parsing, errors, timeouts) against a fake HTTP server, the
  API routes with fake upstreams.
- **Pipeline:** a Pipecat pipeline test with fake STT/LLM/TTS services to check turn
  persistence and interruption handling.
- **End to end (manual, recorded in the PR):** a call from a desktop browser and from a
  phone; time from end of speech to first audio, from the metrics log; barge-in;
  changing voice and model mid-call; a cold start of both upstreams.

## Risks

| Risk | Mitigation |
|---|---|
| CPU-only LLM is too slow for natural conversation | Model picker to try smaller or more quantised models; keep the prompt prefix cached; speak sentence by sentence. |
| faster-whisper too slow on a small CPU | Smaller model, or run the service on a faster host. |
| TTS server cold start (container stopped when idle) | Warm-up on call connect, long TTS timeout, "warming up" state in the UI. |
| Packaging binary Python wheels with Nix | uv2nix fix-ups; fallback to `uv sync` at service start. |
| Browser microphone needs HTTPS | HTTPS proxy or tunnel in front, or a browser flag for testing. |
| Echo from the phone speaker triggers barge-in | Browser echo cancellation; tune the VAD threshold and minimum speech length. |
