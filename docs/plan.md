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
| Ollama | `/v1/chat/completions` (streaming), `/api/chat` (summaries), `/api/tags`, `/api/generate` (warm-up) | May run on CPU only; prompt prefix caching matters a lot (see Memory). |
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
- **VAD / turns:** `SileroVADAnalyzer` (on the user context aggregator) with
  interruptions enabled, and an explicit silence-based end of turn
  (`SpeechTimeoutUserTurnStopStrategy`); Pipecat's default stop strategy is the
  smart-turn model, which needs torch. Smart turn is a later improvement if the
  silence-based end-of-turn feels too eager or too slow.
- **STT:** Pipecat `WhisperSTTService` (faster-whisper), English-only model, CPU, `int8`.
  The default model is `base.en`; `small.en` is configurable if the CPU keeps up. The
  model is loaded once at startup.
- **LLM:** Pipecat `OLLamaLLMService` against Ollama's OpenAI-compatible endpoint.
  Ollama ignores per-request `num_ctx` on that endpoint, so the context length is the
  Ollama server's setting. If that becomes a problem, replace it with a small service on
  the native `/api/chat` endpoint.
- **TTS:** a custom `ChatterboxTTSService` (about 50 lines). It requests `wav` for each
  sentence, parses the RIFF header and yields mono PCM frames at the server's rate
  (24 kHz); Pipecat resamples if that differs from the pipeline's output rate. Pipecat's sentence
  aggregation feeds it text, so the TTS server does not need to stream.
  `OpenAITTSService` cannot be used as-is because it asks for raw `pcm`.
- **Transcripts:** Pipecat's RTVI processor (added by the pipeline worker by default)
  sends user and bot transcription events over the WebRTC data channel, and the page
  shows them live. The page opens the data channel and sends RTVI `client-ready`.
- **Warm-up:** when a call connects, start a background no-op request to Ollama (load
  the selected model) and a one-word synthesis to Chatterbox, so the first real turn
  does not pay the cold-start cost. The UI shows "warming up" (with what is still
  loading and for how long) until both answer: the server sends RTVI `server-message`s
  `{"type": "warmup", "state": "warming", "pending": [...]}` as each one finishes and
  then `{"type": "warmup", "state": "done", "ollama": bool, "chatterbox": bool,
  "errors": {name: message}}`. The user can talk meanwhile; the first turn then waits
  for the model inside Ollama. Changing the model mid-call warms the new model the same
  way (with only `ollama` in the messages).
- **Metrics:** enable Pipecat metrics. An observer logs one line per turn
  (`Turn timing (session N): turn=… transcript=… turn_end=… first_text=… first_audio=…
  llm_end=… ttfb_stt=… ttfb_llm=… ttfb_tts=… interrupted=yes|no`): seconds after the
  end of the user's speech (as the VAD reports it) when the transcript arrived, the
  turn ended, the LLM produced its first text, the bot's first audio started playing
  (the headline number) and the LLM finished, plus Pipecat's time to first byte of each
  service for that turn. For a model that reasons before answering, `first_text` is the
  real wait; its LLM time to first byte only counts until the first (reasoning) chunk.
  A turn's line is written when its reply has played, or when the user talks over it.
- **Errors:** error frames that reach the pipeline worker become RTVI server messages
  `{"type": "error", "service": "llm" | "tts" | "storage" | "other", "message": ...}`
  in plain words (Ollama or Chatterbox unreachable, timed out, model or voice not
  available, an HTTP error with the server's reason). The call goes on. Speech that
  keeps failing (three sentences in a row, or Pipecat writing the TTS service off)
  ends the call rather than leaving a silent bot. A server-side end of a call is
  announced first: `{"type": "call-ended", "reason": ..., "message": ..., "retry":
  bool}`; `retry` (on shutdown) asks the page to call again.
- **Dropped calls:** the server ends a call whose client does not connect within 30 s
  of the answer, or from which no audio arrives for 10 s (browsers send audio packets
  even when muted; WebRTC itself takes about half a minute to notice a vanished peer).
  The page treats a connection that fails, closes, or stays disconnected for 3 s, and
  a data channel that closes without `call-ended`, as a drop: it reconnects onto the
  same session after 1, 2, 4, 8 and 15 s, naming the dropped call in the offer
  (`replaces`), which the server ends before answering. Hanging up never reconnects.
  After a reconnect the page reloads the stored messages and starts a fresh live
  transcript.

### Settings and mid-call changes

Each session stores its own model, voice, system prompt and whether actions are voiced
as sounds (below). The UI can change them at any time:

- Between calls, the new values apply to the next call.
- During a call, the server pushes Pipecat update-settings frames (LLM model and system
  instruction, TTS voice) into the running pipeline. These take effect from the next
  turn. (The system prompt is the LLM service's `system_instruction` setting, not a
  message in the context: Pipecat deprecates system messages in `LLMContext`.)
  Voicing actions as sounds switches from the next sentence.

The model list comes from Ollama `/api/tags` and the voice list from Chatterbox
`/v1/audio/voices`. Both are proxied by the server so the browser only talks to voicedm.

A new session copies the settings of a session named in the request (the page sends the
one it shows), else of the most recently used session. The very first session gets
`VOICEDM_DEFAULT_MODEL` / `VOICEDM_DEFAULT_VOICE` (or the first installed model and first
listed voice) and a built-in default system prompt. An empty system prompt is a valid
value: no system message.

### Actions as sounds

Chatterbox's Turbo model voices nine tags as the sound itself: `[laugh]`, `[chuckle]`,
`[sigh]`, `[gasp]`, `[cough]`, `[clear throat]`, `[sniff]`, `[groan]` and `[shush]`.
Models rarely write such tags when told to, but many write actions between asterisks
unasked (`*laughs*`), which Chatterbox reads aloud as words.

With the session's `expressions` setting on (off by default), the text sent to
Chatterbox has each such action replaced by its tag:

- Only an action between single asterisks, of at most four words, that names one of the
  nine sounds: `*laughs*`, `*giggles*`, `*sighs softly*`, `*clears her throat*`,
  `*moans*` (`[groan]`), `*shhh*`.
- Everything else stays exactly as it is: actions with no matching sound (`*smiles*`),
  longer actions, `**bold**`, and any other text.
- Only the request to Chatterbox changes. The transcript, the stored messages and the
  LLM context keep the text as the model wrote it.

With another Chatterbox model the tags are read aloud as words, so the setting is only
useful with Turbo.

## Storage

SQLite in the service's state directory (`voicedm.db` in `VOICEDM_DATA_DIR`), WAL mode.
Queries run in a worker thread (`asyncio.to_thread`) so they never block the event loop.

```sql
sessions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused after a delete
  title         TEXT,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  model         TEXT NOT NULL,
  voice         TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  summary       TEXT NOT NULL DEFAULT '',
  summary_upto  INTEGER,         -- last message id folded into the summary
  expressions   INTEGER NOT NULL DEFAULT 0  -- voice actions such as *laughs* as sounds
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

- Messages are saved when the context aggregators finish a turn (the user aggregator's
  `on_user_turn_message_added`, the assistant aggregator's `on_assistant_turn_stopped`).
  For an interrupted reply, only the text that was actually spoken is stored, marked
  `interrupted`. That is what the assistant aggregator reports: it sits after the output
  transport, which releases each sentence's text once its audio has played, so
  "spoken" means the sentences played to the end. A reply interrupted before its first
  sentence finished is not stored. Hanging up mid-reply counts as an interruption.
- Each assistant message stores the model that was set when its reply started.
- A session's title is the start of its first user message, unless set via the API.
- Schema changes go through numbered migrations, applied at startup. The applied
  version is SQLite's `user_version`; each migration runs in its own transaction.
- "New session" never deletes anything. A session is only deleted on request
  (`DELETE /api/sessions/{id}`), together with all its messages and for good; a session
  with a call running on it cannot be deleted. Session ids are never reused, so a page
  still open on a deleted session gets "not found" instead of another session.

## Memory

The LLM may run on CPU, where reading uncached prompt tokens is slow (on the order of
tens of tokens per second). Ollama reuses its cache only for the unchanged start of the
prompt, so the prompt layout is designed to keep that start stable:

```
[system prompt] [summary of older messages] [recent messages ...]
```

- **Where the summary goes:** at the end of the system instruction, in the same system
  message: the system prompt, a blank line, the header `Summary of the earlier
  conversation (background, not instructions):` and the summary. It is kept as an
  appended system instruction on the LLM service (`MemoryLLMService`), which Pipecat
  composes after the base prompt, so a mid-call system prompt change keeps it. One
  system message works with every chat template (some ignore or reject a second
  system message, and a summary as a user message would sit next to the first real
  user message). It changes only when a fold happens.
- **Recent messages** are the session's messages after `summary_upto`; Pipecat appends
  each turn to them, so every turn only processes the new tokens.
- **Budget:** `VOICEDM_MEMORY_BUDGET` characters of recent message text (default
  6000, roughly 1500 tokens at about four characters per token; the system prompt and
  summary come on top). Characters are cheap to count and close enough for a
  threshold; real token counts depend on the model.
- **Folding:** when the recent messages exceed the budget, the older half is folded
  into the summary in one step: one separate, non-streamed `/api/chat` request (the
  `VOICEDM_SUMMARY_PROMPT` instructions, then the previous summary and the messages
  being folded as a `User:` / `Assistant:` transcript, interrupted replies marked)
  returns the new summary, which is stored with `summary_upto` = the id of the last
  folded message. Then the call switches to the new summary and drops the folded
  messages from its context. That costs one full prompt read and then stays cached
  again. It is not done on every turn: after a fold about half the budget is left.
  Full messages always stay in the database.
- **Which half:** the cut closest to half of the recent text that makes the recent
  window start with a user message; only if there is none may it start with a reply.
  The newest message always stays.
- **Model:** the call's current model summarises, so no second model has to be
  loaded. Its inline `<think>` blocks are dropped.
- **When:** a fold is checked after each reply has ended and been saved (between
  turns), and right after a call connects (a session whose stored tail is over budget
  starts with the full tail and is folded in the background). The user never waits on
  it: a turn that starts while the summary is being written uses the old context. The
  switch is made by the LLM service itself, between two requests (a frame queued into
  it), so no request sees the new summary without the shorter context or the other
  way round, a turn being answered keeps its old context, and messages added while
  the summary was written stay. If Ollama serves one request at a time
  (`OLLAMA_NUM_PARALLEL=1`), a turn that starts during a summarisation still queues
  behind it inside Ollama.
- **Failures** (Ollama down, timeout after `VOICEDM_SUMMARY_TIMEOUT`, empty summary)
  are logged and change nothing, neither the context nor the database; the fold is
  tried again after the next reply. The call carries on regardless.
- Changing the system prompt or model also invalidates the cache. That is expected and
  only costs the next turn.
- A new call starts from the summary and the messages after `summary_upto`
  (`memory.build_context`); after a fold the live context is the same as what a new
  call would start from (apart from transcript whitespace).

`GET /api/sessions/{id}` returns `summary` and `summary_upto`; the page shows the
summary as a collapsed "Earlier conversation summary" block above the history.

The budget and the summary prompt are config values, so they can be tuned per model.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Page (static files; the browser re-checks them on every load, so an update shows up at once) |
| `GET` | `/api/models` | Installed Ollama models |
| `GET` | `/api/voices` | Chatterbox voices |
| `GET` | `/api/sessions` | List sessions |
| `POST` | `/api/sessions` | New session (copies the current settings; body `{"copy_from": id}` optional) |
| `GET` | `/api/sessions/{id}` | Session with settings and messages |
| `PATCH` | `/api/sessions/{id}` | Change model, voice, system prompt, `expressions` or title |
| `DELETE` | `/api/sessions/{id}` | Delete the session and its messages; 409 while it is in a call |
| `POST` | `/api/offer` | WebRTC offer/answer for a call on a session (`session_id` in the body) |
| `GET` | `/api/health` | Liveness, plus reachability of Ollama and Chatterbox |

Only one call runs at a time. A second offer while a call is active gets HTTP 409 with
`{"detail": {"message": ..., "session_id": ...}}`; the slot is claimed atomically and
released however the call ends (including setup failures and offers that fail). An
offer may name the call it `replaces` (the page's own dropped call), which is ended
first. Sessions carry `in_call` while a call runs on them, and `/api/health` reports
`call` (`{"session_id": ...}` or null). `PATCH` validates its fields (422 for unknown
fields, nulls, an empty model or voice, and a model or voice that Ollama or Chatterbox
does not list; a service that cannot be asked right now does not block the change); a
PATCH to the session in a call also queues update-settings frames into that call's
pipeline. Upstream failures in `/api/models`, `/api/voices` (502) and the first
`POST /api/sessions` (503) carry a plain-words `detail`.

On SIGTERM the server ends a running call (telling the page to call again), waits at
most a few seconds for it, and exits.

## Web UI

Plain HTML, CSS and JavaScript served by the app, with no build step. It talks to
`RTCPeerConnection` and the RTVI data-channel messages directly. If that gets awkward, it
can switch to Pipecat's JS client SDK vendored as ES modules.

- Session list, new session, and deleting a session after a confirmation dialog.
  Deleting the last session leaves a new empty one with the same settings.
- Model, voice, system prompt and "voice actions as sounds" controls for the current
  session.
- Call / Hang up button with state: connecting, warming up, listening, thinking,
  speaking.
- Live transcript, plus history loaded from the API. The bot's reply shows as the LLM
  generates it (`bot-llm-text`), muted until each sentence is reported spoken
  (`bot-tts-text`); after a call the page reloads the stored messages.
- The state stays "speaking" while a reply still has unspoken text, and "listening"
  follows "bot stopped speaking" only after a short delay, because Pipecat reports a
  stop between sentences whenever synthesis is slower than playback.
- Mute toggle.
- Works on a phone-width screen.
- Notices above the call controls for upstream problems and call errors: the page
  checks `/api/health` on load, every 30 s (10 s while Ollama is down) and when it
  regains focus, and disables Call while Ollama is unreachable; it warns when Chatterbox
  is not answering, when the session's model or voice is not installed, when a call
  runs elsewhere (another tab or device), and explains a missing or denied microphone.

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
| `VOICEDM_WHISPER_MODEL` | `base.en` | faster-whisper model name, or a directory holding a converted model |
| `VOICEDM_DEFAULT_MODEL` | first installed model | Model for new sessions |
| `VOICEDM_DEFAULT_VOICE` | first listed voice | Voice for new sessions |
| `VOICEDM_STUN_URL` | unset | Optional STUN server |
| `VOICEDM_TTS_TIMEOUT` | `120` | Seconds; allows for a TTS cold start |
| `VOICEDM_MEMORY_BUDGET` | `6000` | Characters of recent messages before the older half is folded into the summary |
| `VOICEDM_SUMMARY_PROMPT` | built in | Instructions for writing the summary |
| `VOICEDM_SUMMARY_TIMEOUT` | `300` | Seconds for one summarisation request |
| `VOICEDM_LOG_LEVEL` | `INFO` | Log level; `DEBUG` adds Pipecat's per-frame detail, including conversation text |

The memory settings have no dedicated NixOS module options; set them through the
module's `environment` option.

## Repository layout

```
flake.nix                  package, NixOS module, checks, dev shell
nix/package.nix            uv2nix build
nix/module.nix             services.voicedm
nix/vm-test.nix            NixOS VM test
nix/upstream_stub.py       fake Ollama and Chatterbox for the VM test
pyproject.toml, uv.lock
src/voicedm/
  __main__.py              entry point (uvicorn)
  config.py
  app.py                   FastAPI routes, static files
  bot.py                   Pipecat pipeline per call
  tts_chatterbox.py
  expressions.py           actions such as *laughs* to Chatterbox sound tags
  storage.py               SQLite access and migrations
  memory.py                prompt layout, summary folding
  upstream.py              Ollama and Chatterbox listing, health, warm-up
  static/                  index.html, app.js, style.css
scripts/call_client.py     headless test call against a running server
tests/
docs/
```

## Packaging and deployment

- **Development:** `uv` for Python dependencies, a flake dev shell with Python, `uv`,
  `ffmpeg` and `ruff`, and `LD_LIBRARY_PATH` set for the binary wheels (libstdc++,
  zlib). `scripts/call_client.py` makes a headless test call (aiortc) against a running
  server: it speaks WAV files, can barge in, and records and logs what comes back.
- **Package** (`nix/package.nix`): built from `uv.lock` with uv2nix (flake inputs
  `pyproject-nix`, `uv2nix`, `pyproject-build-systems`), preferring wheels, so the
  NixOS service uses exactly the pinned dependencies. pyproject.nix patches the
  binary wheels (onnxruntime, ctranslate2, av, aiortc, numpy, ...) with
  `autoPatchelfHook` and the manylinux libraries, so no `LD_LIBRARY_PATH` is needed;
  the only override is `docopt` (an sdist without its setuptools build dependency).
  `ffmpeg` is not needed at runtime (PyAV bundles its own libraries). The closure is
  about 1 GB, mostly the wheels' bundled libraries (OpenCV, which Pipecat depends on,
  ctranslate2, PyAV, onnxruntime, numpy). `nix run .` starts the server.
- **NixOS module** `services.voicedm` (`nix/module.nix`):
  - `enable`, `package`, `host`, `port`, `openFirewall`
  - `ollamaUrl`, `chatterboxUrl`, `whisperModel`, `stunUrl`
  - `whisperModelPath`: a local CTranslate2 Whisper model directory used instead of
    `whisperModel`, with `HF_HUB_OFFLINE=1`, for hosts without Hugging Face access
    (and the VM test). `VOICEDM_WHISPER_MODEL` takes a model name or such a directory.
  - `environment` for extra variables (the memory settings, default model and voice,
    TTS timeout, log level)
  - systemd service with `DynamicUser`, `StateDirectory=voicedm` (database plus the
    Whisper model under `models/`), `CacheDirectory=voicedm` for any other library
    cache (`HF_HOME`, `XDG_CACHE_HOME`), `Restart=on-failure`, and hardening
    (`systemd-analyze security`: 1.1 "OK"). Two usual options are left off because
    they break the app: `MemoryDenyWriteExecute` (pyOpenSSL's DTLS verify callback is
    a cffi callback that needs write+execute memory) and `ProcSubset=pid`
    (onnxruntime reads `/proc/cpuinfo`). Address families: `AF_INET`, `AF_INET6`,
    `AF_UNIX`, and `AF_NETLINK` for listing the host's addresses for ICE.
  - Firewall: aiortc binds each call's media socket to a kernel-chosen ephemeral UDP
    port and has no setting to pin a range (aioice binds port 0). Rather than patch
    aioice internals, `openFirewall` opens the HTTP port and the Linux default
    ephemeral range, UDP 32768-60999; the option description says so, and how to
    open it on the LAN interface only instead. Without the UDP range a call does not
    connect through the NixOS firewall (checked in the VM test).
- **VM test** (`checks.<system>.vm-test`, `nix/vm-test.nix`): a server with the
  module, its firewall on and `openFirewall`, fake Ollama and Chatterbox
  (`nix/upstream_stub.py`), and a pinned `faster-whisper-tiny.en` model (about 75 MB,
  fetched as fixed-output derivations); a client machine checks `/api/health`, the
  model and voice lists, creating and reading a session, the page, and makes one
  WebRTC call with `scripts/call_client.py` (a question synthesised with espeak-ng),
  which must be transcribed, answered and stored. Other checks: the package, `pytest`
  and `ruff` from the locked environment.
- **Wiring:** the NixOS configuration that hosts the app adds this flake as an input,
  imports `nixosModules.default`, enables the service and sets the two upstream URLs.
  The repository is public, so the input is `github:mantas6/voicedm`.

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
