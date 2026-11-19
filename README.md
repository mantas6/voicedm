<img src="assets/icon.svg" width="64" height="64" alt="">

# voicedm

Created in [T3 Code](https://t3.codes).

Self-hosted, hands-free voice conversation with a local LLM, in the browser. See
[docs/plan.md](docs/plan.md) for the design.

## Development

Enter the dev shell (Python, `uv`, `ffmpeg`, `ruff`), or use your own `uv`:

```sh
nix develop
uv sync
```

Run the server (settings come from `VOICEDM_*` environment variables, see the
Configuration section of the plan):

```sh
VOICEDM_OLLAMA_URL=http://localhost:11434 \
VOICEDM_CHATTERBOX_URL=http://localhost:8004 \
uv run voicedm            # or: uv run python -m voicedm
curl http://localhost:8780/api/health
```

Open the page, press **Call** and talk. The first start downloads the Whisper model
into `VOICEDM_DATA_DIR`. Outside the dev shell, the binary wheels need `libstdc++` and
`zlib` on `LD_LIBRARY_PATH`.

A headless test call (speaks WAV files, can talk over the bot, records the reply):

```sh
uv run python scripts/call_client.py --url http://localhost:8780 \
    --say question.wav --barge-in interrupt.wav --record reply.wav --duration 60
```

Test and lint:

```sh
uv run pytest
uv run ruff check
uv run ruff format --check
```
