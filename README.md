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

Test and lint:

```sh
uv run pytest
uv run ruff check
uv run ruff format --check
```
