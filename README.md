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
into `VOICEDM_DATA_DIR`. Outside the dev shell, the binary wheels in uv's `.venv` need
`libstdc++` and `zlib` on `LD_LIBRARY_PATH` (the Nix package below does not).

A headless test call (speaks WAV files, can talk over the bot, records the reply,
then prints the session's stored messages; without `--session` it starts a new one):

```sh
uv run python scripts/call_client.py --url http://localhost:8780 --session 1 \
    --say question.wav --barge-in interrupt.wav --record reply.wav --duration 60
```

Sessions and their messages are kept in `voicedm.db` in `VOICEDM_DATA_DIR`.

Test and lint:

```sh
uv run pytest
uv run ruff check
uv run ruff format --check
```

Nix checks (package build, `pytest` and `ruff` from the locked environment, and a NixOS
VM test that makes a WebRTC call to the hardened service against fake upstreams):

```sh
nix flake check -L
nix run . # the packaged server, configured by the same VOICEDM_* variables
```

## Installing on NixOS

The flake provides the package (built from `uv.lock` with uv2nix) and a NixOS module.
Add the flake as an input of your system flake and import the module:

```nix
{
  inputs.voicedm.url = "git+ssh://git@example.org/you/voicedm.git"; # wherever it lives

  outputs = { nixpkgs, voicedm, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        voicedm.nixosModules.default
        {
          services.voicedm = {
            enable = true;
            ollamaUrl = "http://ollama.example.lan:11434";
            chatterboxUrl = "http://tts.example.lan:8004";
            openFirewall = true;
            # whisperModel = "small.en";
            # environment.VOICEDM_MEMORY_BUDGET = 8000;
          };
        }
      ];
    };
  };
}
```

The service listens on port 8780 and keeps its database and the Whisper model in
`/var/lib/voicedm`. It downloads the model from Hugging Face on first start and checks
for a newer revision on each start; for a host without that access, point
`whisperModelPath` at a model directory instead. All options are in
[nix/module.nix](nix/module.nix).

- **Microphone and HTTPS:** browsers only allow the microphone on a secure origin, so
  for use from a phone or another computer put an HTTPS reverse proxy in front of the
  HTTP port (see "HTTPS and the microphone" in the [plan](docs/plan.md)). v1 has no
  authentication: keep it on a trusted network or add access control in the proxy.
- **Firewall:** the call audio goes straight from the browser to the host over UDP,
  not through the proxy. WebRTC (aiortc) uses a kernel-chosen ephemeral UDP port per
  call and cannot be pinned to a range, so `openFirewall` opens TCP 8780 and the whole
  default ephemeral range, UDP 32768-60999. To limit that to the LAN, leave
  `openFirewall` off and open the same ports with
  `networking.firewall.interfaces.<lan>.allowedUDPPortRanges`.
