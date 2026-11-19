"""Configuration from ``VOICEDM_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8780
    data_dir: Path = Path("./data")
    ollama_url: str = "http://localhost:11434"
    chatterbox_url: str = "http://localhost:8004"
    whisper_model: str = "base.en"
    # None means "first installed model" / "first listed voice", resolved at runtime.
    default_model: str | None = None
    default_voice: str | None = None
    stun_url: str | None = None
    tts_timeout: float = 120.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = env.get(f"VOICEDM_{name}", "").strip()
            return value or None

        defaults = cls()
        return cls(
            host=get("HOST") or defaults.host,
            port=int(get("PORT") or defaults.port),
            data_dir=Path(get("DATA_DIR") or defaults.data_dir),
            ollama_url=(get("OLLAMA_URL") or defaults.ollama_url).rstrip("/"),
            chatterbox_url=(get("CHATTERBOX_URL") or defaults.chatterbox_url).rstrip("/"),
            whisper_model=get("WHISPER_MODEL") or defaults.whisper_model,
            default_model=get("DEFAULT_MODEL"),
            default_voice=get("DEFAULT_VOICE"),
            stun_url=get("STUN_URL"),
            tts_timeout=float(get("TTS_TIMEOUT") or defaults.tts_timeout),
        )
