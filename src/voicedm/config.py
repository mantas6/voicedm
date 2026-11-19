"""Configuration from ``VOICEDM_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# The system prompt of the first session; later sessions copy theirs.
DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly voice assistant in a spoken conversation. Your replies are "
    "read aloud, so answer in plain conversational sentences: no markdown, lists, "
    "emoji or code. Keep answers short unless asked for detail."
)

# Instructions for folding older messages into a session's summary (see memory.py).
DEFAULT_SUMMARY_PROMPT = (
    "You keep the memory of a spoken conversation between a user and an assistant. "
    "You get the summary so far and the next part of the conversation. Write one "
    "updated summary that merges both: facts about the user, names, numbers, "
    "preferences, decisions, promises and open questions, and what was talked about. "
    "Drop small talk and filler. Write plain third-person sentences, no lists or "
    "markdown, under 200 words. Reply with the summary only."
)


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
    # Characters of recent messages kept verbatim before the older half is folded
    # into the summary (roughly four characters per token).
    memory_budget: int = 6000
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    summary_timeout: float = 300.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = env.get(f"VOICEDM_{name}", "").strip()
            return value or None

        defaults = cls()
        budget = int(get("MEMORY_BUDGET") or defaults.memory_budget)
        if budget < 1:
            raise ValueError("VOICEDM_MEMORY_BUDGET must be a positive number of characters")
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
            memory_budget=budget,
            summary_prompt=get("SUMMARY_PROMPT") or defaults.summary_prompt,
            summary_timeout=float(get("SUMMARY_TIMEOUT") or defaults.summary_timeout),
        )
