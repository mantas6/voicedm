from pathlib import Path

import pytest

from voicedm.config import DEFAULT_SUMMARY_PROMPT, Config


def test_defaults():
    config = Config.from_env({})
    assert config.host == "0.0.0.0"
    assert config.port == 8780
    assert config.data_dir == Path("./data")
    assert config.ollama_url == "http://localhost:11434"
    assert config.chatterbox_url == "http://localhost:8004"
    assert config.whisper_model == "base.en"
    assert config.default_model is None
    assert config.default_voice is None
    assert config.stun_url is None
    assert config.tts_timeout == 120.0
    assert config.memory_budget == 6000
    assert config.summary_prompt == DEFAULT_SUMMARY_PROMPT
    assert config.summary_timeout == 300.0


def test_overrides():
    config = Config.from_env(
        {
            "VOICEDM_HOST": "127.0.0.1",
            "VOICEDM_PORT": "9000",
            "VOICEDM_DATA_DIR": "/srv/voicedm",
            "VOICEDM_OLLAMA_URL": "http://ollama.example:11434/",
            "VOICEDM_CHATTERBOX_URL": "http://tts.example:8004/",
            "VOICEDM_WHISPER_MODEL": "small.en",
            "VOICEDM_DEFAULT_MODEL": "example-model",
            "VOICEDM_DEFAULT_VOICE": "example-voice.wav",
            "VOICEDM_STUN_URL": "stun:stun.example:3478",
            "VOICEDM_TTS_TIMEOUT": "30.5",
            "VOICEDM_MEMORY_BUDGET": "800",
            "VOICEDM_SUMMARY_PROMPT": "Summarise.",
            "VOICEDM_SUMMARY_TIMEOUT": "60",
        }
    )
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.data_dir == Path("/srv/voicedm")
    assert config.ollama_url == "http://ollama.example:11434"
    assert config.chatterbox_url == "http://tts.example:8004"
    assert config.whisper_model == "small.en"
    assert config.default_model == "example-model"
    assert config.default_voice == "example-voice.wav"
    assert config.stun_url == "stun:stun.example:3478"
    assert config.tts_timeout == 30.5
    assert config.memory_budget == 800
    assert config.summary_prompt == "Summarise."
    assert config.summary_timeout == 60.0


def test_empty_values_fall_back_to_defaults():
    config = Config.from_env({"VOICEDM_PORT": "", "VOICEDM_STUN_URL": "  "})
    assert config.port == 8780
    assert config.stun_url is None


def test_memory_budget_must_be_positive():
    with pytest.raises(ValueError):
        Config.from_env({"VOICEDM_MEMORY_BUDGET": "0"})
