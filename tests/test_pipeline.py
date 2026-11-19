"""The call pipeline with fake STT/LLM/TTS: turn persistence, interruptions, history
and mid-call settings changes.

The pipeline is the real one minus the transport and speech-to-text: the real context
aggregators with the bot's turn recorder, an Ollama LLM service whose completion is
scripted, and the Chatterbox TTS service against a fake server. User turns are driven
with the VAD and transcription frames the transport and STT would produce.
"""

import asyncio
import io
import json
import wave

import httpx
import pytest
from pipecat.frames.frames import (
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from voicedm.bot import TurnRecorder, settings_frames
from voicedm.memory import build_context
from voicedm.storage import Storage
from voicedm.tts_chatterbox import ChatterboxTTSService

TIMEOUT = 5.0
SENTENCE_SECONDS = 0.4  # audio the fake TTS server returns per sentence


async def eventually(check) -> None:
    """Poll the async ``check`` until it is true, for at most ``TIMEOUT`` seconds."""
    async with asyncio.timeout(TIMEOUT):
        while not await check():  # noqa: ASYNC110 - polling the database, no event to wait on
            await asyncio.sleep(0.01)


class ScriptedLLM(OLLamaLLMService):
    """Ollama service that answers from a script instead of the network.

    Each reply is a list of text chunks; an ``asyncio.Event`` in it holds the reply
    there until set (or until an interruption cancels it). Records the model, system
    instruction and messages of every request.
    """

    def __init__(self, replies: list[list], **kwargs) -> None:
        super().__init__(base_url="http://ollama.test/v1", **kwargs)
        self.replies = list(replies)
        self.requests: list[dict] = []

    async def _process_context(self, context: LLMContext) -> None:
        self.requests.append(
            {
                "model": self.settings.model,
                "system": self.settings.system_instruction,
                "messages": [dict(m) for m in context.get_messages()],
            }
        )
        for chunk in self.replies.pop(0):
            if isinstance(chunk, asyncio.Event):
                await chunk.wait()
            else:
                await self.push_frame(LLMTextFrame(chunk))


class PlaybackOutput(BaseOutputTransport):
    """Output transport that "plays" audio in real time to nowhere.

    Like the WebRTC one, it releases the TTS text as its audio starts playing and
    drops what is queued when an interruption arrives, which is what lets the
    assistant aggregator know how much of a reply was spoken.
    """

    def __init__(self) -> None:
        super().__init__(TransportParams(audio_out_enabled=True))

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        await asyncio.sleep(len(frame.audio) / 2 / frame.sample_rate / frame.num_channels)
        return True


def wav_bytes(seconds: float = SENTENCE_SECONDS, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x10" * int(rate * seconds))
    return buf.getvalue()


class Call:
    """One pipeline run against a session, with helpers to act as the user."""

    def __init__(self, store: Storage, session: dict, history: list[dict], replies) -> None:
        self.store = store
        self.session_id = session["id"]
        self.tts_requests: list[dict] = []
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self._tts))
        self.llm = ScriptedLLM(
            replies,
            settings=OLLamaLLMService.Settings(
                model=session["model"], system_instruction=session["system_prompt"]
            ),
        )
        self.tts = ChatterboxTTSService(
            base_url="http://tts.test", voice=session["voice"], client=self.http, timeout=2
        )
        aggregators = LLMContextAggregatorPair(
            LLMContext(messages=build_context(session, history)),
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.05)]
                ),
            ),
        )
        TurnRecorder(store, self.session_id, lambda: self.llm.settings.model).attach(aggregators)
        pipeline = Pipeline(
            [aggregators.user(), self.llm, self.tts, PlaybackOutput(), aggregators.assistant()]
        )
        self.worker = PipelineWorker(
            pipeline,
            params=PipelineParams(audio_out_sample_rate=24000),
            enable_rtvi=False,
            idle_timeout_secs=None,
        )
        self.runner = WorkerRunner(handle_sigint=False)
        self._run: asyncio.Task | None = None

    def _tts(self, request: httpx.Request) -> httpx.Response:
        self.tts_requests.append(json.loads(request.content))
        return httpx.Response(200, content=wav_bytes())

    async def __aenter__(self) -> "Call":
        started = asyncio.Event()

        @self.worker.event_handler("on_pipeline_started")
        async def on_started(worker, frame) -> None:
            started.set()

        await self.runner.add_workers(self.worker)
        self._run = asyncio.create_task(self.runner.run())
        await asyncio.wait_for(started.wait(), TIMEOUT)
        return self

    async def __aexit__(self, *exc) -> None:
        # Like a hang-up: the client disconnects and the worker is cancelled.
        await self.worker.cancel()
        await asyncio.wait_for(self._run, TIMEOUT)
        await self.http.aclose()

    async def say(self, text: str) -> None:
        """A user turn: speech starts, is transcribed, and stops."""
        await self.worker.queue_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.05)
        await self.worker.queue_frame(TranscriptionFrame(text, "user", "2026-01-01T00:00:00Z"))
        await self.worker.queue_frame(VADUserStoppedSpeakingFrame())

    async def wait_for_messages(self, count: int) -> list[dict]:
        messages: list[dict] = []

        async def enough() -> bool:
            nonlocal messages
            messages = await self.store.messages(self.session_id)
            return len(messages) >= count

        await eventually(enough)
        return messages


@pytest.fixture
async def store(tmp_path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


def summary(messages: list[dict]) -> list[tuple]:
    return [(m["role"], m["content"], m["model"], m["interrupted"]) for m in messages]


async def test_turns_are_saved_with_role_and_model(store):
    session = await store.create_session("model-a", "Alpha.wav", "Be brief.")
    replies = [["Hello", " there.", " How can I help?"]]
    async with Call(store, session, [], replies) as call:
        await call.say("Hi bot")
        messages = await call.wait_for_messages(2)
    assert summary(messages) == [
        ("user", "Hi bot", None, False),
        ("assistant", "Hello there. How can I help?", "model-a", False),
    ]
    assert call.llm.requests[0]["system"] == "Be brief."
    assert {r["voice"] for r in call.tts_requests} == {"Alpha.wav"}
    stored = await store.get_session(session["id"])
    assert stored["title"] == "Hi bot"
    assert stored["updated_at"] == messages[-1]["created_at"]


async def test_history_continues_across_calls(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    async with Call(store, session, [], [["Paris."]]) as call:
        await call.say("Capital of France?")
        await call.wait_for_messages(2)

    history = await store.messages(session["id"])
    async with Call(store, session, history, [["You asked about France."]]) as call:
        await call.say("What did I just ask?")
        messages = await call.wait_for_messages(4)
    assert call.llm.requests[0]["messages"] == [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "What did I just ask?"},
    ]
    assert [m["content"] for m in messages] == [
        "Capital of France?",
        "Paris.",
        "What did I just ask?",
        "You asked about France.",
    ]


async def into_second_sentence(call: Call) -> None:
    """Wait until the bot is half-way through playing the second sentence of a reply."""

    async def speaking() -> bool:
        return bool(call.tts_requests)

    await eventually(speaking)
    await asyncio.sleep(1.5 * SENTENCE_SECONDS)


# The TTS sentence aggregator needs the start of the next sentence to finish one, and
# the TTS service queues a sentence's text behind its audio, so the output transport
# releases it once the sentence has been played in full. "Spoken" therefore means the
# sentences that were played to the end.
STALLING_REPLY = ["The first sentence.", " The second sentence.", " The", "hold", " third."]


async def test_interrupted_reply_keeps_only_what_was_spoken(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    hold = asyncio.Event()  # never set: the reply stalls in its third sentence
    replies = [[hold if c == "hold" else c for c in STALLING_REPLY], ["Sure."]]
    async with Call(store, session, [], replies) as call:
        await call.say("Tell me a story")
        await into_second_sentence(call)
        await call.say("Stop, something else")
        messages = await call.wait_for_messages(4)
    assert summary(messages) == [
        ("user", "Tell me a story", None, False),
        ("assistant", "The first sentence.", "model-a", True),
        ("user", "Stop, something else", None, False),
        ("assistant", "Sure.", "model-a", False),
    ]
    # The next request sees the interrupted reply as far as it was spoken.
    assert call.llm.requests[1]["messages"][1] == {
        "role": "assistant",
        "content": "The first sentence.",
    }


async def test_hang_up_mid_reply_saves_the_spoken_part(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    hold = asyncio.Event()
    replies = [[hold if c == "hold" else c for c in STALLING_REPLY]]
    async with Call(store, session, [], replies) as call:
        await call.say("Go on")
        await into_second_sentence(call)
    messages = await store.messages(session["id"])
    assert summary(messages) == [
        ("user", "Go on", None, False),
        ("assistant", "The first sentence.", "model-a", True),
    ]


async def test_settings_changes_apply_from_the_next_turn(store):
    session = await store.create_session("model-a", "Alpha.wav", "Old prompt.")
    replies = [["One."], ["Two."]]
    async with Call(store, session, [], replies) as call:
        await call.say("First")
        await call.wait_for_messages(2)
        changes = {"model": "model-b", "voice": "Beta.wav", "system_prompt": ""}
        await call.worker.queue_frames(settings_frames(changes))
        await call.say("Second")
        messages = await call.wait_for_messages(4)
    # Pipecat keeps an empty system instruction as None: no system message either way.
    assert [(r["model"], r["system"] or "") for r in call.llm.requests] == [
        ("model-a", "Old prompt."),
        ("model-b", ""),
    ]
    assert [r["voice"] for r in call.tts_requests] == ["Alpha.wav", "Beta.wav"]
    assert [m["model"] for m in messages if m["role"] == "assistant"] == ["model-a", "model-b"]


def test_settings_frames_only_for_changed_services():
    assert settings_frames({}) == []
    assert len(settings_frames({"voice": "v"})) == 1
    assert len(settings_frames({"model": "m", "system_prompt": "p"})) == 1
    assert len(settings_frames({"model": "m", "voice": "v"})) == 2
