"""The call pipeline with fake STT/LLM/TTS: turn persistence, interruptions, history,
mid-call settings changes and summary folding.

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

from voicedm.bot import ErrorReporter, TurnRecorder, TurnTimingObserver, settings_frames
from voicedm.memory import SUMMARY_HEADER, CallMemory, MemoryLLMService, build_context
from voicedm.storage import Storage
from voicedm.tts_chatterbox import ChatterboxTTSService

TIMEOUT = 5.0
SENTENCE_SECONDS = 0.4  # audio the fake TTS server returns per sentence


async def eventually(check) -> None:
    """Poll the async ``check`` until it is true, for at most ``TIMEOUT`` seconds."""
    async with asyncio.timeout(TIMEOUT):
        while not await check():  # noqa: ASYNC110 - polling the database, no event to wait on
            await asyncio.sleep(0.01)


class ScriptedLLM(MemoryLLMService):
    """The call's LLM service, answering from a script instead of the network.

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

    def __init__(
        self,
        store: Storage,
        session: dict,
        history: list[dict],
        replies,
        *,
        summarise=None,
        budget: int = 6000,
        tts_status: int = 200,
    ) -> None:
        self.tts_status = tts_status
        self.store = store
        self.session_id = session["id"]
        self.tts_requests: list[dict] = []
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self._tts))
        self.llm = ScriptedLLM(
            replies,
            summary=session["summary"],
            settings=OLLamaLLMService.Settings(
                model=session["model"], system_instruction=session["system_prompt"]
            ),
        )
        self.tts = ChatterboxTTSService(
            base_url="http://tts.test",
            voice=session["voice"],
            client=self.http,
            timeout=2,
            expressions=session["expressions"],
        )
        self.context = LLMContext(messages=build_context(session, history))
        self.memory = CallMemory(
            store, session, self.context, self.llm, summarise or self._no_summary, budget
        )
        aggregators = LLMContextAggregatorPair(
            self.context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.05)]
                ),
            ),
        )
        TurnRecorder(
            store,
            self.session_id,
            lambda: self.llm.settings.model,
            after_reply=self.memory.request_fold,
        ).attach(aggregators)
        output = PlaybackOutput()
        pipeline = Pipeline(
            [aggregators.user(), self.llm, self.tts, output, aggregators.assistant()]
        )
        self.timing = TurnTimingObserver({"llm": self.llm, "tts": self.tts}, output)
        self.worker = PipelineWorker(
            pipeline,
            params=PipelineParams(audio_out_sample_rate=24000, enable_metrics=True),
            enable_rtvi=False,
            idle_timeout_secs=None,
            observers=[self.timing],
        )
        self.reports: list[tuple[str, str]] = []
        self.gave_up: list[str] = []
        errors = ErrorReporter(
            self.llm, self.tts, lambda s, m: self.reports.append((s, m)), self.gave_up.append
        )

        @self.worker.event_handler("on_pipeline_error")
        async def on_error(worker, frame) -> None:
            errors.handle(frame)

        self.runner = WorkerRunner(handle_sigint=False)
        self._run: asyncio.Task | None = None

    @staticmethod
    async def _no_summary(previous: str, messages: list[dict]) -> str:
        raise AssertionError("no fold expected")

    def _tts(self, request: httpx.Request) -> httpx.Response:
        self.tts_requests.append(json.loads(request.content))
        if self.tts_status != 200:
            return httpx.Response(self.tts_status, json={"detail": "the voice server broke"})
        return httpx.Response(200, content=wav_bytes())

    async def __aenter__(self) -> "Call":
        started = asyncio.Event()

        @self.worker.event_handler("on_pipeline_started")
        async def on_started(worker, frame) -> None:
            started.set()

        await self.runner.add_workers(self.worker)
        self._run = asyncio.create_task(self.runner.run())
        await asyncio.wait_for(started.wait(), TIMEOUT)
        self.memory.request_fold()  # as the bot does once the client connects
        return self

    async def __aexit__(self, *exc) -> None:
        # Like a hang-up: the client disconnects and the worker is cancelled.
        await self.worker.cancel()
        await asyncio.wait_for(self._run, TIMEOUT)
        await self.memory.close()
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


ACTED = ["*laughs*", " I can't", " believe it.", " *smiles*", " Well done.", " *sighs*"]
ACTED_TEXT = "*laughs* I can't believe it. *smiles* Well done. *sighs*"


async def test_actions_are_voiced_as_sounds_and_kept_as_written(store):
    session = await store.create_session("model-a", "Alpha.wav", "", expressions=True)
    async with Call(store, session, [], [ACTED]) as call:
        await call.say("Tell me")
        messages = await call.wait_for_messages(2)
    spoken = " ".join(r["input"].strip() for r in call.tts_requests)
    # Only the actions Chatterbox has a sound for change, and only in the request.
    assert spoken == "[laugh] I can't believe it. *smiles* Well done. [sigh]"
    assert messages[1]["content"] == ACTED_TEXT


async def test_actions_are_read_as_written_by_default(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    async with Call(store, session, [], [ACTED]) as call:
        await call.say("Tell me")
        messages = await call.wait_for_messages(2)
    assert " ".join(r["input"].strip() for r in call.tts_requests) == ACTED_TEXT
    assert messages[1]["content"] == ACTED_TEXT


async def test_expressions_can_be_switched_during_a_call(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    async with Call(store, session, [], [["*sighs* One."], ["*sighs* Two."]]) as call:
        await call.say("First")
        await call.wait_for_messages(2)
        call.tts.expressions = True
        await call.say("Second")
        await call.wait_for_messages(4)
    assert [r["input"].strip() for r in call.tts_requests] == ["*sighs* One.", "[sigh] Two."]


def test_settings_frames_only_for_changed_services():
    assert settings_frames({}) == []
    assert len(settings_frames({"voice": "v"})) == 1
    assert len(settings_frames({"model": "m", "system_prompt": "p"})) == 1
    assert len(settings_frames({"model": "m", "voice": "v"})) == 2


# -- summary memory --


class Summariser:
    """Records each summarisation request; answers "S1", "S2", ... once ``release`` is
    set."""

    def __init__(self, release: asyncio.Event | None = None) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.release = release

    async def __call__(self, previous: str, messages: list[dict]) -> str:
        self.calls.append((previous, [m["content"] for m in messages]))
        if self.release:
            await self.release.wait()
        return f"S{len(self.calls)}"


def is_extension(longer: dict, shorter: dict) -> bool:
    """Whether request ``longer`` starts with all of ``shorter`` (same system
    instruction, same messages) and adds to it: what keeps Ollama's cache valid."""
    n = len(shorter["messages"])
    return (
        longer["system"] == shorter["system"]
        and longer["messages"][:n] == shorter["messages"]
        and len(longer["messages"]) > n
    )


async def stored_summary(store: Storage, session_id: int) -> str:
    return (await store.get_session(session_id))["summary"]


async def test_fold_mid_call_between_turns_keeps_the_prefix_stable(store):
    session = await store.create_session("model-a", "Alpha.wav", "Be brief.")
    sid = session["id"]
    summarise = Summariser()
    replies = [["Answer one."], ["Answer two."], ["Three."], ["Four."]]
    # 19 + 11 characters fit in the budget; the second exchange takes it to 61.
    async with Call(store, session, [], replies, summarise=summarise, budget=60) as call:
        await call.say("First question here")
        await call.wait_for_messages(2)
        await call.say("Second question here")
        await call.wait_for_messages(4)

        async def folded() -> bool:
            return await stored_summary(store, sid) == "S1"

        await eventually(folded)
        await call.memory.wait()
        await call.say("Third?")
        await call.wait_for_messages(6)
        await call.say("Fourth?")
        messages = await call.wait_for_messages(8)

    first, second, third, fourth = call.llm.requests
    # No fold between the first two turns: the second request extends the first.
    assert first["system"] == "Be brief."
    assert is_extension(second, first)
    # The older half (the first exchange) was folded into one summarisation request.
    assert summarise.calls == [("", ["First question here", "Answer one."])]
    assert third["system"] == f"Be brief.\n\n{SUMMARY_HEADER}\nS1"
    assert [m["content"] for m in third["messages"]] == [
        "Second question here",
        "Answer two.",
        "Third?",
    ]
    # And from there on, turns extend the prompt again.
    assert is_extension(fourth, third)
    # Nothing is deleted; the summary covers up to the first reply.
    assert len(messages) == 8
    stored = await store.get_session(sid)
    assert (stored["summary"], stored["summary_upto"]) == ("S1", messages[1]["id"])


async def test_fold_finishing_during_a_turn_switches_after_it(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    sid = session["id"]
    await store.add_message(sid, "user", "Earlier question")
    await store.add_message(sid, "assistant", "Earlier answer.", model="model-a")
    history = await store.messages(sid)
    release, hold = asyncio.Event(), asyncio.Event()
    summarise = Summariser(release)
    replies = [["Cats purr."], ["Dogs", hold, " bark."], ["Birds sing."]]
    # 31 characters at the start fit; the first exchange takes it over 50.
    async with Call(store, session, history, replies, summarise=summarise, budget=50) as call:
        await call.say("Tell me about cats")
        await call.wait_for_messages(4)

        async def summarising() -> bool:
            return bool(summarise.calls)

        await eventually(summarising)
        # The next turn starts while the summary is being written...
        await call.say("And dogs?")

        async def answering() -> bool:
            return len(call.llm.requests) == 2

        await eventually(answering)
        # ...and the summary is stored while that turn is being answered.
        release.set()

        async def folded() -> bool:
            return await stored_summary(store, sid) == "S1"

        await eventually(folded)
        assert (await store.get_session(sid))["summary_upto"] == history[1]["id"]
        await asyncio.sleep(0.2)
        # The turn in flight keeps the old context until its answer is done.
        assert call.context.messages[0]["content"] == "Earlier question"
        hold.set()
        await call.wait_for_messages(6)
        await call.memory.wait()
        await call.say("And birds?")
        messages = await call.wait_for_messages(8)

    # (The last exchange takes the context over budget again, so a second fold may
    # start as the call ends.)
    assert summarise.calls[0] == ("", ["Earlier question", "Earlier answer."])
    second, third = call.llm.requests[1:]
    assert second["system"] is None
    assert [m["content"] for m in second["messages"]] == [
        "Earlier question",
        "Earlier answer.",
        "Tell me about cats",
        "Cats purr.",
        "And dogs?",
    ]
    # The turn that ran during the fold is not lost: its messages follow the kept ones.
    assert third["system"] == f"{SUMMARY_HEADER}\nS1"
    assert [m["content"] for m in third["messages"]] == [
        "Tell me about cats",
        "Cats purr.",
        "And dogs?",
        "Dogs bark.",
        "And birds?",
    ]
    assert [m["content"] for m in messages][-4:] == [
        "And dogs?",
        "Dogs bark.",
        "And birds?",
        "Birds sing.",
    ]


async def test_call_starts_from_the_summary_and_the_messages_after_it(store):
    session = await store.create_session("model-a", "Alpha.wav", "Be brief.")
    sid = session["id"]
    for role, text in [("user", "My cat is Tom."), ("assistant", "Nice."), ("user", "Hi again")]:
        await store.add_message(sid, role, text, model="model-a" if role == "assistant" else None)
    history = await store.messages(sid)
    await store.set_summary(sid, "The user's cat is called Tom.", history[1]["id"])
    session = await store.get_session(sid)
    async with Call(store, session, history, [["Hello."]]) as call:
        await call.say("What is my cat called?")
        await call.wait_for_messages(5)
    request = call.llm.requests[0]
    assert request["system"] == f"Be brief.\n\n{SUMMARY_HEADER}\nThe user's cat is called Tom."
    # Two user messages in a row (a reply that was never spoken) are replayed as they are.
    assert request["messages"] == [
        {"role": "user", "content": "Hi again"},
        {"role": "user", "content": "What is my cat called?"},
    ]


async def test_over_budget_session_is_folded_after_connect(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    sid = session["id"]
    for i in range(3):
        await store.add_message(sid, "user", f"Question number {i}")
        await store.add_message(sid, "assistant", f"Answer number {i}.", model="model-a")
    history = await store.messages(sid)
    summarise = Summariser()
    async with Call(store, session, history, [["Yes."]], summarise=summarise, budget=80) as call:
        await call.memory.wait()
        await call.say("Still there?")
        await call.wait_for_messages(8)
    assert summarise.calls == [
        ("", ["Question number 0", "Answer number 0.", "Question number 1", "Answer number 1."])
    ]
    assert [m["content"] for m in call.llm.requests[0]["messages"]] == [
        "Question number 2",
        "Answer number 2.",
        "Still there?",
    ]


async def test_settings_change_keeps_the_summary(store):
    session = await store.create_session("model-a", "Alpha.wav", "Old prompt.")
    await store.set_summary(session["id"], "Earlier things.", 0)
    session = await store.get_session(session["id"])
    async with Call(store, session, [], [["One."], ["Two."]]) as call:
        await call.say("First")
        await call.wait_for_messages(2)
        await call.worker.queue_frames(settings_frames({"system_prompt": "New prompt."}))
        await call.say("Second")
        await call.wait_for_messages(4)
    assert [r["system"] for r in call.llm.requests] == [
        f"Old prompt.\n\n{SUMMARY_HEADER}\nEarlier things.",
        f"New prompt.\n\n{SUMMARY_HEADER}\nEarlier things.",
    ]


# -- latency logging and errors in a running call --


async def test_turn_timing_in_a_call_with_an_interruption(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    hold = asyncio.Event()
    replies = [[hold if c == "hold" else c for c in STALLING_REPLY], ["Sure."]]
    async with Call(store, session, [], replies) as call:
        await call.say("Tell me a story")
        await into_second_sentence(call)
        await call.say("Stop, something else")
        await call.wait_for_messages(4)

        async def logged() -> bool:
            return len(call.timing.turns) == 2

        await eventually(logged)
    first, second = call.timing.turns
    assert (first["interrupted"], second["interrupted"]) == (True, False)
    for turn in (first, second):
        assert 0 < turn["turn_end"] <= turn["first_text"] <= turn["first_audio"]
        assert turn["ttfb_tts"] is not None
    assert second["llm_end"] is not None
    # The second turn's times are its own, not the interrupted reply's.
    assert second["first_audio"] < 2.0


async def test_speech_that_keeps_failing_ends_the_call(store):
    session = await store.create_session("model-a", "Alpha.wav", "")
    replies = [["One. Two. Three. Four."]]
    async with Call(store, session, [], replies, tts_status=500) as call:
        await call.say("Count to four")

        async def gave_up() -> bool:
            return bool(call.gave_up)

        await eventually(gave_up)
    assert len(call.tts_requests) >= 3
    service, message = call.reports[0]
    assert service == "tts"
    assert "HTTP 500" in message and "the voice server broke" in message
