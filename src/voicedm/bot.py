"""The Pipecat voice pipeline, one per call."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import openai
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from voicedm.config import Config
from voicedm.memory import CallMemory, MemoryLLMService, build_context, ollama_summariser
from voicedm.storage import Storage
from voicedm.tts_chatterbox import ChatterboxTTSService
from voicedm.upstream import OLLAMA, UpstreamError, Upstreams, explain

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

    from voicedm.app import ActiveCall

IN_SAMPLE_RATE = 16000  # what Silero and Whisper read
OUT_SAMPLE_RATE = 24000  # what Chatterbox produces

# Seconds for the client to connect once it has the answer to its offer.
CONNECT_TIMEOUT = 30.0
# Seconds without any audio from the client after which its connection counts as gone.
# A browser sends audio packets continuously, muted or not.
SILENCE_TIMEOUT = 10.0
# Seconds for a last message to the page to go out before the server ends a call.
END_MESSAGE_GRACE = 0.3


class SharedWhisperSTTService(WhisperSTTService):
    """``WhisperSTTService`` that uses an already loaded model instead of loading one."""

    def __init__(self, *, whisper: WhisperModel, **kwargs) -> None:
        self._shared_model = whisper
        super().__init__(**kwargs)

    def _load(self) -> None:
        self._model = self._shared_model


@dataclass
class _Turn:
    number: int
    speech_end: float | None = None  # the last time the VAD heard the user stop
    replying: bool = False  # the user's turn has ended and the reply is under way
    marks: dict[str, float] = field(default_factory=dict)  # event -> monotonic time
    ttfb: dict[str, float] = field(default_factory=dict)  # service -> seconds


class TurnTimingObserver(BaseObserver):
    """Logs one line per turn: where the time from the end of the user's speech to
    the bot's first audio went.

    Times are seconds after the end of the user's speech as the VAD reports it
    (which is itself the VAD's stop delay after the user fell silent):

    - ``transcript``: the transcript arrived; ``turn_end``: the user's turn was
      declared over and the reply started;
    - ``first_text``: the first text the LLM produced (for a model that reasons
      first, much later than its time to first byte);
    - ``first_audio``: the bot's first audio started playing (the headline number);
    - ``llm_end``: the LLM finished its reply;
    - ``ttfb_stt``, ``ttfb_llm``, ``ttfb_tts``: Pipecat's time-to-first-byte metric
      of each service for this turn (TTS: its first sentence).

    A turn ends when its reply has been played to the end, or when the user starts
    the next turn (``interrupted`` if a reply was under way then), or with the call.
    What the interrupted reply's services report after that is not counted again.
    """

    MARKS = ("transcript", "turn_end", "first_text", "first_audio", "llm_end")

    def __init__(
        self,
        services: dict[str, FrameProcessor],
        output: FrameProcessor,
        label: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._clock = clock
        self._roles = {processor.name: role for role, processor in services.items()}
        self._output = output
        self._label = label
        self._count = 0
        self._turn: _Turn | None = None
        self.turns: list[dict] = []  # what was logged, for tests

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        frame, now, turn = data.frame, self._clock(), self._turn
        # The reply's end, once the output transport has played everything before it.
        if isinstance(frame, LLMFullResponseEndFrame) and data.source is self._output:
            if turn and turn.replying:
                self._finish(interrupted=False)
            return
        # Broadcast frames travel both ways as two frames; count the downstream one,
        # where it is first pushed.
        if not data.first_push:
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            # A reply still under way is interrupted; speech that ended no turn
            # (and got no reply) is not a turn to report.
            if turn and turn.replying:
                self._finish(interrupted=True)
            self._count += 1
            self._turn = _Turn(self._count)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            if turn and turn.replying:
                self._finish(interrupted=True)
        elif turn is None:
            return
        elif not turn.replying:
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                turn.speech_end = now
            elif isinstance(frame, TranscriptionFrame):
                turn.marks["transcript"] = now
            elif isinstance(frame, UserStoppedSpeakingFrame):
                turn.replying = True
                turn.marks["turn_end"] = now
            elif isinstance(frame, MetricsFrame):
                self._ttfb(turn, frame, ("stt",), latest=True)
        elif isinstance(frame, LLMTextFrame):
            turn.marks.setdefault("first_text", now)
        elif isinstance(frame, BotStartedSpeakingFrame):
            turn.marks.setdefault("first_audio", now)
        elif isinstance(frame, LLMFullResponseEndFrame):
            turn.marks.setdefault("llm_end", now)
        elif isinstance(frame, MetricsFrame):
            self._ttfb(turn, frame, ("llm", "tts"), latest=False)

    def _ttfb(self, turn: _Turn, frame: MetricsFrame, roles: tuple, latest: bool) -> None:
        for metric in frame.data:
            role = self._roles.get(metric.processor)
            fresh = latest or role not in turn.ttfb
            if isinstance(metric, TTFBMetricsData) and role in roles and fresh:
                turn.ttfb[role] = metric.value

    def _finish(self, interrupted: bool) -> None:
        turn, self._turn = self._turn, None
        if turn is None:
            return
        base = turn.speech_end

        def seconds(value: float | None) -> str:
            return "-" if value is None else f"{value:.2f}"

        values: dict[str, float | None] = {
            name: (turn.marks[name] - base if base and name in turn.marks else None)
            for name in self.MARKS
        }
        for role in ("stt", "llm", "tts"):
            values[f"ttfb_{role}"] = turn.ttfb.get(role)
        self.turns.append({"turn": turn.number, **values, "interrupted": interrupted})
        fields = " ".join(f"{name}={seconds(value)}" for name, value in values.items())
        logger.info(
            f"Turn timing{self._label}: turn={turn.number} {fields} "
            f"interrupted={'yes' if interrupted else 'no'}"
        )


class InputAudioWatch(BaseObserver):
    """Remembers when audio last arrived from the client.

    A browser sends audio packets all the time, even when muted (silence), so a
    long gap means the connection is gone, whether or not WebRTC has noticed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last = time.monotonic()

    async def on_push_frame(self, data: FramePushed) -> None:
        if isinstance(data.frame, InputAudioRawFrame):
            self.last = time.monotonic()


class TurnRecorder:
    """Saves each finished turn of a call as a message of the call's session.

    Hooks the context aggregators' turn events: the user aggregator reports the text
    it wrote to the context, the assistant aggregator the reply as far as it was
    spoken (it sits after the output transport, which releases text as its audio
    plays) and whether it was interrupted. Event handlers run as separate tasks, so
    a lock keeps the writes in the order the turns finished. ``after_reply`` is called
    once each reply has ended and been saved: between turns.
    """

    def __init__(
        self,
        store: Storage,
        session_id: int,
        llm_model: Callable[[], str],
        after_reply: Callable[[], None] | None = None,
        on_save_error: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self._llm_model = llm_model
        self._after_reply = after_reply
        self._on_save_error = on_save_error
        self._reply_model: str | None = None
        self._lock = asyncio.Lock()

    def attach(self, aggregators: LLMContextAggregatorPair) -> None:
        user, assistant = aggregators.user(), aggregators.assistant()

        @user.event_handler("on_user_turn_message_added")
        async def on_user_turn(aggregator, message: UserTurnMessageAddedMessage) -> None:
            await self._save("user", message.content)

        @assistant.event_handler("on_assistant_turn_started")
        async def on_assistant_started(aggregator) -> None:
            # A model change mid-reply applies from the next turn; remember this one's.
            self._reply_model = self._llm_model()

        @assistant.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn(aggregator, message: AssistantTurnStoppedMessage) -> None:
            model = self._reply_model or self._llm_model()
            await self._save("assistant", message.content, model, message.interrupted)
            if self._after_reply:
                self._after_reply()

    async def _save(
        self, role: str, content: str, model: str | None = None, interrupted: bool = False
    ) -> None:
        # The aggregators join transcript segments and sentences with runs of spaces.
        content = " ".join(content.split())
        async with self._lock:
            # A reply interrupted before any of it was spoken leaves nothing to keep.
            if not content:
                return
            try:
                await self.store.add_message(
                    self.session_id, role, content, model=model, interrupted=interrupted
                )
            except Exception:
                # The call goes on; the page is told the history has a gap.
                logger.exception(f"Session {self.session_id}: could not save a {role} message")
                if self._on_save_error:
                    await self._on_save_error()


def settings_frames(changes: dict[str, Any]) -> list[Frame]:
    """Update-settings frames for changed session settings, to queue into a running call.

    The LLM and TTS services apply them when the frames reach them, between turns.
    """
    frames: list[Frame] = []
    llm: dict[str, str] = {}
    if "model" in changes:
        llm["model"] = changes["model"]
    if "system_prompt" in changes:
        llm["system_instruction"] = changes["system_prompt"]
    if llm:
        frames.append(LLMUpdateSettingsFrame(delta=OLLamaLLMService.Settings(**llm)))
    if "voice" in changes:
        delta = ChatterboxTTSService.Settings(voice=changes["voice"])
        frames.append(TTSUpdateSettingsFrame(delta=delta))
    return frames


def load_whisper(config: Config) -> WhisperModel:
    """Load the faster-whisper model on CPU (``int8``), downloading it on first use."""
    from faster_whisper import WhisperModel

    root = config.data_dir / "models"
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading Whisper model {config.whisper_model}")
    return WhisperModel(
        config.whisper_model, device="cpu", compute_type="int8", download_root=str(root)
    )


def explain_llm_error(error: ErrorFrame, model: str) -> str:
    """A message for the page about a failed LLM request."""
    e = error.exception
    if isinstance(e, openai.APITimeoutError | TimeoutError):
        failure = UpstreamError(str(e), "timeout")
    elif isinstance(e, openai.APIConnectionError):
        failure = UpstreamError(str(e), "unreachable")
    elif isinstance(e, openai.APIStatusError):
        failure = UpstreamError(str(e), "status", e.status_code, getattr(e, "message", "")[:200])
    else:
        return f"The language model failed: {error.error}"
    return explain(failure, OLLAMA, model)


# Sentences in a row that the TTS service failed to speak before a call is ended.
MAX_TTS_FAILURES = 3
TTS_GAVE_UP = (
    "Speech has stopped working, so the call was ended. Check the voice server "
    "(Chatterbox) and call again."
)


class ErrorReporter:
    """Turns the error frames that reach the pipeline worker into messages for the
    page: ``report(service, message)``.

    Speech that has stopped working ends the call (``give_up(message)``) rather than
    leaving a bot that answers in silence: after ``MAX_TTS_FAILURES`` sentences in a
    row failed, or once Pipecat has written the TTS service off for the rest of the
    call (it adds its own error for each reply that got no audio, and does that after
    a few in a row).
    """

    def __init__(
        self,
        llm: OLLamaLLMService,
        tts: ChatterboxTTSService,
        report: Callable[[str, str], None],
        give_up: Callable[[str], None],
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.report = report
        self.give_up = give_up
        self._gave_up = False

    def handle(self, frame: ErrorFrame) -> None:
        if frame.processor is self.tts:
            if "completed with no audio" not in frame.error:
                self.report("tts", frame.error)  # already a message for the page
            failing = self.tts.failures_in_a_row >= MAX_TTS_FAILURES
            if (failing or not self.tts.is_usable) and not self._gave_up:
                self._gave_up = True
                self.give_up(TTS_GAVE_UP)
        elif frame.processor is self.llm:
            self.report("llm", explain_llm_error(frame, self.llm.settings.model))
        else:
            self.report("other", f"Something went wrong: {frame.error}")


class Bot:
    """Holds the models shared by all calls and runs one pipeline per call."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.whisper: WhisperModel | None = None

    async def load(self) -> None:
        """Load the speech-to-text model once, at startup."""
        self.whisper = await asyncio.to_thread(load_whisper, self.config)

    async def run(
        self,
        connection: SmallWebRTCConnection,
        upstreams: Upstreams,
        store: Storage,
        call: ActiveCall,
    ) -> None:
        """Run the pipeline for one call on ``call.session_id`` until the call ends.

        A call ends when the client hangs up or its connection is gone, when the
        server ends it (``call.end``: shutdown, a replacing call, speech that has
        stopped working), or when the client never connects. The connection is
        closed however it ends, also if setting the pipeline up fails.
        """
        tasks: set[asyncio.Task] = set()
        memory: CallMemory | None = None
        try:
            worker, memory = await self._prepare(connection, upstreams, store, call, tasks)
            runner = WorkerRunner(handle_sigint=False)
            await runner.add_workers(worker)
            await runner.run()
        finally:
            call.detach()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if memory:
                await memory.close()
            await connection.disconnect()
            logger.info(f"Call {connection.pc_id} ended")

    async def _prepare(
        self,
        connection: SmallWebRTCConnection,
        upstreams: Upstreams,
        store: Storage,
        call: ActiveCall,
        tasks: set[asyncio.Task],
    ) -> tuple[PipelineWorker, CallMemory]:
        """Build the call's pipeline and hook up its events; ``tasks`` collects the
        background tasks to cancel when the call ends."""
        if self.whisper is None:
            raise RuntimeError("Bot.load() has not run")
        config = self.config
        pc_id = connection.pc_id
        session = await store.get_session(call.session_id)
        if session is None:
            raise RuntimeError(f"session {call.session_id} does not exist")
        history = await store.messages(call.session_id)
        model, voice = session["model"], session["voice"]
        context = LLMContext(messages=build_context(session, history))
        logger.info(
            f"Call {pc_id} on session {call.session_id}: model {model}, voice {voice}, "
            f"{len(history)} earlier messages, {len(context.messages)} of them after the "
            f"summary ({len(session['summary'])} characters)"
        )

        def background(coro: Coroutine) -> None:
            task = asyncio.create_task(coro)
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        def send(data: dict) -> None:
            """A server message to the page, straight onto the data channel (queued
            until it opens), so it is not held up behind the pipeline's frames."""
            message = {"label": "rtvi-ai", "type": "server-message", "data": data}
            connection.send_app_message(message)

        def report(service: str, message: str) -> None:
            send({"type": "error", "service": service, "message": message})

        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        )
        stt = SharedWhisperSTTService(
            whisper=self.whisper,
            settings=WhisperSTTService.Settings(model=config.whisper_model, language=Language.EN),
        )
        # The system instruction is the session's system prompt plus its summary.
        llm = MemoryLLMService(
            summary=session["summary"],
            base_url=f"{config.ollama_url}/v1",
            settings=OLLamaLLMService.Settings(
                model=model, system_instruction=session["system_prompt"]
            ),
        )
        tts = ChatterboxTTSService(
            base_url=config.chatterbox_url,
            voice=voice,
            client=upstreams.client,
            timeout=config.tts_timeout,
            expressions=session["expressions"],
            silent_actions=session["silent_actions"],
        )
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
                # Silence-based end of turn; the default smart-turn model needs torch.
                user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()]),
            ),
        )
        # Summarises with the call's current model, so no second model gets loaded.
        summarise = ollama_summariser(
            upstreams, lambda: llm.settings.model, config.summary_prompt, config.summary_timeout
        )
        memory = CallMemory(store, session, context, llm, summarise, config.memory_budget)

        async def save_failed() -> None:
            report(
                "storage",
                "Could not save the conversation; the call goes on, but the history "
                "may miss what was just said.",
            )

        TurnRecorder(
            store,
            call.session_id,
            lambda: llm.settings.model,
            after_reply=memory.request_fold,
            on_save_error=save_failed,
        ).attach(aggregators)

        output = transport.output()
        pipeline = Pipeline(
            [transport.input(), stt, aggregators.user(), llm, tts, output, aggregators.assistant()]
        )
        audio_in = InputAudioWatch()
        timing = TurnTimingObserver(
            {"stt": stt, "llm": llm, "tts": tts}, output, label=f" (session {call.session_id})"
        )
        # RTVI is added by the worker by default: transcripts and bot state go to the
        # page over the data channel. It also forwards error frames as raw RTVI
        # "error" messages; the page shows the server's own messages instead.
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=IN_SAMPLE_RATE,
                audio_out_sample_rate=OUT_SAMPLE_RATE,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[timing, audio_in],
            # A call ends when the client hangs up, not after a quiet spell.
            idle_timeout_secs=None,
        )
        ending = False

        async def end_call(reason: str, message: str | None = None, retry: bool = False) -> None:
            """End the call from the server side, telling the page why first."""
            nonlocal ending
            if ending:
                return
            ending = True
            logger.info(f"Call {pc_id}: ending ({reason})")
            send({"type": "call-ended", "reason": reason, "message": message, "retry": retry})
            await asyncio.sleep(END_MESSAGE_GRACE)  # let the message go out first
            await worker.cancel(reason=reason)

        async def update_settings(changes: dict[str, Any]) -> None:
            logger.info(f"Call {pc_id}: settings changed: {sorted(changes)}")
            for flag in ("expressions", "silent_actions"):
                if flag in changes:
                    setattr(tts, flag, changes[flag])
            frames = settings_frames(changes)
            if frames:
                await worker.queue_frames(frames)
            if "model" in changes:
                # Pipecat writes the LLM service off after an error such as a missing
                # model; a new model is a fresh start. Load it now rather than on the
                # next turn.
                await llm.set_usable(True)
                background(warm_up({"ollama": changes["model"]}))

        await call.attach(update_settings, end_call)

        async def warm_up(targets: dict[str, str]) -> None:
            """Warm the upstreams in ``targets`` (name -> model or voice), telling the
            page which are still pending and, at the end, which failed and why."""
            pending = set(targets)
            errors: dict[str, str] = {}
            send({"type": "warmup", "state": "warming", "pending": sorted(pending)})
            started = time.monotonic()

            async def one(name: str, what: str) -> None:
                error = await upstreams.warm_up_one(name, what)
                pending.discard(name)
                if error:
                    errors[name] = error
                if pending:
                    send({"type": "warmup", "state": "warming", "pending": sorted(pending)})

            await asyncio.gather(*(one(name, what) for name, what in targets.items()))
            result = {name: name not in errors for name in targets}
            logger.info(f"Warm-up finished in {time.monotonic() - started:.1f}s: {result}")
            send({"type": "warmup", "state": "done", **result, "errors": errors})

        connected = asyncio.Event()

        async def watch_connection() -> None:
            """End a call whose client never connects, or whose audio stops coming:
            WebRTC itself can take half a minute to notice a vanished peer."""
            try:
                await asyncio.wait_for(connected.wait(), CONNECT_TIMEOUT)
            except TimeoutError:
                logger.warning(f"Call {pc_id}: the client did not connect in {CONNECT_TIMEOUT:g}s")
                await worker.cancel(reason="client never connected")
                return
            audio_in.last = time.monotonic()
            while True:
                await asyncio.sleep(1.0)
                if time.monotonic() - audio_in.last > SILENCE_TIMEOUT:
                    logger.warning(
                        f"Call {pc_id}: no audio from the client for {SILENCE_TIMEOUT:g}s; "
                        "the connection is gone"
                    )
                    await worker.cancel(reason="connection lost")
                    return

        errors = ErrorReporter(
            llm, tts, report, lambda message: background(end_call("tts", message))
        )

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker: PipelineWorker, frame: ErrorFrame) -> None:
            errors.handle(frame)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client) -> None:
            connected.set()
            background(warm_up({"ollama": llm.settings.model, "chatterbox": tts.settings.voice}))
            # A session whose stored tail is over budget is folded now, in the
            # background; turns before it finishes use the full tail.
            memory.request_fold()

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client) -> None:
            logger.info(f"Call {pc_id}: client disconnected")
            await worker.cancel(reason="client disconnected")

        background(watch_connection())
        return worker, memory
