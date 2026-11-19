"""The Pipecat voice pipeline, one per call."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.processors.frame_processor import FrameDirection
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
from voicedm.memory import build_context
from voicedm.storage import Storage
from voicedm.tts_chatterbox import ChatterboxTTSService
from voicedm.upstream import Upstreams

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

    from voicedm.app import ActiveCall

IN_SAMPLE_RATE = 16000  # what Silero and Whisper read
OUT_SAMPLE_RATE = 24000  # what Chatterbox produces


class SharedWhisperSTTService(WhisperSTTService):
    """``WhisperSTTService`` that uses an already loaded model instead of loading one."""

    def __init__(self, *, whisper: WhisperModel, **kwargs) -> None:
        self._shared_model = whisper
        super().__init__(**kwargs)

    def _load(self) -> None:
        self._model = self._shared_model


class TurnTimingObserver(BaseObserver):
    """Logs when each step of a turn happens, relative to the end of the user's speech.

    Shows where latency goes and whether speech starts before the LLM has finished.
    """

    # Each logged once per reply, the first time it happens.
    EVENTS: dict[type, str] = {
        LLMTextFrame: "llm first token",
        TTSAudioRawFrame: "tts first audio",
        BotStartedSpeakingFrame: "bot started speaking",
        LLMFullResponseEndFrame: "llm end",
        BotStoppedSpeakingFrame: "bot stopped speaking",
    }

    def __init__(self) -> None:
        super().__init__()
        self._turn = 0
        self._speech_end: float | None = None
        self._seen: set[str] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        # Broadcast frames travel both ways as two frames; count the downstream one.
        if not data.first_push or data.direction != FrameDirection.DOWNSTREAM:
            return
        frame, now = data.frame, time.monotonic()
        if isinstance(frame, UserStartedSpeakingFrame):
            self._turn += 1
            self._seen.clear()
            self._speech_end = None
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speech_end = now
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # The reply starts here; anything earlier is the tail of an interrupted one.
            self._seen.clear()
            self._log("user turn end", now)
        elif isinstance(frame, InterruptionFrame):
            self._log("interruption", now)
        elif isinstance(frame, TranscriptionFrame):
            self._log(f"transcript {frame.text.strip()!r}", now)
        else:
            for cls, name in self.EVENTS.items():
                if isinstance(frame, cls) and name not in self._seen:
                    self._seen.add(name)
                    self._log(name, now)

    def _log(self, event: str, now: float) -> None:
        offset = f"+{now - self._speech_end:.2f}s" if self._speech_end else "-"
        logger.info(f"turn {self._turn}: {event} at {offset} after end of speech")


class TurnRecorder:
    """Saves each finished turn of a call as a message of the call's session.

    Hooks the context aggregators' turn events: the user aggregator reports the text
    it wrote to the context, the assistant aggregator the reply as far as it was
    spoken (it sits after the output transport, which releases text as its audio
    plays) and whether it was interrupted. Event handlers run as separate tasks, so
    a lock keeps the writes in the order the turns finished.
    """

    def __init__(self, store: Storage, session_id: int, llm_model: Callable[[], str]) -> None:
        self.store = store
        self.session_id = session_id
        self._llm_model = llm_model
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
                logger.exception(f"Could not save a {role} message")


def settings_frames(changes: dict[str, str]) -> list[Frame]:
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
        """Run the pipeline for one call on ``call.session_id`` until the client leaves."""
        if self.whisper is None:
            raise RuntimeError("Bot.load() has not run")
        config = self.config
        session = await store.get_session(call.session_id)
        if session is None:
            raise RuntimeError(f"session {call.session_id} does not exist")
        history = await store.messages(call.session_id)
        model, voice = session["model"], session["voice"]
        logger.info(
            f"Call {connection.pc_id} on session {call.session_id}: model {model}, "
            f"voice {voice}, {len(history)} earlier messages"
        )

        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        )
        stt = SharedWhisperSTTService(
            whisper=self.whisper,
            settings=WhisperSTTService.Settings(model=config.whisper_model, language=Language.EN),
        )
        llm = OLLamaLLMService(
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
        )
        context = LLMContext(messages=build_context(session, history))
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
                # Silence-based end of turn; the default smart-turn model needs torch.
                user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()]),
            ),
        )
        TurnRecorder(store, call.session_id, lambda: llm.settings.model).attach(aggregators)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                aggregators.user(),
                llm,
                tts,
                transport.output(),
                aggregators.assistant(),
            ]
        )
        # RTVI is added by the worker by default: transcripts and bot state go to the
        # page over the data channel.
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=IN_SAMPLE_RATE,
                audio_out_sample_rate=OUT_SAMPLE_RATE,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[MetricsLogObserver(), TurnTimingObserver()],
            # A call ends when the client hangs up, not after a quiet spell.
            idle_timeout_secs=None,
        )

        async def update_settings(changes: dict[str, str]) -> None:
            frames = settings_frames(changes)
            if frames:
                logger.info(f"Call {connection.pc_id}: settings changed: {sorted(changes)}")
                await worker.queue_frames(frames)

        await call.attach(update_settings)
        warmup: asyncio.Task | None = None

        async def warm_up() -> None:
            await worker.rtvi.send_server_message({"type": "warmup", "state": "warming"})
            started = time.monotonic()
            result = await upstreams.warm_up(llm.settings.model, tts.settings.voice)
            logger.info(f"Warm-up finished in {time.monotonic() - started:.1f}s: {result}")
            await worker.rtvi.send_server_message({"type": "warmup", "state": "done", **result})

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client) -> None:
            nonlocal warmup
            warmup = asyncio.create_task(warm_up())

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client) -> None:
            logger.info(f"Call {connection.pc_id}: client disconnected")
            await worker.cancel()

        try:
            runner = WorkerRunner(handle_sigint=False)
            await runner.add_workers(worker)
            await runner.run()
        finally:
            call.detach()
            if warmup and not warmup.done():
                warmup.cancel()
            await connection.disconnect()
            logger.info(f"Call {connection.pc_id} ended")
