"""The Pipecat voice pipeline, one per call."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
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
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
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
from voicedm.tts_chatterbox import ChatterboxTTSService
from voicedm.upstream import Upstreams

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

IN_SAMPLE_RATE = 16000  # what Silero and Whisper read
OUT_SAMPLE_RATE = 24000  # what Chatterbox produces

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly voice assistant in a spoken conversation. Your replies are "
    "read aloud, so answer in plain conversational sentences: no markdown, lists, "
    "emoji or code. Keep answers short unless asked for detail."
)


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
        model: str,
        voice: str,
    ) -> None:
        """Run the pipeline for one call until the client disconnects."""
        if self.whisper is None:
            raise RuntimeError("Bot.load() has not run")
        config = self.config
        logger.info(f"Call {connection.pc_id}: model {model}, voice {voice}")

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
                model=model, system_instruction=DEFAULT_SYSTEM_PROMPT
            ),
        )
        tts = ChatterboxTTSService(
            base_url=config.chatterbox_url,
            voice=voice,
            client=upstreams.client,
            timeout=config.tts_timeout,
        )
        context = LLMContext()
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
                # Silence-based end of turn; the default smart-turn model needs torch.
                user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()]),
            ),
        )

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

        warmup: asyncio.Task | None = None

        async def warm_up() -> None:
            await worker.rtvi.send_server_message({"type": "warmup", "state": "warming"})
            started = time.monotonic()
            result = await upstreams.warm_up(model, voice)
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
            if warmup and not warmup.done():
                warmup.cancel()
            await connection.disconnect()
            logger.info(f"Call {connection.pc_id} ended")
