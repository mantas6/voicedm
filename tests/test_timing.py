"""The per-turn latency line: normal turns, interruptions, duplicated frames."""

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voicedm.bot import TurnTimingObserver

DOWN, UP = FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM


class Harness:
    def __init__(self) -> None:
        self.now = 0.0
        self.stt, self.llm, self.tts = FrameProcessor(), FrameProcessor(), FrameProcessor()
        self.output, self.sink = FrameProcessor(), FrameProcessor()
        services = {"stt": self.stt, "llm": self.llm, "tts": self.tts}
        self.observer = TurnTimingObserver(services, self.output, clock=lambda: self.now)

    async def push(self, at: float, frame, source=None, direction=DOWN, first=True) -> None:
        self.now = at
        data = FramePushed(source or self.stt, self.sink, frame, direction, 0, first)
        await self.observer.on_push_frame(data)

    async def broadcast(self, at: float, frame_type) -> None:
        """A frame pushed both ways, as Pipecat does: two frames, both first pushes."""
        await self.push(at, frame_type(), direction=UP)
        await self.push(at, frame_type())

    async def ttfb(self, at: float, service: FrameProcessor, value: float) -> None:
        metric = TTFBMetricsData(processor=service.name, value=value)
        await self.push(at, MetricsFrame(data=[metric]), source=service)

    async def user_turn(self, start: float, stop: float, end: float) -> None:
        await self.broadcast(start, UserStartedSpeakingFrame)
        await self.push(stop, VADUserStoppedSpeakingFrame())
        await self.push(stop + 0.3, TranscriptionFrame("hello", "user", "now"))
        await self.ttfb(stop + 0.3, self.stt, 0.25)
        await self.broadcast(end, UserStoppedSpeakingFrame)

    async def reply_end(self, at: float) -> None:
        """The reply's end frame leaves the LLM, then the TTS, then the output once
        everything before it has been played."""
        await self.push(at, LLMFullResponseEndFrame(), source=self.llm)
        frame = LLMFullResponseEndFrame()
        await self.push(at, frame, source=self.llm)
        await self.push(at, frame, source=self.tts, first=False)
        await self.push(at + 2.0, frame, source=self.output, first=False)


async def test_one_line_per_turn_with_times_after_end_of_speech():
    h = Harness()
    await h.user_turn(start=10.0, stop=11.0, end=11.6)
    await h.ttfb(11.7, h.llm, 0.1)
    await h.push(12.5, LLMTextFrame("Hi"), source=h.llm)
    await h.push(12.6, LLMTextFrame(" there."), source=h.llm)
    await h.ttfb(13.0, h.tts, 0.5)
    await h.broadcast(13.1, BotStartedSpeakingFrame)
    await h.ttfb(13.5, h.tts, 0.9)  # the second sentence's: not the turn's
    await h.reply_end(14.0)
    [turn] = h.observer.turns
    assert turn == pytest.approx(
        {
            "turn": 1,
            "transcript": 0.3,
            "turn_end": 0.6,
            "first_text": 1.5,
            "first_audio": 2.1,
            "llm_end": 3.0,
            "ttfb_stt": 0.25,
            "ttfb_llm": 0.1,
            "ttfb_tts": 0.5,
            "interrupted": False,
        }
    )


async def test_interrupted_turn_does_not_leak_into_the_next():
    h = Harness()
    await h.user_turn(start=0.0, stop=1.0, end=1.5)
    await h.push(2.0, LLMTextFrame("A long story"), source=h.llm)
    await h.broadcast(2.5, BotStartedSpeakingFrame)
    # The user talks over the reply...
    await h.broadcast(4.0, UserStartedSpeakingFrame)
    # ...and late reports of the interrupted reply arrive while they speak,
    await h.ttfb(4.2, h.tts, 9.9)
    await h.push(4.3, LLMTextFrame(" that goes on"), source=h.llm)
    await h.push(4.4, LLMFullResponseEndFrame(), source=h.llm)  # (dropped at the output)
    await h.push(5.0, VADUserStoppedSpeakingFrame())
    await h.push(5.3, TranscriptionFrame("stop", "user", "now"))
    await h.broadcast(5.5, UserStoppedSpeakingFrame)
    # The second reply.
    await h.ttfb(5.6, h.llm, 0.2)
    await h.push(6.0, LLMTextFrame("Sure."), source=h.llm)
    await h.ttfb(6.1, h.tts, 0.4)
    await h.broadcast(6.6, BotStartedSpeakingFrame)
    await h.reply_end(7.0)
    first, second = h.observer.turns
    assert first["interrupted"] is True
    assert first["first_audio"] == pytest.approx(1.5)
    assert second["interrupted"] is False
    assert second["first_text"] == pytest.approx(1.0)
    assert second["first_audio"] == pytest.approx(1.6)
    assert (second["ttfb_llm"], second["ttfb_tts"]) == (0.2, 0.4)


async def test_turn_without_reply_and_hang_up_mid_reply():
    h = Harness()
    # The user's speech ends no turn (e.g. a cough), then another turn starts.
    await h.broadcast(0.0, UserStartedSpeakingFrame)
    await h.push(0.5, VADUserStoppedSpeakingFrame())
    await h.user_turn(start=1.0, stop=2.0, end=2.5)
    await h.push(3.0, LLMTextFrame("Well"), source=h.llm)
    await h.push(3.5, CancelFrame())  # hang-up
    [only] = h.observer.turns
    assert only["turn"] == 2
    assert only["interrupted"] is True
    assert only["first_audio"] is None
