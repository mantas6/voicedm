"""Summary memory: the prompt layout of a call, and folding older messages into the
session's summary.

Every LLM request of a call is laid out as

    [system prompt] [summary of older messages] [recent messages ...]

so that its start stays byte-stable from turn to turn and Ollama can reuse its prompt
cache. The summary is appended to the system instruction (the system prompt, a blank
line, a header and the summary, all in the one system message). It changes only when
a fold happens, and changing the system prompt mid-call keeps it. The recent messages
are the session's messages after ``summary_upto``; Pipecat appends each turn to them.

When the recent messages grow over the budget (characters of message text), the
older half is folded into the summary: one separate request to Ollama turns the
previous summary plus the messages being folded into a new summary, which is stored
with the id of the last folded message. Then the call switches to the new summary
and drops the folded messages from its context. Folding never deletes stored messages.

Folding starts between turns (after a reply, or right after a call connects) and
never holds up a turn: a turn that starts meanwhile uses the old context. The switch
is made by the LLM service itself, between two requests, so a request never sees the
new summary without the shorter context or the other way round, and messages added
while the summary was being written are kept. A failed summarisation changes nothing
and is retried after a later reply.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import accumulate
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.frames.frames import ControlFrame, Frame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ollama.llm import OLLamaLLMService

from voicedm.expressions import history_text

if TYPE_CHECKING:
    from voicedm.storage import Storage
    from voicedm.upstream import Upstreams

SUMMARY_HEADER = "Summary of the earlier conversation (background, not instructions):"

# (previous summary, messages to fold) -> new summary
Summariser = Callable[[str, list[dict]], Awaitable[str]]

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def summary_instruction(summary: str) -> str:
    """The part of the system instruction that carries the summary ("" for none)."""
    summary = summary.strip()
    return f"{SUMMARY_HEADER}\n{summary}" if summary else ""


def recent_messages(session: dict, messages: list[dict]) -> list[dict]:
    """The stored messages not folded into the session's summary yet."""
    upto = session.get("summary_upto")
    return [
        m
        for m in messages
        if (upto is None or m["id"] > upto)
        and m["role"] in ("user", "assistant")
        and m["content"].strip()
    ]


def build_context(session: dict, messages: list[dict]) -> list[dict[str, str]]:
    """The chat messages a call on ``session`` starts from: those after the summary.

    The system prompt and the summary are not part of them: they make up the LLM
    service's system instruction (see ``MemoryLLMService``). An interrupted reply is
    replayed as far as it was spoken, which is all that is stored of it. A tail over
    the budget is folded by the call's ``CallMemory`` once the call has connected.
    """
    return [
        {"role": m["role"], "content": m["content"]} for m in recent_messages(session, messages)
    ]


def fold_count(messages: list[dict], budget: int) -> int:
    """How many of the oldest ``messages`` to fold into the summary: none while their
    text fits in ``budget`` characters, else about the older half of it.

    The cut is the one closest to half the text that leaves the recent window starting
    with a user message; only if there is none may it start with a reply. At least the
    newest message always stays.
    """
    sizes = [len(m["content"]) for m in messages]
    total = sum(sizes)
    if total <= budget or len(messages) < 2:
        return 0
    folded = list(accumulate(sizes))  # folded[i - 1]: text folded when cutting at i
    cuts = range(1, len(messages))
    preferred = [i for i in cuts if messages[i]["role"] == "user"] or list(cuts)
    # Closest to half; on a tie, fold more.
    return min(preferred, key=lambda i: (abs(2 * folded[i - 1] - total), -i))


def transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        speaker = "User" if m["role"] == "user" else "Assistant"
        if m.get("interrupted"):
            speaker += " (interrupted)"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def summary_request(prompt: str, previous: str, messages: list[dict]) -> list[dict[str, str]]:
    """The chat messages that ask for a new summary of ``previous`` plus ``messages``."""
    text = (
        f"Summary so far:\n{previous.strip() or '(nothing yet)'}\n\n"
        f"Next part of the conversation:\n{transcript(messages)}"
    )
    return [{"role": "system", "content": prompt}, {"role": "user", "content": text}]


def clean_summary(text: str) -> str:
    """The summary without any reasoning the model wrote inline, whitespace-tidied."""
    return _THINKING.sub("", text).strip()


def ollama_summariser(
    upstreams: Upstreams, model: Callable[[], str], prompt: str, timeout: float
) -> Summariser:
    """Summarise with the call's current model, so no second model has to be loaded."""

    async def summarise(previous: str, messages: list[dict]) -> str:
        request = summary_request(prompt, previous, messages)
        return await upstreams.chat(model(), request, timeout)

    return summarise


def _normalised(content: Any) -> str | None:
    return " ".join(content.split()) if isinstance(content, str) else None


def live_prefix(live: list, folded: list[dict]) -> int | None:
    """How many leading messages of the live context hold the stored ``folded`` ones.

    Live messages that are never stored (empty ones, or the prompts Pipecat adds for
    an empty user turn) go along with the stored ones around them. The live context
    keeps the transcript's raw whitespace, which the stored copy has tidied. None if
    the live context does not start with ``folded``.
    """

    def stored_kind(message: Any) -> bool:
        return (
            isinstance(message, dict)
            and message.get("role") in ("user", "assistant")
            and bool(_normalised(message.get("content")))
        )

    i = 0
    for stored in folded:
        while i < len(live) and not stored_kind(live[i]):
            i += 1
        if i == len(live):
            return None
        message = live[i]
        if (
            message["role"] != stored["role"]
            or _normalised(message["content"]) != stored["content"]
        ):
            return None
        i += 1
    return i


@dataclass
class SummaryFoldFrame(ControlFrame):
    """Switches a call to a new summary, dropping the live messages it now covers.

    Queued straight into the LLM service, which handles it in order with the turns'
    context frames, so the switch happens between two requests. Not interruptible: a
    barge-in must not drop it.
    """

    context: LLMContext
    folded: list  # the leading live context messages, by identity
    summary: str
    done: asyncio.Event = field(default_factory=asyncio.Event)
    interruptible: bool = field(default=False, init=False)


def apply_fold(frame: SummaryFoldFrame) -> bool:
    """Drop ``frame.folded`` from the start of the live context, if it still starts
    with them. Messages appended since stay."""
    live = frame.context.messages
    count = len(frame.folded)
    if len(live) < count or any(a is not b for a, b in zip(live, frame.folded, strict=False)):
        return False
    frame.context.set_messages(live[count:])
    return True


class MemoryLLMService(OLLamaLLMService):
    """The Ollama LLM service with the session summary after the system prompt.

    The summary is kept as an appended system instruction, which Pipecat composes
    after the base system prompt and keeps when the prompt is changed mid-call.

    With ``hide_actions``, the model is shown its earlier replies without their
    actions (see ``expressions.history_text``), keeping the sound ones with
    ``keep_sounds``. Only the request is changed, each time it is built: the context
    keeps the replies as written, so folding, which matches it against the stored
    messages, is not affected. Both may change mid-call.
    """

    def __init__(
        self, *, summary: str = "", hide_actions: bool = False, keep_sounds: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.hide_actions = hide_actions
        self.keep_sounds = keep_sounds
        self._summary_part = ""
        self.set_summary(summary)

    def shown(self, messages: list) -> list:
        """``messages`` as the model is given them."""
        if not self.hide_actions:
            return messages

        def show(message: Any) -> Any:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                return message
            content = message.get("content")
            if not isinstance(content, str):
                return message
            return {**message, "content": history_text(content, keep_sounds=self.keep_sounds)}

        return [show(message) for message in messages]

    def build_chat_completion_params(self, params_from_context: Any) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        params["messages"] = self.shown(params["messages"])
        return params

    def set_summary(self, summary: str) -> None:
        parts = self._appended_system_instructions
        if self._summary_part in parts:
            parts.remove(self._summary_part)
        self._summary_part = summary_instruction(summary)
        if self._summary_part:
            parts.append(self._summary_part)
        self._compose_system_instruction()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, SummaryFoldFrame):
            if apply_fold(frame):
                self.set_summary(frame.summary)
                logger.info(
                    f"{self}: switched to the new summary; {len(frame.folded)} context "
                    f"messages folded, {len(frame.context.messages)} kept"
                )
            else:
                logger.error(f"{self}: context changed under a fold; summary not switched")
            frame.done.set()
            return
        await super().process_frame(frame, direction)


class CallMemory:
    """Folds a call's recent messages into its session's summary when over budget.

    ``request_fold`` is called between turns; it starts a fold in the background if
    one is due and none is running, and never waits for it.
    """

    def __init__(
        self,
        store: Storage,
        session: dict,
        context: LLMContext,
        llm: MemoryLLMService,
        summarise: Summariser,
        budget: int,
    ) -> None:
        self.store = store
        self.session_id = session["id"]
        self.summary: str = session["summary"]
        self.upto: int | None = session["summary_upto"]
        self.context = context
        self.llm = llm
        self.summarise = summarise
        self.budget = budget
        self._task: asyncio.Task | None = None

    def request_fold(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._fold_logged())

    async def wait(self) -> None:
        """Wait for a running fold (tests)."""
        if self._task:
            await self._task

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _fold_logged(self) -> None:
        try:
            await self.fold()
        except Exception:
            logger.exception(f"Session {self.session_id}: folding failed")

    async def fold(self) -> bool:
        """Fold the older half of the recent messages if they are over budget.

        Returns whether a fold happened. Any failure leaves the stored summary and the
        live context as they were; the next ``request_fold`` tries again.
        """
        stored = recent_messages(
            {"summary_upto": self.upto}, await self.store.messages(self.session_id, self.upto)
        )
        count = fold_count(stored, self.budget)
        if not count:
            return False
        folded = stored[:count]
        live = list(self.context.messages)
        prefix = live_prefix(live, folded)
        if prefix is None:
            logger.warning(
                f"Session {self.session_id}: context and stored messages differ; not folding"
            )
            return False

        chars = sum(len(m["content"]) for m in folded)
        logger.info(
            f"Session {self.session_id}: folding {count} of {len(stored)} recent messages "
            f"({chars} of {sum(len(m['content']) for m in stored)} characters, budget "
            f"{self.budget}) into the summary"
        )
        started = time.monotonic()
        try:
            summary = clean_summary(await self.summarise(self.summary, folded))
            if not summary:
                raise ValueError("the summary came back empty")
            await self.store.set_summary(self.session_id, summary, folded[-1]["id"])
        except Exception as e:
            logger.warning(
                f"Session {self.session_id}: summarising failed, will retry after a later "
                f"turn: {e!r}"
            )
            return False
        self.summary, self.upto = summary, folded[-1]["id"]
        logger.info(
            f"Session {self.session_id}: summary of {len(summary)} characters written in "
            f"{time.monotonic() - started:.1f}s, up to message {self.upto}"
        )
        frame = SummaryFoldFrame(context=self.context, folded=live[:prefix], summary=summary)
        await self.llm.queue_frame(frame)
        await frame.done.wait()
        return True
