"""Summary memory: fold boundaries, the starting context, and folding against a real
store with a fake LLM service and summariser."""

import asyncio

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from voicedm.memory import (
    SUMMARY_HEADER,
    CallMemory,
    MemoryLLMService,
    SummaryFoldFrame,
    apply_fold,
    build_context,
    clean_summary,
    fold_count,
    live_prefix,
    summary_instruction,
    summary_request,
)
from voicedm.storage import Storage
from voicedm.upstream import UpstreamError


def msgs(*specs: tuple[str, int]) -> list[dict]:
    """Messages with the given roles ("u"/"a") and content lengths."""
    roles = {"u": "user", "a": "assistant"}
    return [
        {"id": i + 1, "role": roles[r], "content": "x" * size, "interrupted": False}
        for i, (r, size) in enumerate(specs)
    ]


# -- where to fold --


def test_no_fold_at_or_under_budget():
    history = msgs(("u", 10), ("a", 10), ("u", 10), ("a", 10))
    assert fold_count(history, 40) == 0
    assert fold_count(history, 39) == 2  # one character over: the older half goes
    assert fold_count([], 1) == 0


def test_folds_the_older_half():
    history = msgs(*[("u", 10), ("a", 10)] * 4)
    assert fold_count(history, 50) == 4


def test_recent_window_starts_with_a_user_message_when_possible():
    # Cutting at 3 (45 of 100 characters) is closest to half but would start the
    # recent window with a reply; cutting at 2 (40) starts it with the user.
    history = msgs(("u", 20), ("a", 20), ("u", 5), ("a", 35), ("u", 20))
    assert fold_count(history, 60) == 2


def test_reply_may_start_the_window_when_nothing_else_fits():
    assert fold_count(msgs(("u", 10), ("a", 100)), 50) == 1


def test_newest_message_always_stays():
    assert fold_count(msgs(("u", 500)), 10) == 0
    assert fold_count(msgs(("u", 5), ("a", 5), ("u", 500)), 10) == 2


def test_adjacent_user_messages_can_be_split():
    # A reply that was never spoken leaves two user messages in a row.
    history = msgs(("u", 20), ("u", 20), ("a", 5), ("u", 10))
    assert fold_count(history, 50) == 1


# -- the prompt layout --


def test_build_context_starts_after_the_summary():
    history = msgs(("u", 3), ("a", 3), ("u", 3), ("a", 3))
    history[3]["content"] = "   "  # never replayed
    assert build_context({"summary_upto": None}, []) == []
    assert len(build_context({"summary_upto": None}, history)) == 3
    assert build_context({"summary_upto": 1}, history) == [
        {"role": "assistant", "content": "xxx"},
        {"role": "user", "content": "xxx"},
    ]


def test_summary_goes_after_the_system_prompt():
    assert summary_instruction("  ") == ""
    llm = MemoryLLMService(
        summary="They like tea.",
        base_url="http://ollama.test/v1",
        settings=MemoryLLMService.Settings(model="m", system_instruction="Be brief."),
    )
    assert llm.settings.system_instruction == f"Be brief.\n\n{SUMMARY_HEADER}\nThey like tea."
    llm.set_summary("They like coffee now.")
    assert llm.settings.system_instruction.endswith("\nThey like coffee now.")
    assert llm.settings.system_instruction.count(SUMMARY_HEADER) == 1
    llm.set_summary("")
    assert llm.settings.system_instruction == "Be brief."


def test_summary_request_marks_interrupted_replies():
    history = [
        {"role": "user", "content": "Tell me a story", "interrupted": False},
        {"role": "assistant", "content": "Once upon a time.", "interrupted": True},
        {"role": "user", "content": "Stop", "interrupted": False},
    ]
    system, user = summary_request("Summarise.", "", history)
    assert system == {"role": "system", "content": "Summarise."}
    assert user["content"] == (
        "Summary so far:\n(nothing yet)\n\nNext part of the conversation:\n"
        "User: Tell me a story\nAssistant (interrupted): Once upon a time.\nUser: Stop"
    )
    assert "Summary so far:\nEarlier." in summary_request("p", "Earlier.", history)[1]["content"]


def test_clean_summary_drops_inline_reasoning():
    assert clean_summary("<think>hmm\nok</think>\n The summary. ") == "The summary."
    assert clean_summary("<think>only thinking</think>") == ""


def test_live_prefix_matches_raw_whitespace_and_skips_unstored_messages():
    folded = [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi. How are you?"},
    ]
    live = [
        {"role": "user", "content": "Hello  there "},
        {"role": "developer", "content": "The user interrupted; respond."},
        {"role": "assistant", "content": "Hi.  How are you?"},
        {"role": "user", "content": "Fine"},
    ]
    assert live_prefix(live, folded) == 3
    assert live_prefix(live, folded[:1]) == 1
    assert live_prefix(live[:1], folded) is None
    assert live_prefix(live[2:], folded) is None


def test_apply_fold_keeps_messages_added_since():
    a, b, c, d = ({"role": "user", "content": x} for x in "abcd")
    context = LLMContext(messages=[a, b, c])
    frame = SummaryFoldFrame(context=context, folded=context.messages[:2], summary="S")
    context.add_message(d)  # a turn finished while the summary was written
    assert apply_fold(frame)
    assert context.messages == [c, d]
    # A context that no longer starts with the folded messages is left alone.
    assert not apply_fold(frame)
    assert context.messages == [c, d]


# -- folding against a store --


class FakeLLM:
    """Applies fold frames at once, like the LLM service between two requests."""

    def __init__(self) -> None:
        self.summaries: list[str] = []

    async def queue_frame(self, frame) -> None:
        if apply_fold(frame):
            self.summaries.append(frame.summary)
        frame.done.set()


class Summariser:
    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, list[tuple[str, str, bool]]]] = []

    async def __call__(self, previous: str, messages: list[dict]) -> str:
        self.calls.append(
            (previous, [(m["role"], m["content"], m["interrupted"]) for m in messages])
        )
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
async def store(tmp_path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


async def add(store: Storage, sid: int, role: str, text: str, interrupted: bool = False) -> dict:
    model = "m" if role == "assistant" else None
    return await store.add_message(sid, role, text, model=model, interrupted=interrupted)


async def setup_memory(store, summarise, budget, exchanges=2):
    session = await store.create_session("m", "v", "p")
    sid = session["id"]
    for i in range(exchanges):
        await add(store, sid, "user", f"question {i}")
        await add(store, sid, "assistant", f"answer {i}.")
    session = await store.get_session(sid)
    context = LLMContext(messages=build_context(session, await store.messages(sid)))
    llm = FakeLLM()
    return CallMemory(store, session, context, llm, summarise, budget), context, llm


async def test_under_budget_does_not_summarise(store):
    summarise = Summariser()
    memory, context, llm = await setup_memory(store, summarise, budget=38)  # 38 characters
    assert not await memory.fold()
    assert summarise.calls == []


async def test_fold_stores_summary_and_trims_the_context(store):
    summarise = Summariser("Asked two things.")
    memory, context, llm = await setup_memory(store, summarise, budget=37)
    assert await memory.fold()
    stored = await store.messages(memory.session_id)
    session = await store.get_session(memory.session_id)
    assert (session["summary"], session["summary_upto"]) == ("Asked two things.", stored[1]["id"])
    assert summarise.calls == [
        ("", [("user", "question 0", False), ("assistant", "answer 0.", False)])
    ]
    assert [m["content"] for m in context.messages] == ["question 1", "answer 1."]
    assert llm.summaries == ["Asked two things."]
    assert len(stored) == 4  # the folded messages stay in the database
    # The context of the next call is the same as the trimmed one.
    assert build_context(session, stored) == context.messages


async def test_repeated_folds_accumulate_into_one_summary(store):
    summarise = Summariser("S1", "S2")
    memory, context, llm = await setup_memory(store, summarise, budget=37)
    assert await memory.fold()
    sid = memory.session_id
    await add(store, sid, "user", "question 2")
    context.add_message({"role": "user", "content": "question 2"})
    assert not await memory.fold()  # 29 characters: within budget
    await add(store, sid, "assistant", "answer 2.")
    context.add_message({"role": "assistant", "content": "answer 2."})
    assert await memory.fold()
    stored = await store.messages(sid)
    session = await store.get_session(sid)
    assert summarise.calls[1][0] == "S1"  # the previous summary goes into the next one
    assert session["summary"] == "S2"
    assert session["summary_upto"] == stored[3]["id"]
    assert [m["content"] for m in context.messages] == ["question 2", "answer 2."]


@pytest.mark.parametrize(
    "failure", [UpstreamError("Ollama is down"), TimeoutError(), "", "<think>…</think>"]
)
async def test_summariser_failure_leaves_everything_as_it_was(store, failure):
    summarise = Summariser(failure, "Recovered.")
    memory, context, llm = await setup_memory(store, summarise, budget=37)
    before = list(context.messages)
    assert not await memory.fold()
    session = await store.get_session(memory.session_id)
    assert (session["summary"], session["summary_upto"]) == ("", None)
    assert context.messages == before
    assert llm.summaries == []
    # The next try (after a later turn) succeeds.
    assert await memory.fold()
    assert (await store.get_session(memory.session_id))["summary"] == "Recovered."


async def test_interrupted_and_adjacent_messages_are_folded_as_stored(store):
    summarise = Summariser("S")
    session = await store.create_session("m", "v", "p")
    sid = session["id"]
    await add(store, sid, "user", "Tell me a long story")
    await add(store, sid, "assistant", "Once upon a time.", interrupted=True)
    await add(store, sid, "user", "Stop")
    await add(store, sid, "user", "Actually, go on")  # the reply to "Stop" was never spoken
    await add(store, sid, "assistant", "The end.")
    session = await store.get_session(sid)
    context = LLMContext(messages=build_context(session, await store.messages(sid)))
    memory = CallMemory(store, session, context, FakeLLM(), summarise, budget=40)
    assert await memory.fold()
    assert summarise.calls[0][1] == [
        ("user", "Tell me a long story", False),
        ("assistant", "Once upon a time.", True),
    ]
    assert [m["content"] for m in context.messages] == ["Stop", "Actually, go on", "The end."]


async def test_only_one_fold_runs_at_a_time(store):
    release = asyncio.Event()

    async def summarise(previous, messages):
        calls.append(previous)
        await release.wait()
        return "S"

    calls: list[str] = []
    memory, context, llm = await setup_memory(store, summarise, budget=37)
    memory.request_fold()
    await asyncio.sleep(0.05)
    memory.request_fold()  # a turn ends while the first fold is still running
    await asyncio.sleep(0.05)
    release.set()
    await memory.wait()
    assert calls == [""]
    await memory.close()


async def test_closing_cancels_a_running_fold(store):
    async def summarise(previous, messages):
        await asyncio.Event().wait()

    memory, context, llm = await setup_memory(store, summarise, budget=37)
    memory.request_fold()
    await asyncio.sleep(0.05)
    await memory.close()
    assert (await store.get_session(memory.session_id))["summary"] == ""
