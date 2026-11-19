"""Error frames from the pipeline become messages for the page."""

import httpx
import openai
from pipecat.frames.frames import ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ollama.llm import OLLamaLLMService

from voicedm.bot import MAX_TTS_FAILURES, TTS_GAVE_UP, ErrorReporter, explain_llm_error
from voicedm.tts_chatterbox import ChatterboxTTSService

REQUEST = httpx.Request("POST", "http://ollama.test/v1/chat/completions")


def services():
    llm = OLLamaLLMService(
        base_url="http://ollama.test/v1", settings=OLLamaLLMService.Settings(model="small:1b")
    )
    client = httpx.AsyncClient()
    tts = ChatterboxTTSService(base_url="http://tts.test", voice="Alpha.wav", client=client)
    return llm, tts


def reporter():
    llm, tts = services()
    reports, gave_up = [], []
    errors = ErrorReporter(llm, tts, lambda s, m: reports.append((s, m)), gave_up.append)
    return errors, llm, tts, reports, gave_up


def llm_error(exception: Exception) -> ErrorFrame:
    return ErrorFrame(error=f"Error during completion: {exception}", exception=exception)


def test_llm_errors_name_the_problem():
    unreachable = openai.APIConnectionError(request=REQUEST)
    missing = openai.NotFoundError(
        "model 'small:1b' not found",
        response=httpx.Response(404, request=REQUEST),
        body=None,
    )
    failing = openai.InternalServerError(
        "out of memory", response=httpx.Response(500, request=REQUEST), body=None
    )
    timeout = openai.APITimeoutError(request=REQUEST)
    assert explain_llm_error(llm_error(unreachable), "small:1b") == (
        "Cannot reach the language model server (Ollama)."
    )
    assert "model “small:1b” is not available" in explain_llm_error(llm_error(missing), "small:1b")
    message = explain_llm_error(llm_error(failing), "small:1b")
    assert "HTTP 500" in message and "out of memory" in message
    assert "did not answer in time" in explain_llm_error(llm_error(timeout), "small:1b")
    assert explain_llm_error(ErrorFrame(error="odd"), "m") == "The language model failed: odd"


async def test_reporter_forwards_each_service_error():
    errors, llm, tts, reports, gave_up = reporter()
    errors.handle(ErrorFrame(error="Cannot reach the voice server (Chatterbox).", processor=tts))
    frame = llm_error(openai.APIConnectionError(request=REQUEST))
    frame.processor = llm
    errors.handle(frame)
    errors.handle(ErrorFrame(error="boom", processor=FrameProcessor()))
    assert reports == [
        ("tts", "Cannot reach the voice server (Chatterbox)."),
        ("llm", "Cannot reach the language model server (Ollama)."),
        ("other", "Something went wrong: boom"),
    ]
    assert gave_up == []


async def test_reporter_ends_the_call_once_speech_is_written_off():
    errors, llm, tts, reports, gave_up = reporter()
    # Pipecat's own note about one silent reply is not repeated to the page...
    errors.handle(ErrorFrame(error="TTS context x completed with no audio", processor=tts))
    assert (reports, gave_up) == ([], [])
    # ...but once it has written the service off, the call is ended, once.
    await tts.set_usable(False)
    for _ in range(2):
        errors.handle(
            ErrorFrame(error="3 consecutive TTS contexts completed with no audio", processor=tts)
        )
    assert gave_up == [TTS_GAVE_UP]
    assert reports == []


async def test_reporter_ends_the_call_after_failed_sentences_in_a_row():
    errors, llm, tts, reports, gave_up = reporter()
    for count in range(1, MAX_TTS_FAILURES + 2):
        tts.failures_in_a_row = count
        errors.handle(
            ErrorFrame(error="Cannot reach the voice server (Chatterbox).", processor=tts)
        )
    assert len(reports) == MAX_TTS_FAILURES + 1
    assert gave_up == [TTS_GAVE_UP]
