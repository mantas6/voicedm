import os

import httpx
import pytest
from fakes import FakeBot
from fastapi.testclient import TestClient

from voicedm import app as app_module
from voicedm.config import Config

CONFIG = Config(ollama_url="http://ollama.test", chatterbox_url="http://tts.test")


def client() -> TestClient:
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    return TestClient(app_module.create_app(CONFIG, transport=transport, bot=FakeBot()))


@pytest.mark.parametrize("path", ["/", "/app.js", "/style.css"])
def test_page_files_are_checked_on_every_load(path):
    with client() as c:
        response = c.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["etag"]
    # A modification time would let a browser guess how long its copy stays fresh.
    assert "last-modified" not in response.headers


def test_unchanged_file_is_not_sent_again():
    with client() as c:
        etag = c.get("/app.js").headers["etag"]
        again = c.get("/app.js", headers={"if-none-match": etag})
    assert again.status_code == 304
    assert again.headers["etag"] == etag
    assert again.headers["cache-control"] == "no-cache"
    assert again.content == b""


def test_new_version_with_same_time_and_size_is_sent(tmp_path, monkeypatch):
    """Installed files can share one modification time across versions."""
    static = tmp_path / "static"
    static.mkdir()
    script = static / "app.js"
    monkeypatch.setattr(app_module, "STATIC_DIR", static)

    def install(content: str) -> None:
        script.write_text(content)
        os.utime(script, (1, 1))

    install("let version = 1;")
    with client() as c:
        old = c.get("/app.js").headers["etag"]
        install("let version = 2;")
        response = c.get("/app.js", headers={"if-none-match": old})
    assert response.status_code == 200
    assert response.text == "let version = 2;"
    assert response.headers["etag"] != old
