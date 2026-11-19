"""Entry point: ``voicedm`` or ``python -m voicedm``."""

from __future__ import annotations

import logging
import sys

import uvicorn
from loguru import logger

from voicedm.app import create_app
from voicedm.config import Config

# Seconds uvicorn waits for open HTTP requests on shutdown before closing them.
GRACEFUL_SHUTDOWN = 5


def setup_logging(level: str) -> None:
    """One plain line per message on stderr; the journal adds time and host.

    Pipecat logs through loguru, whose default is DEBUG: every frame-level detail,
    including what was said. The service logs at ``level`` (INFO by default).
    Uvicorn's own messages go the same way; per-request lines (uvicorn's access
    log, httpx's requests to the upstreams, aioice's connection checks) only from
    DEBUG on, as the page polls the health endpoint.
    """
    logger.remove()
    logger.add(sys.stderr, level=level, format="{level: <7} {name}: {message}")
    logging.basicConfig(level=level, format="%(levelname)-7s %(name)s: %(message)s")
    if level not in ("TRACE", "DEBUG"):
        for name in ("uvicorn.access", "httpx", "httpcore", "httpx2", "httpcore2", "aioice"):
            logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    config = Config.from_env()
    setup_logging(config.log_level)
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN,
        log_config=None,
    )


if __name__ == "__main__":
    main()
