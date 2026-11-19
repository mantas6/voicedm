"""Entry point: ``voicedm`` or ``python -m voicedm``."""

from __future__ import annotations

import uvicorn

from voicedm.app import create_app
from voicedm.config import Config


def main() -> None:
    config = Config.from_env()
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
