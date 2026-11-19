"""Test doubles shared by the test modules."""

import asyncio


class FakeBot:
    """Stands in for the pipeline: records calls and stays 'in a call' until cancelled."""

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[tuple[str, str, str]] = []

    async def load(self) -> None:
        self.loaded = True

    async def run(self, connection, upstreams, model: str, voice: str) -> None:
        self.calls.append((connection.pc_id, model, voice))
        await asyncio.Event().wait()
