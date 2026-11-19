"""Building the LLM context for a call from a session's stored messages.

For now the stored history is replayed as it is. The memory milestone replaces this
with the cache-friendly layout from the plan: a summary of older messages followed by
the recent ones.
"""

from __future__ import annotations


def build_context(session: dict, messages: list[dict]) -> list[dict[str, str]]:
    """The chat messages a call on ``session`` starts from.

    The system prompt is not part of them: it is the LLM service's
    ``system_instruction`` setting. An interrupted reply is replayed as far as it was
    spoken, which is all that is stored of it.
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant") and m["content"].strip()
    ]
