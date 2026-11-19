"""Turn the actions a model writes between asterisks into Chatterbox's sound tags.

Chatterbox's Turbo model voices a handful of tags, such as ``[laugh]``, as the sound
itself. Models rarely write those tags when told to, but many write actions such as
``*laughs*`` unasked, and a TTS engine reads those aloud as words. ``with_sound_tags``
rewrites the actions that match one of the tags, for the text sent to Chatterbox only.
Everything else, including actions with no matching tag, is left exactly as it is.
"""

from __future__ import annotations

import re

# An action longer than this many words says more than a sound, so it is left alone.
MAX_ACTION_WORDS = 4

# One span between single asterisks, on one line and not inside a word or a formula:
# ``*laughs*`` but not ``**bold**``, ``2*3*4`` or ``* a list item``.
ACTION = re.compile(r"(?<![*\w])\*(?![\s*])([^*\n]*?)(?<!\s)\*(?![*\w])")
WORD = re.compile(r"[a-z]+")
SHH = re.compile(r"sh{2,}")


def verb_forms(verb: str) -> set[str]:
    stem = verb.removesuffix("e")
    return {verb, verb + ("es" if verb.endswith("sh") else "s"), stem + "ing", stem + "ed"}


# The words that stand for each of Chatterbox's tags.
SOUNDS = {
    "laugh": verb_forms("laugh") | verb_forms("giggle") | {"laughter"},
    "chuckle": verb_forms("chuckle") | verb_forms("snicker"),
    "sigh": verb_forms("sigh"),
    "gasp": verb_forms("gasp"),
    "cough": verb_forms("cough"),
    "sniff": verb_forms("sniff") | verb_forms("sniffle"),
    "groan": verb_forms("groan") | verb_forms("moan"),
    "shush": verb_forms("shush"),
}
TAG_OF = {word: tag for tag, words in SOUNDS.items() for word in words}
CLEAR = verb_forms("clear")


def sound_tags(action: str) -> list[str]:
    """The tags for the sounds an action names, in the order it names them."""
    words = WORD.findall(action.lower())
    if not words or len(words) > MAX_ACTION_WORDS:
        return []
    tags: list[str] = []
    for position, word in enumerate(words):
        tag = TAG_OF.get(word)
        if tag is None and SHH.fullmatch(word):
            tag = "shush"
        if tag is None and word in CLEAR and "throat" in words[position + 1 :]:
            tag = "clear throat"
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def with_sound_tags(text: str) -> str:
    """``text`` with each action that names a Chatterbox sound replaced by its tag."""

    def replace(match: re.Match[str]) -> str:
        tags = sound_tags(match.group(1))
        return " ".join(f"[{tag}]" for tag in tags) if tags else match.group(0)

    return ACTION.sub(replace, text)
