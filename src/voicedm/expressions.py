"""Prepare a reply for speech: the actions a model writes between asterisks.

Many models write actions such as ``*laughs*`` or ``*looks at you with a smile*``
unasked, and a TTS engine reads them aloud as words. ``speech_text`` rewrites the text
sent to Chatterbox, on request in two ways:

- ``sounds``: Chatterbox's Turbo model voices a handful of tags, such as ``[laugh]``,
  as the sound itself. An action that names one of those sounds becomes its tag.
- ``silent``: other actions are left out, so they are not read aloud.

Text between asterisks is not always an action: ``I *never* said that`` is emphasis and
has to be spoken. Whatever is not recognised as an action is left exactly as it is.
"""

from __future__ import annotations

import re

# With actions read aloud, one longer than this many words says more than a sound and
# is left alone. With actions silent, any action that names a sound becomes the sound.
MAX_SOUND_WORDS = 4
# Text between asterisks of at least this many words is narration, not emphasis.
MIN_NARRATION_WORDS = 4

# One span between single asterisks, on one line and not inside a word or a formula:
# ``*laughs*`` but not ``**bold**``, ``2*3*4`` or ``* a list item``.
ACTION = re.compile(r"(?<![*\w])\*(?![\s*])([^*\n]*?)(?<!\s)\*(?![*\w])")
# A word, with its inner apostrophes and hyphens: ``don't`` and ``well-known`` are one.
WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
SHH = re.compile(r"sh{2,}")
VOWELS = "aeiou"
PUNCTUATION = ".,!?;:"


def third_person(verb: str) -> str:
    if verb.endswith(("sh", "ch", "s", "x", "z")):
        return verb + "es"
    if verb.endswith("y") and verb[-2] not in VOWELS:
        return verb[:-1] + "ies"
    return verb + "s"


def stems(verb: str) -> set[str]:
    """What ``ing`` and ``ed`` attach to: ``smil``, and both ``nod`` and ``nodd``.

    Whether a final consonant doubles depends on stress (``nodding``, ``whispering``);
    offering both spellings is simpler, and the wrong one is not a word anyone writes.
    """
    found = {verb.removesuffix("e")}
    ends_consonant_vowel_consonant = (
        len(verb) >= 3
        and verb[-1] not in VOWELS + "wxy"
        and verb[-2] in VOWELS
        and verb[-3] not in VOWELS
    )
    if ends_consonant_vowel_consonant:
        found.add(verb + verb[-1])
    return found


def progressive(verb: str) -> set[str]:
    return {stem + "ing" for stem in stems(verb)}


def verb_forms(verb: str) -> set[str]:
    past = {stem + "ed" for stem in stems(verb)}
    return {verb, third_person(verb)} | progressive(verb) | past


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

# Verbs that short actions are made of: ``*smiles*``, ``*nods slowly*``, ``*looks away*``.
# A short span without one of them is taken for emphasis and spoken.
ACTION_VERBS = """
    adjust beam bite blink blush bounce bow chuckle clap close cough cross cuddle fidget
    fold frown gasp gaze gesture giggle glance glare grab grin groan growl grumble gulp
    hesitate hiss hold hop hug hum jump kiss laugh lean lick look moan mumble murmur
    mutter narrow nod nuzzle pat pause point ponder pout pull purr push raise reach roll
    rub scoff scratch shake shiver shrug shush sigh sit smile smirk sniff sniffle snicker
    snort snuggle sob spin squeal stand stare step stretch stroke tap tease tilt touch
    tremble turn twirl walk wave whimper whisper whistle widen wince wink yawn
""".split()  # noqa: SIM905 (a word list reads better than a list literal)
BASE = set(ACTION_VERBS)
THIRD_PERSON = {third_person(verb) for verb in ACTION_VERBS}
PROGRESSIVE = {form for verb in ACTION_VERBS for form in progressive(verb)}


def words_of(action: str) -> list[str]:
    return WORD.findall(action.lower())


def sound_tags(action: str, limit: int | None = MAX_SOUND_WORDS) -> list[str]:
    """The tags for the sounds an action names, in the order it names them."""
    words = words_of(action)
    if not words or (limit is not None and len(words) > limit):
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


def is_action(span: str, before: str, after: str) -> bool:
    """Whether the text between two asterisks is an action rather than emphasis.

    ``before`` and ``after`` are the text around the span. A single word is the hard
    case: in ``that *looks* great`` and ``*Look* at me`` it is part of the sentence.
    """
    words = words_of(span)
    if len(words) >= MIN_NARRATION_WORDS:
        return True
    if len(words) > 1:
        return any(word in THIRD_PERSON or word in PROGRESSIVE for word in words)
    if not words:
        return False
    follows_a_word, sentence_goes_on = surroundings(before, after)
    if words[0] in THIRD_PERSON:
        return not (follows_a_word and sentence_goes_on)
    if words[0] in BASE or words[0] in PROGRESSIVE:
        return not follows_a_word and not sentence_goes_on
    return False


def surroundings(before: str, after: str) -> tuple[bool, bool]:
    """Whether a span follows a word, and whether the sentence goes on after it."""
    return before.rstrip()[-1:].isalnum(), after.lstrip()[:1].islower()


def is_sound(span: str, before: str, after: str) -> bool:
    """Whether a span that names a sound is the sound, not a word of the sentence
    (``Don't *laugh* at me``)."""
    return len(words_of(span)) > 1 or not all(surroundings(before, after))


def speech_text(text: str, *, sounds: bool = False, silent: bool = False) -> str:
    """``text`` as it should be sent for speech.

    With ``sounds``, each action that names a Chatterbox sound is replaced by its tag;
    with ``silent``, the other actions are removed. The rest of the text is unchanged.
    """
    result = ""
    position = 0
    for match in ACTION.finditer(text):
        span, around = match.group(1), (text[: match.start()], text[match.end() :])
        tags: list[str] = []
        if sounds and is_sound(span, *around):
            tags = sound_tags(span, None if silent else MAX_SOUND_WORDS)
        remove = not tags and silent and is_action(span, *around)
        if not tags and not remove:
            continue
        result += text[position : match.start()]
        position = match.end()
        if tags:
            result += " ".join(f"[{tag}]" for tag in tags)
            continue
        # The action goes together with the space after it.
        while text[position : position + 1] in (" ", "\t"):
            position += 1
        if text[position : position + 1] in tuple(PUNCTUATION):
            head = result.rstrip()
            if head and head[-1] not in PUNCTUATION:
                result = head  # "Fine *shrugs*." is "Fine."
            else:
                # The punctuation ended the action: "*smiles*. Hello" is "Hello".
                while text[position : position + 1] in (*PUNCTUATION, " ", "\t"):
                    position += 1
    return result + text[position:]


def has_speech(text: str) -> bool:
    """Whether there is anything in ``text`` to say."""
    return any(character.isalnum() for character in text)
